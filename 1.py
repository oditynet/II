#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import os
import sys
import cv2
from tqdm import tqdm
import io
import multiprocessing as mp
from multiprocessing import Queue, Process
import time
import shutil

# ============================================
# НАСТРОЙКИ
# ============================================
OUTPUT = Path("dataset/ocr_dataset_v3")
IMAGE_SIZE = 64

# Размеры шрифта
MIN_FONT_SIZE = 190
MAX_FONT_SIZE = 241
FONT_SIZE_STEP = 20

# Количество сэмплов на каждую комбинацию
SAMPLES_PER_COMBO = 1

# Параметры фона
BACKGROUND_VALUES = [255,  210]

# Параметры текста
TEXT_VALUES = [0, 60, 90]

# Геометрические искажения
ROTATION_VALUES = [-3, 0, 3]
PERSPECTIVE_VALUES = [0]
SHIFT_VALUES = [-2, 0, 2]

# Размытие
BLUR_VALUES = [0, 2]

# Гауссов шум
NOISE_STD_VALUES = [0]

# JPEG сжатие
JPEG_VALUES = [100, 60]

# Сжатие PNG при сохранении
PNG_COMPRESSION = 3

# Количество процессов
NUM_PROCESSES = 32

# Размер очереди
QUEUE_SIZE = 100000

# ============================================
# СИМВОЛЫ
# ============================================
CHARS = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)

def safe_folder(char):
    """Безопасное имя папки для спецсимволов"""
    mapping = {
        ' ': 'SPACE', '!': 'EXCL', '"': 'DQUOTE', '#': 'HASH',
        '$': 'DOLLAR', '%': 'PERCENT', '&': 'AMP', "'": 'SQUOTE',
        '(': 'LPAREN', ')': 'RPAREN', '*': 'ASTERISK', '+': 'PLUS',
        ',': 'COMMA', '-': 'MINUS', '.': 'DOT', '/': 'SLASH',
        ':': 'COLON', ';': 'SEMICOLON', '<': 'LT', '=': 'EQUAL',
        '>': 'GT', '?': 'QUEST', '@': 'AT', '[': 'LBRACKET',
        '\\': 'BACKSLASH', ']': 'RBRACKET', '^': 'CARET',
        '_': 'UNDERSCORE', '`': 'BACKTICK', '{': 'LBRACE',
        '|': 'PIPE', '}': 'RBRACE', '~': 'TILDE',
        'Ё': 'YO', 'ё': 'yo',
    }
    return mapping.get(char, char)

# ============================================
# ФУНКЦИИ ГЕНЕРАЦИИ (для процессов)
# ============================================
def apply_perspective(img, angle_deg):
    """Имитация перспективы с поворотом на angle_deg"""
    if angle_deg == 0:
        return img
    
    h, w = img.shape[:2]
    offset = int(np.tan(np.radians(abs(angle_deg))) * w * 0.3)
    
    if angle_deg > 0:
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = np.float32([
            [offset, 0], [w - offset, 0],
            [0, h], [w, h]
        ])
    else:
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = np.float32([
            [0, offset], [w, offset],
            [0, h - offset], [w, h - offset]
        ])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    return result

def add_gaussian_noise(img, std):
    """Добавление гауссова шума"""
    if std == 0:
        return img
    
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_jpeg_compression(img, quality):
    """Применение JPEG сжатия"""
    if quality >= 100:
        return img
    
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    return np.array(compressed)

def generate_sample(char, font_path, font_size, 
                    bg_intensity, text_intensity,
                    shift_x, shift_y, rotation, perspective,
                    blur_radius, noise_std, jpeg_quality):
    """Генерация одного изображения со всеми параметрами"""
    
    # Загружаем шрифт внутри процесса
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        return None
    
    bg_color = (bg_intensity, bg_intensity, bg_intensity)
    
    # Создаем временное большое изображение
    temp_size = IMAGE_SIZE * 4
    temp_img = Image.new('RGB', (temp_size, temp_size), bg_color)
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Рисуем символ
    bbox = temp_draw.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    if tw <= 0 or th <= 0:
        return None
    
    # Центрируем
    x = (temp_size - tw) // 2 - bbox[0]
    y = (temp_size - th) // 2 - bbox[1]
    
    text_color = (text_intensity, text_intensity, text_intensity)
    temp_draw.text((x, y), char, fill=text_color, font=font)
    
    # Поворот
    if rotation != 0:
        temp_img = temp_img.rotate(rotation, expand=False, fillcolor=bg_color)
    
    # Уменьшаем до целевого размера
    img = temp_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img_np = np.array(img)
    
    # Перспектива
    if perspective != 0:
        img_np = apply_perspective(img_np, perspective)
    
    # Сдвиг
    if shift_x != 0 or shift_y != 0:
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img_np = cv2.warpAffine(img_np, matrix, (IMAGE_SIZE, IMAGE_SIZE), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(bg_intensity, bg_intensity, bg_intensity))
    
    # Размытие
    if blur_radius > 0:
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img_np = np.array(img_pil)
    
    # Шум
    if noise_std > 0:
        img_np = add_gaussian_noise(img_np, noise_std)
    
    # JPEG сжатие
    if jpeg_quality < 100:
        img_np = apply_jpeg_compression(img_np, jpeg_quality)
    
    return Image.fromarray(img_np)

def create_filename(char, font_name, font_size, bg_int, text_int,
                    shift_x, shift_y, rotation, perspective,
                    blur, noise_std, jpeg_qual, sample_id):
    """Создание информативного имени файла"""
    
    safe = safe_folder(char)
    
    fname = (
        f"{safe}_"
        f"F{font_name[:20]}_"
        f"S{font_size:02d}_"
        f"BG{bg_int:03d}_"
        f"TXT{text_int:03d}_"
        f"X{shift_x:+03d}_"
        f"Y{shift_y:+03d}_"
        f"R{rotation:+04.1f}_"
        f"P{perspective:+04.1f}_"
        f"B{blur:.1f}_"
        f"N{noise_std:.1f}_"
        f"J{jpeg_qual:02d}_"
        f"ID{sample_id:06d}"
        f".png"
    )
    
    return fname

# ============================================
# ФУНКЦИЯ ВОРКЕРА
# ============================================
def worker_process(task_queue, result_queue, output_dir, fonts_list):
    """Процесс-воркер для генерации изображений"""
    
    # Создаем папки для символов
    char_dirs = {}
    for char in CHARS:
        char_dir = Path(output_dir) / safe_folder(char)
        char_dir.mkdir(exist_ok=True)
        char_dirs[char] = char_dir
    
    # Кэш имен шрифтов
    font_names = {f: Path(f).stem for f in fonts_list}
    
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # Сигнал остановки
                break
        except:
            continue
        
        try:
            (char, font_path, font_size, bg_val, text_val,
             shift_x, shift_y, rotation, perspective,
             blur_radius, noise_std, jpeg_qual, sample_id) = task
            
            # Генерация изображения
            img = generate_sample(
                char, font_path, font_size,
                bg_val, text_val,
                shift_x, shift_y,
                rotation, perspective,
                blur_radius, noise_std, jpeg_qual
            )
            
            if img is None:
                result_queue.put((None, True))
                continue
            
            # Создание имени файла и сохранение
            font_name = font_names[font_path]
            filename = create_filename(
                char, font_name, font_size,
                bg_val, text_val,
                shift_x, shift_y,
                rotation, perspective,
                blur_radius, noise_std, jpeg_qual,
                sample_id
            )
            
            char_dir = char_dirs[char]
            img.save(char_dir / filename, compress_level=PNG_COMPRESSION)
            result_queue.put((filename, False))
            
        except Exception as e:
            result_queue.put((None, True))

# ============================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================
def main():
    print("="*60)
    print("  ГЕНЕРАЦИЯ ДАТАСЕТА OCR v3 (32 процесса + очередь)")
    print("="*60)
    
    # Поиск шрифтов
    print("\n🔍 Поиск шрифтов в папке 'font'...")
    fonts_dir = Path("font")
    if not fonts_dir.exists():
        print(f"❌ Папка {fonts_dir.absolute()} не найдена!")
        sys.exit(1)
    
    fonts = []
    for f in fonts_dir.rglob("*.[tT][tT][fF]"):
        fonts.append(str(f))
    for f in fonts_dir.rglob("*.[oO][tT][fF]"):
        fonts.append(str(f))
    
    fonts = sorted(set(fonts))
    print(f"   Найдено шрифтов: {len(fonts)}")
    
    if not fonts:
        print("❌ Шрифты не найдены! Поместите .ttf файлы в папку 'font'")
        sys.exit(1)
    
    # Проверяем валидность шрифтов
    print("   Проверка шрифтов...")
    valid_fonts = []
    for f in fonts:
        try:
            ImageFont.truetype(f, 20)
            valid_fonts.append(f)
        except:
            pass
    fonts = valid_fonts
    print(f"   Валидных шрифтов: {len(fonts)}")
    
    if not fonts:
        print("❌ Нет валидных шрифтов!")
        sys.exit(1)
    
    # Параметры для перебора
    font_sizes = list(range(MIN_FONT_SIZE, MAX_FONT_SIZE + 1, FONT_SIZE_STEP))
    
    # Расчет общего количества
    combinations_per_char = (
        len(fonts) * 
        len(font_sizes) * 
        len(BACKGROUND_VALUES) * 
        len(TEXT_VALUES) * 
        len(SHIFT_VALUES) * len(SHIFT_VALUES) *
        len(ROTATION_VALUES) * 
        len(PERSPECTIVE_VALUES) * 
        len(BLUR_VALUES) * 
        len(NOISE_STD_VALUES) * 
        len(JPEG_VALUES) * 
        SAMPLES_PER_COMBO
    )
    
    total_images = len(CHARS) * combinations_per_char
    
    print(f"\n📊 Параметры перебора:")
    print(f"   Символов: {len(CHARS)}")
    print(f"   Шрифтов: {len(fonts)}")
    print(f"   Размеров шрифта: {len(font_sizes)}")
    print(f"   Всего изображений: {total_images:,}")
    print(f"   Процессов: {NUM_PROCESSES}")
    print(f"   Размер очереди: {QUEUE_SIZE:,}")
    
    # Создаем папки
    print("\n📁 Создание папок...")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    
    for char in CHARS:
        (OUTPUT / safe_folder(char)).mkdir()
    
    # Создаем очереди
    task_queue = Queue(maxsize=QUEUE_SIZE)
    result_queue = Queue()
    
    # Запускаем воркеры
    print(f"\n🚀 Запуск {NUM_PROCESSES} процессов...")
    processes = []
    for _ in range(NUM_PROCESSES):
        p = Process(target=worker_process, args=(task_queue, result_queue, str(OUTPUT), fonts))
        p.daemon = True
        p.start()
        processes.append(p)
    
    # Генератор задач
    print("\n🎨 Генерация изображений...")
    sample_id = 0
    total_generated = 0
    total_skipped = 0
    
    pbar = tqdm(total=total_images, desc="   Прогресс")
    
    try:
        # Заполняем очередь задачами
        for char in CHARS:
            for font_path in fonts:
                font_name = Path(font_path).stem
                for font_size in font_sizes:
                    # Проверяем, что шрифт загружается
                    try:
                        ImageFont.truetype(font_path, font_size)
                    except:
                        continue
                    
                    for bg_val in BACKGROUND_VALUES:
                        for text_val in TEXT_VALUES:
                            for shift_x in SHIFT_VALUES:
                                for shift_y in SHIFT_VALUES:
                                    for rotation in ROTATION_VALUES:
                                        for perspective in PERSPECTIVE_VALUES:
                                            for blur_radius in BLUR_VALUES:
                                                for noise_std in NOISE_STD_VALUES:
                                                    for jpeg_qual in JPEG_VALUES:
                                                        for _ in range(SAMPLES_PER_COMBO):
                                                            # Ждем, если очередь почти полная
                                                            while task_queue.qsize() > QUEUE_SIZE * 0.9:
                                                                try:
                                                                    _, skipped = result_queue.get(timeout=0.1)
                                                                    if skipped:
                                                                        total_skipped += 1
                                                                    else:
                                                                        total_generated += 1
                                                                    pbar.update(1)
                                                                except:
                                                                    time.sleep(0.01)
                                                            
                                                            task_queue.put((
                                                                char, font_path, font_size,
                                                                bg_val, text_val,
                                                                shift_x, shift_y,
                                                                rotation, perspective,
                                                                blur_radius, noise_std,
                                                                jpeg_qual, sample_id
                                                            ))
                                                            sample_id += 1
        
        # Отправляем сигнал остановки
        for _ in range(NUM_PROCESSES):
            task_queue.put(None)
        
        # Собираем оставшиеся результаты
        completed = 0
        while completed < total_images:
            try:
                _, skipped = result_queue.get(timeout=0.1)
                if skipped:
                    total_skipped += 1
                else:
                    total_generated += 1
                pbar.update(1)
                completed += 1
            except:
                # Проверяем, живы ли процессы
                alive = any(p.is_alive() for p in processes)
                if not alive and result_queue.empty():
                    break
                continue
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Прервано пользователем")
        
        # Завершаем процессы
        for p in processes:
            p.terminate()
    
    finally:
        pbar.close()
        
        # Ждем завершения процессов
        for p in processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
    
    # Добираем оставшиеся результаты
    while not result_queue.empty():
        try:
            _, skipped = result_queue.get(timeout=0.1)
            if skipped:
                total_skipped += 1
            else:
                total_generated += 1
        except:
            pass
    
    # Итоговая статистика
    size_mb = sum(f.stat().st_size for f in OUTPUT.rglob("*.png")) / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f"  ✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print(f"  Создано: {total_generated:,} изображений")
    print(f"  Пропущено: {total_skipped:,}")
    print(f"  Размер датасета: {size_mb:.1f} МБ")
    print(f"  Папка: {OUTPUT.absolute()}")
    if len(CHARS) > 0 and total_generated > 0:
        print(f"  Комбинаций на символ: {total_generated // len(CHARS):,}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Устанавливаем метод запуска для совместимости
    mp.set_start_method('spawn', force=True)
    main()
