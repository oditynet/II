#!/usr/bin/env python3
"""
Добавление изображений в существующий датасет
Раскомментируй нужные параметры - только они будут добавлены
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import sys
import cv2
from tqdm import tqdm
import io
from datetime import datetime

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

# ============================================
# Параметры - раскомментируй нужные!
# ============================================

# Параметры фона
BACKGROUND_VALUES = [255, 210]
# BACKGROUND_VALUES = [255]  # Только белый фон
# BACKGROUND_VALUES = [180, 200, 220]  # Разные оттенки серого

# Параметры текста
TEXT_VALUES = [0, 60, 90]
# TEXT_VALUES = [0]  # Только черный текст
# TEXT_VALUES = [0, 30, 60, 90, 120]  # Разные оттенки

# Геометрические искажения
ROTATION_VALUES = [-3, 0, 3]
# ROTATION_VALUES = [0]  # Без поворота
# ROTATION_VALUES = [-5, -3, 0, 3, 5]  # Сильные повороты

PERSPECTIVE_VALUES = [0]
# PERSPECTIVE_VALUES = [0, 2, 4]  # С перспективой

SHIFT_VALUES = [-2, 0, 2]
# SHIFT_VALUES = [0]  # Без сдвига
# SHIFT_VALUES = [-4, -2, 0, 2, 4]  # Сильные сдвиги

# Размытие
BLUR_VALUES = [0.3,0.7]
# BLUR_VALUES = [0]  # Без размытия
# BLUR_VALUES = [1, 2, 3]  # Сильное размытие

# Гауссов шум
NOISE_STD_VALUES = [0, 3]
# NOISE_STD_VALUES = [0, 5, 10]  # С шумом
# NOISE_STD_VALUES = [0, 3, 5, 10, 15]  # Разный шум

# JPEG сжатие
JPEG_VALUES = [95, 80]
# JPEG_VALUES = [100]  # Без сжатия
# JPEG_VALUES = [100, 80, 60, 40]  # Разное сжатие

# Сжатие PNG при сохранении
PNG_COMPRESSION = 3

# ============================================
# СИМВОЛЫ - раскомментируй нужные!
# ============================================
CHARS = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

# CHARS = list("VvWw")  # Только проблемные буквы
# CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Только заглавные
# CHARS = list("0123456789")  # Только цифры

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

def apply_perspective(img, angle_deg):
    """Имитация перспективы"""
    if angle_deg == 0:
        return img
    
    h, w = img.shape[:2]
    offset = int(np.tan(np.radians(abs(angle_deg))) * w * 0.3)
    
    if angle_deg > 0:
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = np.float32([[offset, 0], [w - offset, 0], [0, h], [w, h]])
    else:
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = np.float32([[0, offset], [w, offset], [0, h - offset], [w, h - offset]])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def add_gaussian_noise(img, std):
    """Добавление гауссова шума"""
    if std == 0:
        return img
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def apply_jpeg_compression(img, quality):
    """Применение JPEG сжатия"""
    if quality >= 100:
        return img
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def create_filename(char, font_name, font_size, bg_int, text_int,
                    shift_x, shift_y, rotation, perspective,
                    blur, noise_std, jpeg_qual, timestamp):
    """Создание имени файла с временной меткой"""
    safe = safe_folder(char)
    return (f"{safe}_"
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
            f"T{timestamp}"
            f".png")

def main():
    print("="*60)
    print("  ДОБАВЛЕНИЕ ИЗОБРАЖЕНИЙ В СУЩЕСТВУЮЩИЙ ДАТАСЕТ")
    print("="*60)
    
    # Проверяем существование датасета
    if not OUTPUT.exists():
        print(f"❌ Папка {OUTPUT} не существует!")
        print("   Сначала создайте датасет основным скриптом")
        sys.exit(1)
    
    # Поиск шрифтов
    print("\n🔍 Поиск шрифтов...")
    fonts_dir = Path("font")
    if not fonts_dir.exists():
        print(f"❌ Папка {fonts_dir.absolute()} не найдена!")
        sys.exit(1)
    
    fonts = []
    for ext in ["*.ttf", "*.TTF", "*.otf", "*.OTF"]:
        fonts.extend([str(f) for f in fonts_dir.rglob(ext)])
    
    fonts = sorted(set(fonts))
    
    # Проверка валидности
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
    
    # Расчет количества
    font_sizes = list(range(MIN_FONT_SIZE, MAX_FONT_SIZE + 1, FONT_SIZE_STEP))
    
    total = (len(CHARS) * len(fonts) * len(font_sizes) * 
             len(BACKGROUND_VALUES) * len(TEXT_VALUES) * 
             len(SHIFT_VALUES) * len(SHIFT_VALUES) *
             len(ROTATION_VALUES) * len(PERSPECTIVE_VALUES) * 
             len(BLUR_VALUES) * len(NOISE_STD_VALUES) * 
             len(JPEG_VALUES) * SAMPLES_PER_COMBO)
    
    print(f"\n📊 Будет добавлено:")
    print(f"   Символов: {len(CHARS)}")
    print(f"   Шрифтов: {len(fonts)}")
    print(f"   Размеров шрифта: {len(font_sizes)}")
    print(f"   Изображений всего: {total:,}")
    
    # Показываем активные параметры
    print(f"\n🎨 Активные параметры:")
    print(f"   Фон: {BACKGROUND_VALUES}")
    print(f"   Текст: {TEXT_VALUES}")
    print(f"   Поворот: {ROTATION_VALUES}")
    print(f"   Перспектива: {PERSPECTIVE_VALUES}")
    print(f"   Сдвиг: {SHIFT_VALUES}")
    print(f"   Размытие: {BLUR_VALUES}")
    print(f"   Шум: {NOISE_STD_VALUES}")
    print(f"   JPEG: {JPEG_VALUES}")
    
    # Подтверждение
    if input(f"\n▶️  Продолжить? (y/n): ").lower() != 'y':
        print("Отменено")
        sys.exit(0)
    
    # Создаем недостающие папки
    print("\n📁 Проверка папок...")
    for char in CHARS:
        char_dir = OUTPUT / safe_folder(char)
        char_dir.mkdir(exist_ok=True)
    
    # Генерация
    print("\n🎨 Генерация изображений...")
    timestamp = datetime.now().strftime("%H%M%S")
    generated = 0
    skipped = 0
    
    pbar = tqdm(total=total, desc="   Прогресс")
    
    try:
        for char in CHARS:
            char_dir = OUTPUT / safe_folder(char)
            
            for font_path in fonts:
                font_name = Path(font_path).stem
                
                for font_size in font_sizes:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                    except:
                        skipped += 1
                        pbar.update(1)
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
                                                            
                                                            # Генерация
                                                            bg_color = (bg_val,) * 3
                                                            temp_size = IMAGE_SIZE * 4
                                                            temp_img = Image.new('RGB', (temp_size, temp_size), bg_color)
                                                            temp_draw = ImageDraw.Draw(temp_img)
                                                            
                                                            # Рисуем символ
                                                            bbox = temp_draw.textbbox((0, 0), char, font=font)
                                                            tw = bbox[2] - bbox[0]
                                                            th = bbox[3] - bbox[1]
                                                            
                                                            if tw <= 0 or th <= 0:
                                                                skipped += 1
                                                                pbar.update(1)
                                                                continue
                                                            
                                                            x = (temp_size - tw) // 2 - bbox[0]
                                                            y = (temp_size - th) // 2 - bbox[1]
                                                            
                                                            text_color = (text_val,) * 3
                                                            temp_draw.text((x, y), char, fill=text_color, font=font)
                                                            
                                                            # Поворот
                                                            if rotation != 0:
                                                                temp_img = temp_img.rotate(rotation, expand=False, fillcolor=bg_color)
                                                            
                                                            # Уменьшаем
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
                                                                                        borderValue=(bg_val,) * 3)
                                                            
                                                            # Размытие
                                                            if blur_radius > 0:
                                                                img_pil = Image.fromarray(img_np)
                                                                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                                                                img_np = np.array(img_pil)
                                                            
                                                            # Шум
                                                            if noise_std > 0:
                                                                img_np = add_gaussian_noise(img_np, noise_std)
                                                            
                                                            # JPEG сжатие
                                                            if jpeg_qual < 100:
                                                                img_np = apply_jpeg_compression(img_np, jpeg_qual)
                                                            
                                                            # Сохранение
                                                            filename = create_filename(
                                                                char, font_name, font_size, bg_val, text_val,
                                                                shift_x, shift_y, rotation, perspective,
                                                                blur_radius, noise_std, jpeg_qual, timestamp
                                                            )
                                                            
                                                            img_result = Image.fromarray(img_np)
                                                            img_result.save(char_dir / filename, compress_level=PNG_COMPRESSION)
                                                            
                                                            generated += 1
                                                            pbar.update(1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Прервано пользователем")
    
    finally:
        pbar.close()
    
    # Статистика
    print(f"\n{'='*60}")
    print(f"  ✅ ДОБАВЛЕНИЕ ЗАВЕРШЕНО!")
    print(f"  Создано: {generated:,} изображений")
    print(f"  Пропущено: {skipped:,}")
    print(f"  Папка: {OUTPUT.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
