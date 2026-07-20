#!/usr/bin/env python3
"""
Генератор ПРОБЕЛОВ из скриншотов
Пробел = расстояние между буквами > 64px
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
import uuid

# ============================================
# НАСТРОЙКИ
# ============================================
PAD_PIXELS = 1
ARTIFACT_TOLERANCE = 0.08
SPLIT_MERGE_THRESHOLD = 0.3
OUTPUT_SIZE = 64

# Параметры пробела
SPACE_THRESHOLD_PX = 64     # Расстояние > 64px → пробел
WHITE_THRESHOLD = 250       # Порог белого пикселя
WHITE_MIN_PERCENT = 90      # Минимальный % белых пикселей в промежутке

# ============================================
# ПУТИ
# ============================================


INPUT_DIR = "nonchar"
SPACE_DIR = "dataset/ocr_dataset_v3/SPACE"
DEBUG_DIR = "debug_space_generator"

os.makedirs(SPACE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

print("="*70)
print(f"  ГЕНЕРАТОР ПРОБЕЛОВ (расстояние > {SPACE_THRESHOLD_PX}px)")
print("="*70)

# ============================================
# ФУНКЦИИ
# ============================================
def load_and_preprocess(image_path):
    """Загрузка скриншота"""
    print(f"\n📁 {Path(image_path).name}")
    
    img = Image.open(image_path)
    img_gray = img.convert('L')
    arr = np.array(img_gray)
    
    print(f"   Размер: {img.size}, Яркость: {arr.mean():.0f}")
    
    if arr.mean() < 128:
        print(f"   🔄 Инвертирую (тёмный фон)")
        img_gray = ImageOps.invert(img_gray)
        arr = np.array(img_gray)
    
    return img_gray, arr

def find_lines(arr):
    """Поиск строк"""
    H, W = arr.shape
    dark_per_row = (arr < 128).sum(axis=1)
    threshold = W * 0.003
    text_rows = dark_per_row > threshold
    
    lines = []
    in_line = False
    start = 0
    
    for y in range(H):
        if text_rows[y] and not in_line:
            start = y
            in_line = True
        elif not text_rows[y] and in_line:
            if y - start > 3:
                lines.append((start, y))
            in_line = False
    
    if in_line and H - start > 3:
        lines.append((start, H))
    
    print(f"   Строк: {len(lines)}")
    return lines

def find_chars_in_line(line_arr):
    """Находит буквы в строке"""
    H, W = line_arr.shape
    has_dark = (line_arr < 128).any(axis=0)
    
    chars = []
    x = 0
    
    while x < W:
        while x < W and not has_dark[x]:
            x += 1
        
        if x >= W:
            break
        
        start = x
        
        dark_heights = []
        while x < W and has_dark[x]:
            dark_heights.append((line_arr[:, x] < 128).sum())
            x += 1
        
        end = x - 1
        
        if dark_heights:
            max_height = max(dark_heights)
            artifact_threshold = max(1, int(max_height * ARTIFACT_TOLERANCE))
        else:
            artifact_threshold = 1
        
        while x < W:
            dark_count = (line_arr[:, x] < 128).sum()
            if dark_count == 0:
                break
            elif dark_count <= artifact_threshold:
                end = x
                x += 1
            else:
                break
        
        if end - start >= 2:
            col_heights = [(line_arr[:, col] < 128).sum() for col in range(start, end + 1)]
            max_h = max(col_heights)
            min_h = min(col_heights)
            
            if min_h < max_h * SPLIT_MERGE_THRESHOLD and (end - start) > 6:
                split_col = start + col_heights.index(min_h)
                
                if split_col - start >= 2:
                    chars.append((start, split_col - 1))
                if end - split_col >= 2:
                    chars.append((split_col + 1, end))
            else:
                chars.append((start, end))
    
    return chars

def count_spaces_between(chars, i, line_arr):
    """
    Определяет сколько пробелов между chars[i] и chars[i+1]
    Возвращает (количество_пробелов, ширина_промежутка)
    """
    current_end = chars[i][1]
    next_start = chars[i+1][0]
    gap = next_start - current_end
    
    if gap <= 0:
        return 0, 0
    
    # Средняя ширина букв вокруг
    current_width = chars[i][1] - chars[i][0]
    next_width = chars[i+1][1] - chars[i+1][0]
    avg_char_width = (current_width + next_width) / 2
    
    # Проверяем, что в промежутке достаточно белого
    H = line_arr.shape[0]
    region = line_arr[:, current_end:next_start]
    white_pixels = (region >= WHITE_THRESHOLD).sum()
    total_pixels = region.size
    white_percent = 100 * white_pixels / total_pixels if total_pixels > 0 else 0
    
    # Если недостаточно белого - это не пробел
    if white_percent < WHITE_MIN_PERCENT:
        return 0, gap
    
    # Считаем количество пробелов
    # 1 пробел = расстояние примерно равное ширине буквы
    # Если gap > SPACE_THRESHOLD_PX → это пробел(ы)
    
    if gap > SPACE_THRESHOLD_PX:
        # Вычисляем количество пробелов
        # Например: gap=200px, avg_char_width=50px → 200/50 = 4 пробела
        num_spaces = max(1, round(gap / avg_char_width))
        return num_spaces, gap
    
    return 0, gap

def create_space_image():
    """Создаёт изображение пробела 64×64 (белое)"""
    # Абсолютно белое изображение
    img = np.full((OUTPUT_SIZE, OUTPUT_SIZE), 255, dtype=np.uint8)
    
    # Добавляем минимальный шум (1-2 пикселя) для реалистичности
    num_noise = np.random.randint(0, 3)
    if num_noise > 0:
        y_coords = np.random.randint(0, OUTPUT_SIZE, num_noise)
        x_coords = np.random.randint(0, OUTPUT_SIZE, num_noise)
        noise_values = np.random.randint(250, 256, num_noise)
        img[y_coords, x_coords] = noise_values
    
    return img

def debug_save_line(line_arr, chars, spaces_info, line_idx, file_prefix):
    """Сохраняет дебаг строки"""
    debug_img = Image.fromarray(line_arr).convert('RGB')
    draw = ImageDraw.Draw(debug_img)
    H = debug_img.height
    
    # Буквы - зелёным
    for x1, x2 in chars:
        draw.rectangle([x1, 0, x2, H-1], outline=(0, 255, 0), width=1)
    
    # Пробелы - оранжевым с номером
    for gap_start, gap_end, num_spaces in spaces_info:
        draw.rectangle([gap_start, 0, gap_end, H-1], outline=(255, 165, 0), width=2)
        draw.text((gap_start, 2), f"SP×{num_spaces}", fill=(255, 165, 0))
    
    debug_path = os.path.join(DEBUG_DIR, f"{file_prefix}_line{line_idx:02d}.png")
    debug_img.save(debug_path)

# ============================================
# ОБРАБОТКА СКРИНШОТА
# ============================================
def process_screenshot(image_path):
    """Обрабатывает скриншот, находит и сохраняет пробелы"""
    file_prefix = Path(image_path).stem.replace(' ', '_')
    
    # Загрузка
    img_gray, arr = load_and_preprocess(image_path)
    
    # Поиск строк
    lines = find_lines(arr)
    
    if not lines:
        return 0, 0
    
    total_chars = 0
    total_spaces = 0
    
    # Обрабатываем каждую строку
    for line_idx, (y1, y2) in enumerate(lines):
        line_arr = arr[y1:y2, :]
        
        # Находим буквы
        chars = find_chars_in_line(line_arr)
        
        if len(chars) < 2:
            total_chars += len(chars)
            continue
        
        total_chars += len(chars)
        
        print(f"\n   Строка {line_idx+1}: {len(chars)} букв")
        
        spaces_info = []  # Для дебага
        
        # Проверяем промежутки между буквами
        for i in range(len(chars) - 1):
            num_spaces, gap = count_spaces_between(chars, i, line_arr)
            
            if num_spaces > 0:
                gap_start = chars[i][1]
                gap_end = chars[i+1][0]
                
                spaces_info.append((gap_start, gap_end, num_spaces))
                
                # Сохраняем каждый пробел
                for s in range(num_spaces):
                    space_img = create_space_image()
                    
                    filename = f"SPACE_{file_prefix}_L{line_idx}_gap{gap}px_{uuid.uuid4().hex[:8]}.png"
                    filepath = os.path.join(SPACE_DIR, filename)
                    Image.fromarray(space_img).save(filepath)
                    total_spaces += 1
                
                print(f"      ✓ ПРОБЕЛОВ×{num_spaces}: x={gap_start}-{gap_end} "
                      f"(gap={gap}px, буквы={chars[i][1]-chars[i][0]}px + {chars[i+1][1]-chars[i+1][0]}px)")
        
        # Дебаг
        if spaces_info:
            debug_save_line(line_arr, chars, spaces_info, line_idx, file_prefix)
    
    return total_chars, total_spaces

# ============================================
# ГЛАВНАЯ
# ============================================
def main():
    print(f"\n📂 Поиск скриншотов в: {INPUT_DIR}/")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Папка не найдена: {INPUT_DIR}")
        print(f"   Создайте папку 'screenshots' и положите туда скриншоты")
        return
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = []
    
    for file in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, file)
        if os.path.isfile(file_path):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file_path)
    
    if not image_files:
        print(f"❌ Нет изображений")
        return
    
    print(f"   Найдено: {len(image_files)}")
    
    total_all_chars = 0
    total_all_spaces = 0
    
    for image_path in sorted(image_files):
        try:
            chars, spaces = process_screenshot(image_path)
            total_all_chars += chars
            total_all_spaces += spaces
            print(f"\n   ✅ Букв: {chars}, Пробелов: {spaces}")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    # Итоги
    print(f"\n{'='*70}")
    print(f"  ИТОГИ")
    print(f"{'='*70}")
    print(f"  Обработано скриншотов: {len(image_files)}")
    print(f"  Найдено букв:          {total_all_chars:,}")
    print(f"  Найдено пробелов:      {total_all_spaces:,}")
    print(f"  Сохранено в:           {SPACE_DIR}/")
    print(f"  Дебаг:                 {DEBUG_DIR}/")
    
    # Проверяем пробелы
    space_files = list(Path(SPACE_DIR).glob("*.png"))
    if space_files:
        print(f"\n  Всего пробелов в датасете: {len(space_files):,}")
        
        # Показываем пример
        sample = space_files[0]
        img = Image.open(sample)
        arr = np.array(img)
        white = (arr >= WHITE_THRESHOLD).sum()
        print(f"\n  Пример пробела: {sample.name}")
        print(f"    Размер: {img.size}")
        print(f"    Белых пикселей: {100*white/arr.size:.1f}%")
        print(f"    Мин/Макс значение: {arr.min()}/{arr.max()}")

if __name__ == "__main__":
    main()
