import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
import uuid

# ============================================
# НАСТРОЙКИ
# ============================================
PAD_PIXELS = 1              # Отступ вокруг буквы
ARTIFACT_TOLERANCE = 0.08   # 8% погрешности на артефакты
SPLIT_MERGE_THRESHOLD = 0.3 # Если в узком месте < 30% высоты — разрезать
SHIFT_RATIO = 0.66           # Смещение влево на 50% от размера
OUTPUT_SIZE = 64            # Размер выходного изображения (64x64)
SHIFT_PIXELS = int(OUTPUT_SIZE * SHIFT_RATIO)  # 32 пикселя для 64x64

# ============================================
# ПУТИ
# ============================================
INPUT_DIR = "nonchar"
OUTPUT_DIR = "nonchar/dataset/ocr_dataset_v3"  # Папка внутри nonchar

# Создаём выходную папку
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================
def load_and_preprocess(image_path):
    """Загрузка и предобработка изображения"""
    print(f"\n📁 Файл: {image_path}")
    
    img = Image.open(image_path)
    print(f"   Размер: {img.size}, Режим: {img.mode}")
    
    # Конвертируем в оттенки серого
    img_gray = img.convert('L')
    arr = np.array(img_gray)
    
    print(f"   Средняя яркость: {arr.mean():.0f} (0=чёрный, 255=белый)")
    
    # Инвертируем если тёмный фон
    if arr.mean() < 128:
        print(f"   🔄 Тёмный фон → инвертирую")
        img_gray = ImageOps.invert(img_gray)
        arr = np.array(img_gray)
    
    return img_gray, arr

def find_lines(arr):
    """Поиск строк текста"""
    H, W = arr.shape
    dark_per_row = (arr < 128).sum(axis=1)
    threshold = W * 0.003
    text_rows = dark_per_row > threshold
    
    print(f"   Порог для строк: {threshold:.0f} пикселей")
    print(f"   Строк с текстом: {text_rows.sum()} из {H}")
    
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
    
    print(f"   Найдено строк: {len(lines)}")
    for i, (y1, y2) in enumerate(lines):
        print(f"      Строка {i+1}: Y=[{y1}, {y2}], высота={y2-y1}px")
    
    return lines

def extract_chars_from_line(line_arr):
    """
    Выделение символов из строки.
    Возвращает список изображений выделенных символов
    """
    H, W = line_arr.shape
    has_dark = (line_arr < 128).any(axis=0)
    
    chars = []
    x = 0
    
    while x < W:
        # Ищем начало буквы
        while x < W and not has_dark[x]:
            x += 1
        
        if x >= W:
            break
        
        start = x
        
        # Собираем высоты тёмных столбцов
        dark_heights = []
        while x < W and has_dark[x]:
            dark_heights.append((line_arr[:, x] < 128).sum())
            x += 1
        
        end = x - 1
        
        # Порог артефакта
        if dark_heights:
            max_height = max(dark_heights)
            artifact_threshold = max(1, int(max_height * ARTIFACT_TOLERANCE))
        else:
            artifact_threshold = 1
        
        # Артефакты после буквы
        while x < W:
            dark_count = (line_arr[:, x] < 128).sum()
            
            if dark_count == 0:
                x += 1
                break
            elif dark_count <= artifact_threshold:
                dark_heights.append(dark_count)
                end = x
                x += 1
            else:
                break
        
        if end - start >= 2:
            # Проверка на склеенные буквы
            col_heights = [(line_arr[:, col] < 128).sum() for col in range(start, end + 1)]
            max_h = max(col_heights)
            min_h = min(col_heights)
            
            if min_h < max_h * SPLIT_MERGE_THRESHOLD and (end - start) > 6:
                # Разрезаем в самом узком месте
                split_col = start + col_heights.index(min_h)
                
                # Левая буква
                if split_col - start >= 2:
                    char_img = extract_char_region(line_arr, start, split_col - 1, H, W)
                    if char_img is not None:
                        chars.append(char_img)
                
                # Правая буква
                if end - split_col >= 2:
                    char_img = extract_char_region(line_arr, split_col + 1, end, H, W)
                    if char_img is not None:
                        chars.append(char_img)
            else:
                # Обычная буква
                char_img = extract_char_region(line_arr, start, end, H, W)
                if char_img is not None:
                    chars.append(char_img)
    
    return chars

def extract_char_region(line_arr, start, end, H, W):
    """
    Извлечение региона с символом с паддингом
    """
    x1p = max(0, start - PAD_PIXELS)
    x2p = min(W - 1, end + PAD_PIXELS)
    char_img = line_arr[0:H, x1p:x2p+1]
    new_h = H + 2 * PAD_PIXELS
    new_w = (x2p - x1p + 1) + 2 * PAD_PIXELS
    padded = np.full((new_h, new_w), 255, dtype=np.uint8)
    padded[PAD_PIXELS:PAD_PIXELS+H, PAD_PIXELS:PAD_PIXELS+(x2p-x1p+1)] = char_img
    
    return padded

def resize_to_square(img_array, target_size=64):
    """
    Изменение размера изображения до target_size × target_size
    с сохранением пропорций и добавлением белых полей
    """
    H, W = img_array.shape
    
    # Создаём квадратное изображение с белым фоном
    size = max(H, W)
    square = np.full((size, size), 255, dtype=np.uint8)
    
    # Центрируем изображение
    y_off = (size - H) // 2
    x_off = (size - W) // 2
    square[y_off:y_off+H, x_off:x_off+W] = img_array
    
    # Ресайзим до целевого размера
    square_img = Image.fromarray(square)
    square_img = square_img.resize((target_size, target_size), Image.LANCZOS)
    
    return np.array(square_img)

def create_shifted_image(img_array_64x64, shift_pixels=32):
    """
    Создаёт сдвинутую версию изображения 64x64.
    Сдвигает содержимое ВЛЕВО на shift_pixels.
    Справа остаётся белое пространство.
    """
    H, W = img_array_64x64.shape
    
    # Создаём новое изображение с белым фоном
    shifted = np.full((H, W), 255, dtype=np.uint8)
    
    # Сдвигаем влево: копируем правую часть на левую позицию
    shifted[:, 0:W-shift_pixels] = img_array_64x64[:, shift_pixels:W]
    
    return shifted

def save_shifted_char(padded_img, output_dir):
    """
    Сохраняет сдвинутую версию символа в единую папку.
    Возвращает путь к сохранённому файлу.
    """
    # Приводим к 64x64
    img_64 = resize_to_square(padded_img, OUTPUT_SIZE)
    
    # Создаём сдвинутую версию
    shifted_64 = create_shifted_image(img_64, SHIFT_PIXELS)
    
    # Генерируем случайное имя
    random_name = uuid.uuid4().hex[:12]
    
    # Сохраняем сдвинутую версию
    shifted_path = os.path.join(output_dir, f"{random_name}.png")
    Image.fromarray(shifted_64).save(shifted_path)
    
    return shifted_path

def process_file(image_path):
    """Обработка одного файла"""
    print("\n" + "="*70)
    print(f"  ОБРАБОТКА: {os.path.basename(image_path)}")
    print("="*70)
    
    # Загрузка и предобработка
    img_gray, arr = load_and_preprocess(image_path)
    H, W = arr.shape
    
    # Поиск строк
    lines = find_lines(arr)
    
    if not lines:
        print("   ❌ Текст не найден!")
        return 0
    
    total_chars = 0
    
    # Обрабатываем каждую строку
    for line_idx, (y1, y2) in enumerate(lines):
        print(f"\n   ── Строка {line_idx+1} (Y:{y1}-{y2}) ──")
        
        line_arr = arr[y1:y2, :]
        chars = extract_chars_from_line(line_arr)
        
        print(f"   Символов найдено: {len(chars)}")
        
        # Сохраняем каждый символ
        for char_idx, padded_img in enumerate(chars):
            shifted_path = save_shifted_char(padded_img, OUTPUT_DIR)
            print(f"      [{char_idx+1}] ✓ {os.path.basename(shifted_path)}")
            total_chars += 1
    
    print(f"\n   ✅ Всего сохранено: {total_chars} символов")
    return total_chars

def main():
    """Главная функция"""
    print("="*70)
    print("  ИЗВЛЕЧЕНИЕ СИМВОЛОВ С SHIFT-АУГМЕНТАЦИЕЙ")
    print(f"  Размер выходных изображений: {OUTPUT_SIZE}×{OUTPUT_SIZE}")
    print(f"  Сдвиг: {SHIFT_PIXELS} пикселей влево ({SHIFT_RATIO*100:.0f}%)")
    print(f"  Выходная папка: {OUTPUT_DIR}/")
    print("="*70)
    
    # Проверяем существование входной папки
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Папка не найдена: {INPUT_DIR}")
        return
    
    # Получаем список всех файлов изображений
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    image_files = []
    
    # Исключаем папку dataset из списка обрабатываемых файлов
    for file in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, file)
        if os.path.isfile(file_path):
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                image_files.append(file_path)
    
    if not image_files:
        print(f"❌ Изображения не найдены в папке: {INPUT_DIR}")
        return
    
    print(f"\n📁 Найдено файлов: {len(image_files)}")
    for f in image_files:
        print(f"   - {os.path.basename(f)}")
    
    # Обрабатываем каждый файл
    total_all_chars = 0
    processed_files = 0
    
    for image_path in sorted(image_files):
        try:
            chars_count = process_file(image_path)
            total_all_chars += chars_count
            processed_files += 1
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Итоговая статистика
    print("\n" + "="*70)
    print("  ИТОГИ")
    print("="*70)
    print(f"  Обработано файлов: {processed_files}/{len(image_files)}")
    print(f"  Всего извлечено символов: {total_all_chars}")
    print(f"  Все изображения сохранены в: {OUTPUT_DIR}/")
    print(f"  Формат: {OUTPUT_SIZE}×{OUTPUT_SIZE} пикселей, сдвиг влево на {SHIFT_PIXELS}px")
    
    # Показываем пример содержимого папки
    if total_all_chars > 0:
        saved_files = os.listdir(OUTPUT_DIR)
        print(f"\n  📊 Статистика сохранённых файлов:")
        print(f"     Всего файлов: {len(saved_files)}")
        print(f"     Примеры имён:")
        for f in saved_files[:5]:
            print(f"       - {f}")
        if len(saved_files) > 5:
            print(f"       ... и ещё {len(saved_files) - 5} файлов")

if __name__ == "__main__":
    main()
