#!/usr/bin/env python3
"""
OCR РАСПОЗНАВАНИЕ с дебагом
Поддерживает 163 класса: 161 символ + NONCHAR + SPACE
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import sys
import os
from pathlib import Path

# ============================================
# НАСТРОЙКИ
# ============================================
PAD_PIXELS = 1
ARTIFACT_TOLERANCE = 0.08
SPLIT_MERGE_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.5

# ============================================
# ВСЕ СИМВОЛЫ + СПЕЦКЛАССЫ
# ============================================
CHARS = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)

SAFE_NAME = {
    '/': 'SLASH', '\\': 'BACKSLASH', ':': 'COLON',
    '*': 'ASTERISK', '?': 'QUEST', '"': 'DQUOTE',
    '<': 'LT', '>': 'GT', '|': 'PIPE',
    ' ': 'SPACE', '!': 'EXCL', '#': 'HASH',
    '$': 'DOLLAR', '%': 'PERCENT', '&': 'AMP',
    "'": 'SQUOTE', '(': 'LPAREN', ')': 'RPAREN',
    '+': 'PLUS', ',': 'COMMA', '-': 'MINUS',
    '.': 'DOT', ';': 'SEMICOLON', '=': 'EQUAL',
    '@': 'AT', '[': 'LBRACKET', ']': 'RBRACKET',
    '^': 'CARET', '_': 'UNDERSCORE', '`': 'BACKTICK',
    '{': 'LBRACE', '}': 'RBRACE', '~': 'TILDE',
    'Ё': 'YO', 'ё': 'yo',
}

# Создаём маппинги
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Индексы специальных классов
NUM_CHAR_CLASSES = len(CHARS)       # 161
NONCHAR_IDX = NUM_CHAR_CLASSES      # 161
SPACE_IDX = NUM_CHAR_CLASSES + 1    # 162
NUM_CLASSES = NUM_CHAR_CLASSES + 2  # 163

# ============================================
# ДЕБАГ ПАПКИ
# ============================================
DEBUG_DIR = "debug_ocr"
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(f"{DEBUG_DIR}/chars", exist_ok=True)
os.makedirs(f"{DEBUG_DIR}/nonchars", exist_ok=True)
os.makedirs(f"{DEBUG_DIR}/spaces", exist_ok=True)
os.makedirs(f"{DEBUG_DIR}/lines", exist_ok=True)
os.makedirs(f"{DEBUG_DIR}/steps", exist_ok=True)

def safe_filename(char):
    """Безопасное имя файла"""
    if char in SAFE_NAME:
        return SAFE_NAME[char]
    if char.isalnum():
        return char
    return f"SYM_{ord(char)}"

# ============================================
# МОДЕЛЬ (163 класса)
# ============================================
class CharCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 163 класса
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================
def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Устройство: {device}")
    
    # Пытаемся определить количество классов из чекпоинта
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        elif 'model_state_dict' in checkpoint:
            # Определяем по размеру последнего слоя
            state_dict = checkpoint['model_state_dict']
            # Ищем ключ fc.4.weight или fc.4.bias
            for key in state_dict.keys():
                if 'fc.4.weight' in key or 'fc.4.bias' in key:
                    num_classes = state_dict[key].shape[0]
                    break
            else:
                num_classes = NUM_CLASSES
        else:
            num_classes = NUM_CLASSES
        
        print(f"   Классов в чекпоинте: {num_classes}")
    else:
        num_classes = NUM_CLASSES
    
    # Создаём модель с правильным количеством классов
    model = CharCNN(num_classes=num_classes).to(device)
    
    # Извлекаем state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Убираем префикс module.
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Инфа о чекпоинте
    if isinstance(checkpoint, dict):
        print(f"   Эпоха: {checkpoint.get('epoch', '?')}")
        print(f"   Val Acc: {checkpoint.get('val_acc', '?')}")
        if 'space_acc' in checkpoint:
            print(f"   Space Acc: {checkpoint.get('space_acc', '?')}")
    
    return model, device, num_classes

# ============================================
# ДЕБАГ ФУНКЦИИ
# ============================================
def debug_save(name, img, note=""):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    path = f"{DEBUG_DIR}/{name}"
    img.save(path)
    if note:
        print(f"   🐛 {name} — {note}")

def draw_boxes_on_image(img_array, lines, all_chars_by_line, output_path):
    """Рисует рамки на изображении"""
    img_rgb = Image.fromarray(img_array).convert('RGB')
    draw = ImageDraw.Draw(img_rgb)
    
    # Строки зелёным
    for y1, y2 in lines:
        draw.rectangle([0, y1, img_rgb.width-1, y2], outline=(0, 255, 0), width=2)
    
    # Символы
    for line_idx, (y1, y2) in enumerate(lines):
        if line_idx < len(all_chars_by_line):
            for item in all_chars_by_line[line_idx]:
                x1, x2, item_type, char, conf = item
                
                if item_type == 'char':
                    color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (255, 255, 0)
                    label = char if char else "?"
                elif item_type == 'space':
                    color = (0, 200, 255)  # Голубой для пробелов
                    label = "␣"
                else:
                    color = (255, 0, 0)  # Красный для nonchar
                    label = "✗"
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
                draw.text((x1, y1-15), f"{label} ({conf:.0%})", fill=color)
    
    img_rgb.save(output_path)
    print(f"   🐛 Сохранено с разметкой: {output_path}")

# ============================================
# ШАГ 1: ЗАГРУЗКА
# ============================================
def step1_load(image_path):
    print("\n" + "="*70)
    print("  ШАГ 1: ЗАГРУЗКА ИЗОБРАЖЕНИЯ")
    print("="*70)
    
    img = Image.open(image_path)
    print(f"   Файл: {image_path}")
    print(f"   Размер: {img.size}")
    
    debug_save("steps/01_original.png", img, "исходное")
    
    img_gray = img.convert('L')
    arr = np.array(img_gray)
    
    print(f"   Средняя яркость: {arr.mean():.0f}")
    
    if arr.mean() < 128:
        print(f"   🔄 Инвертирую (тёмный фон)")
        img_gray = ImageOps.invert(img_gray)
        arr = np.array(img_gray)
        debug_save("steps/02_inverted.png", img_gray, "после инверсии")
    else:
        debug_save("steps/02_grayscale.png", img_gray, "без инверсии")
    
    return img_gray, arr

# ============================================
# ШАГ 2: ПОИСК СТРОК
# ============================================
def step2_find_lines(arr):
    print("\n" + "="*70)
    print("  ШАГ 2: ПОИСК СТРОК")
    print("="*70)
    
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
    
    print(f"   Найдено строк: {len(lines)}")
    for i, (y1, y2) in enumerate(lines):
        print(f"      Строка {i+1}: Y=[{y1}, {y2}], высота={y2-y1}px")
    
    debug_img = Image.fromarray(arr).convert('RGB')
    draw = ImageDraw.Draw(debug_img)
    for y1, y2 in lines:
        draw.rectangle([0, y1, arr.shape[1]-1, y2], outline=(0, 255, 0), width=1)
    debug_save("steps/03_lines.png", debug_img, f"{len(lines)} строк")
    
    return lines

# ============================================
# ШАГ 3: ПОИСК СИМВОЛОВ
# ============================================
def step3_find_chars_in_line(line_arr, line_idx):
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
                x += 1
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
                    padded = extract_char_with_padding(line_arr, start, split_col - 1, H, W)
                    chars.append((start, split_col - 1, padded))
                
                if end - split_col >= 2:
                    padded = extract_char_with_padding(line_arr, split_col + 1, end, H, W)
                    chars.append((split_col + 1, end, padded))
            else:
                padded = extract_char_with_padding(line_arr, start, end, H, W)
                chars.append((start, end, padded))
    
    # Дебаг
    if chars:
        line_debug = Image.fromarray(line_arr).convert('RGB')
        draw = ImageDraw.Draw(line_debug)
        for x1, x2, _ in chars:
            draw.rectangle([x1, 0, x2, H-1], outline=(255, 0, 0), width=1)
        debug_save(f"lines/line_{line_idx:02d}_detected.png", line_debug,
                   f"строка {line_idx+1}: {len(chars)} областей")
    
    return chars

def extract_char_with_padding(line_arr, start, end, H, W):
    x1p = max(0, start - PAD_PIXELS)
    x2p = min(W - 1, end + PAD_PIXELS)
    char_img = line_arr[0:H, x1p:x2p+1]
    new_h = H + 2 * PAD_PIXELS
    new_w = (x2p - x1p + 1) + 2 * PAD_PIXELS
    padded = np.full((new_h, new_w), 255, dtype=np.uint8)
    padded[PAD_PIXELS:PAD_PIXELS+H, PAD_PIXELS:PAD_PIXELS+(x2p-x1p+1)] = char_img
    return padded

# ============================================
# ШАГ 4: РАСПОЗНАВАНИЕ
# ============================================
def step4_recognize(model, device, padded_img, x1, x2, line_idx, char_idx, num_classes):
    H, W = padded_img.shape
    
    size = max(H, W)
    square = np.full((size, size), 255, dtype=np.uint8)
    y_off = (size - H) // 2
    x_off = (size - W) // 2
    square[y_off:y_off+H, x_off:x_off+W] = padded_img
    
    square_64 = np.array(Image.fromarray(square).resize((64, 64), Image.LANCZOS))
    
    arr = square_64.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        top5_probs, top5_idx = torch.topk(probs[0], 5)
    
    best_idx = top5_idx[0].item()
    best_prob = top5_probs[0].item()
    
    char_img_pil = Image.fromarray(square_64)
    
    # Определяем специальные классы
    space_idx = num_classes - 1  # Последний класс
    nonchar_idx = num_classes - 2  # Предпоследний
    
    if best_idx == nonchar_idx:
        # NONCHAR
        char_img_rgb = char_img_pil.convert('RGB')
        draw = ImageDraw.Draw(char_img_rgb)
        draw.rectangle([0, 0, 63, 63], outline=(255, 0, 0), width=2)
        
        fname = f"nonchars/L{line_idx:02d}_C{char_idx:03d}_x{x1}-{x2}_NONCHAR_{best_prob:.2f}.png"
        char_img_rgb.save(f"{DEBUG_DIR}/{fname}")
        
        return 'nonchar', None, best_prob, []
    
    elif best_idx == space_idx:
        # SPACE (пробел)
        char_img_rgb = char_img_pil.convert('RGB')
        draw = ImageDraw.Draw(char_img_rgb)
        draw.rectangle([0, 0, 63, 63], outline=(0, 200, 255), width=2)
        
        fname = f"spaces/L{line_idx:02d}_C{char_idx:03d}_x{x1}-{x2}_SPACE_{best_prob:.2f}.png"
        char_img_rgb.save(f"{DEBUG_DIR}/{fname}")
        
        return 'space', ' ', best_prob, []
    
    else:
        # СИМВОЛ
        char = idx_to_char.get(best_idx, '?')
        safe_char = safe_filename(char)
        
        top5_chars = []
        for i in range(5):
            idx = top5_idx[i].item()
            prob = top5_probs[i].item()
            if idx < NUM_CHAR_CLASSES:
                top5_chars.append((idx_to_char.get(idx, '?'), prob))
            elif idx == nonchar_idx:
                top5_chars.append(("NONCHAR", prob))
            elif idx == space_idx:
                top5_chars.append(("SPACE", prob))
        
        if best_prob >= CONFIDENCE_THRESHOLD:
            fname = f"chars/L{line_idx:02d}_C{char_idx:03d}_x{x1}-{x2}_{safe_char}_{best_prob:.2f}.png"
            char_img_pil.save(f"{DEBUG_DIR}/{fname}")
        else:
            char_img_rgb = char_img_pil.convert('RGB')
            draw = ImageDraw.Draw(char_img_rgb)
            draw.rectangle([0, 0, 63, 63], outline=(255, 255, 0), width=2)
            fname = f"chars/L{line_idx:02d}_C{char_idx:03d}_x{x1}-{x2}_{safe_char}_LOW_{best_prob:.2f}.png"
            char_img_rgb.save(f"{DEBUG_DIR}/{fname}")
        
        return 'char', char, best_prob, top5_chars

# ============================================
# ГЛАВНАЯ
# ============================================
def main():
    if len(sys.argv) < 2:
        print("Использование: python 3-char-not-char2.py screenshot.png [модель.pth]")
        return
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "full_ocr_model.pth"
    
    print("="*70)
    print("  OCR: РАСПОЗНАВАНИЕ С ДЕБАГОМ (CHAR + SPACE + NONCHAR)")
    print("="*70)
    
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    print(f"\n🔍 Загрузка модели: {model_path}")
    model, device, num_classes = load_model(model_path)
    
    img_gray, arr = step1_load(image_path)
    lines = step2_find_lines(arr)
    
    if not lines:
        print("\n❌ Текст не найден!")
        return
    
    print("\n" + "="*70)
    print("  РАСПОЗНАВАНИЕ")
    print("="*70)
    
    all_results = []
    all_chars_by_line = []
    total_chars = 0
    total_spaces = 0
    total_nonchars = 0
    
    for line_idx, (y1, y2) in enumerate(lines):
        print(f"\n── Строка {line_idx+1} ──")
        
        line_arr = arr[y1:y2, :]
        chars = step3_find_chars_in_line(line_arr, line_idx)
        
        if not chars:
            all_chars_by_line.append([])
            continue
        
        print(f"   Областей: {len(chars)}")
        
        line_text_parts = []
        prev_x2 = None
        line_debug_data = []
        
        for char_idx, (x1, x2, padded_img) in enumerate(chars):
            item_type, char, conf, top5 = step4_recognize(
                model, device, padded_img, x1, x2, line_idx, char_idx, num_classes
            )
            
            if prev_x2 is not None and x1 - prev_x2 > (x2 - x1) * 0.8:
                line_text_parts.append(" ")
            
            if item_type == 'char':
                total_chars += 1
                display_char = char if char != ' ' else '␣'
                line_text_parts.append(display_char)
                
                top3_str = " | ".join([f"'{c}':{p:.1%}" for c, p in top5[:3]]) if top5 else ""
                marker = "✓" if conf >= CONFIDENCE_THRESHOLD else "?"
                print(f"   {marker} x={x1:4d}  '{display_char}'  ({conf:.1%})  [{top3_str}]")
                
                line_debug_data.append((x1, x2, 'char', char, conf))
                
            elif item_type == 'space':
                total_spaces += 1
                line_text_parts.append(" ")
                print(f"   ␣ x={x1:4d}  ПРОБЕЛ  ({conf:.1%})")
                
                line_debug_data.append((x1, x2, 'space', ' ', conf))
                
            else:
                total_nonchars += 1
                line_text_parts.append("·")
                print(f"   ✗ x={x1:4d}  НЕ СИМВОЛ  ({conf:.1%})")
                
                line_debug_data.append((x1, x2, 'nonchar', None, conf))
            
            prev_x2 = x2
        
        all_chars_by_line.append(line_debug_data)
        
        line_text = "".join(line_text_parts)
        print(f"   └─ Строка: \"{line_text}\"")
        all_results.append({'text': line_text, 'line_idx': line_idx})
    
    # Сохраняем размеченное изображение
    print(f"\n{'='*70}")
    print("  СОХРАНЕНИЕ РАЗМЕТКИ")
    print(f"{'='*70}")
    draw_boxes_on_image(np.array(img_gray), lines, all_chars_by_line, 
                       f"{DEBUG_DIR}/steps/04_annotated.png")
    
    # Итоги
    print(f"\n{'='*70}")
    print("  ИТОГОВЫЙ ТЕКСТ")
    print("="*70)
    for r in all_results:
        print(f"  {r['text']}")
    
    total = total_chars + total_spaces + total_nonchars
    print(f"\n{'='*70}")
    print(f"  СТАТИСТИКА")
    print(f"{'='*70}")
    print(f"  Всего областей: {total}")
    if total > 0:
        print(f"  СИМВОЛОВ:       {total_chars} ({100*total_chars/total:.1f}%)")
        print(f"  ПРОБЕЛОВ:       {total_spaces} ({100*total_spaces/total:.1f}%)")
        print(f"  НЕ СИМВОЛОВ:    {total_nonchars} ({100*total_nonchars/total:.1f}%)")
    
    print(f"\n🐛 ДЕБАГ-ФАЙЛЫ: {DEBUG_DIR}/")
    print(f"   Зелёный  = символ")
    print(f"   Голубой  = пробел")
    print(f"   Красный  = не символ")
    print(f"   Жёлтый   = низкая уверенность")

if __name__ == "__main__":
    main()
