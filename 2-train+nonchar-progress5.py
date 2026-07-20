#!/usr/bin/env python3
"""
Обучение ОДНОЙ модели: 161 символ + 1 nonchar + 1 SPACE = 163 класса
Модель определяет КОНКРЕТНЫЙ СИМВОЛ, ПРОБЕЛ или говорит "НЕ СИМВОЛ"
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ
# ============================================
DATA_DIR = Path("dataset/ocr_dataset_v3")
NONCHAR_DIR = Path("dataset/nonchar")
MODEL_PATH = "full_ocr_model.pth"
BATCH_SIZE = 8196
EPOCHS = 100
LEARNING_RATE = 3e-3
NUM_WORKERS = 16
VAL_SPLIT = 0.2

# ============================================
# ВСЕ СИМВОЛЫ + NONCHAR + SPACE
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

# Создаём маппинг символ → индекс
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Индексы
NUM_CHAR_CLASSES = len(CHARS)       # 161
NONCHAR_IDX = NUM_CHAR_CLASSES      # 161
SPACE_IDX = NUM_CHAR_CLASSES + 1    # 162
NUM_CLASSES = NUM_CHAR_CLASSES + 2  # 163

IGNORE_FOLDERS = {'nonchar', 'dataset', 'debug', '__pycache__', '.git'}

print("="*70)
print(f"  ОБУЧЕНИЕ ПОЛНОЙ OCR МОДЕЛИ ({NUM_CLASSES} классов)")
print(f"  Символов: {NUM_CHAR_CLASSES} + NONCHAR + SPACE")
print("="*70)

# ============================================
# ДАТАСЕТ (ВСЁ В ОДНОМ)
# ============================================
class FullOCRDataset(Dataset):
    def __init__(self, data_dir, nonchar_dir, transform=None):
        self.data = []
        self.transform = transform
        
        print("\n📂 ЗАГРУЗКА ДАННЫХ...")
        print("="*50)
        
        # ============================================
        # Загружаем символы с их КОНКРЕТНЫМИ классами
        # ============================================
        print("\n🔤 Загрузка СИМВОЛОВ...")
        
        char_folders = []
        
        if data_dir.exists():
            all_subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
            print(f"   Всего подпапок: {len(all_subdirs)}")
            
            for subdir in all_subdirs:
                folder_name = subdir.name
                
                if folder_name.lower() in IGNORE_FOLDERS:
                    print(f"   ⏭️  Пропущена: {folder_name}")
                    continue
                
                # Пропускаем SPACE (будет загружена отдельно)
                if folder_name == 'SPACE':
                    continue
                
                # Ищем символ по имени папки
                found_char = None
                for char, safe_name in SAFE_NAME.items():
                    if safe_name == folder_name:
                        found_char = char
                        break
                
                if found_char is None and folder_name in CHARS:
                    found_char = folder_name
                
                if found_char and found_char in char_to_idx:
                    char_folders.append((found_char, subdir))
                elif found_char:
                    print(f"   ⚠️  Символ '{found_char}' не в char_to_idx!")
                else:
                    print(f"   ⚠️  Неизвестная папка: {folder_name}")
        
        print(f"   Папок с символами: {len(char_folders)}")
        
        char_counts = {}
        total_chars = 0
        
        for char, char_folder in tqdm(char_folders, desc="   Загрузка", unit="папка"):
            images = list(char_folder.glob("*.png"))
            if images:
                label = char_to_idx[char]
                for img_path in images:
                    self.data.append((img_path, label))
                char_counts[char] = len(images)
                total_chars += len(images)
        
        print(f"   Загружено символов: {total_chars:,}")
        print(f"   Классов символов: {len(char_counts)}")
        
        # ============================================
        # Загружаем ПРОБЕЛЫ из папки SPACE
        # ============================================
        print(f"\n␣ Загрузка ПРОБЕЛОВ (класс {SPACE_IDX})...")
        
        space_dir = data_dir / "SPACE"
        space_count = 0
        
        if space_dir.exists():
            space_images = list(space_dir.glob("*.png"))
            print(f"   Найдено пробелов: {len(space_images):,}")
            
            for img_path in space_images:
                self.data.append((img_path, SPACE_IDX))
                space_count += 1
        else:
            print(f"   ⚠️  Папка SPACE не найдена")
        
        print(f"   Загружено пробелов: {space_count:,}")
        
        # ============================================
        # Загружаем NONCHAR (класс NONCHAR_IDX)
        # ============================================
        print(f"\n🚫 Загрузка NONCHAR (класс {NONCHAR_IDX})...")
        
        nonchar_count = 0
        
        if nonchar_dir.exists():
            nonchar_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                nonchar_images.extend(list(nonchar_dir.glob(ext)))
            
            print(f"   Найдено: {len(nonchar_images):,}")
            
            for img_path in nonchar_images:
                self.data.append((img_path, NONCHAR_IDX))
                nonchar_count += 1
        else:
            print(f"   ❌ Папка {nonchar_dir} не найдена!")
        
        print(f"   Загружено nonchar: {nonchar_count:,}")
        
        # Перемешиваем
        random.seed(42)
        random.shuffle(self.data)
        
        # Статистика
        labels = [l for _, l in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        
        n_chars = sum(1 for l in labels if l < NUM_CHAR_CLASSES)
        n_spaces = sum(1 for l in labels if l == SPACE_IDX)
        n_nonchar = sum(1 for l in labels if l == NONCHAR_IDX)
        
        print(f"\n📊 ИТОГО:")
        print(f"   Символов:     {n_chars:,} ({100*n_chars/len(labels):.1f}%)")
        print(f"   Пробелов:     {n_spaces:,} ({100*n_spaces/len(labels):.1f}%)")
        print(f"   Nonchar:      {n_nonchar:,} ({100*n_nonchar/len(labels):.1f}%)")
        print(f"   Всего:        {len(self.data):,}")
        print(f"   Классов:      {len(unique)}")
        print(f"   Мин/Макс:     {counts.min():,} / {counts.max():,}")
        print(f"   Среднее:      {counts.mean():.0f}\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            img = Image.open(img_path).convert('L')
        except:
            img = Image.new('L', (64, 64), 255)
        
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.LANCZOS)
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)

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
            nn.Linear(512, num_classes)  # 163 класса!
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))

# ============================================
# ОБУЧЕНИЕ
# ============================================
def train():
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🎮 Устройство: {device}")
    print(f"   Доступно GPU: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    # Очищаем кэш
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    
    # Трансформации
    train_transform = T.Compose([
        T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Загружаем данные
    print("\n🔄 Загрузка датасета...")
    full_dataset = FullOCRDataset(DATA_DIR, NONCHAR_DIR, transform=None)
    
    if len(full_dataset) == 0:
        print("\n❌ Пустой датасет!")
        return
    
    # Разделяем на train/val
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    class SubsetDataset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            img_path, label = self.dataset.data[self.indices[idx]]
            try:
                img = Image.open(img_path).convert('L')
            except:
                img = Image.new('L', (64, 64), 255)
            if img.size != (64, 64):
                img = img.resize((64, 64), Image.LANCZOS)
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
    
    train_dataset = SubsetDataset(full_dataset, train_indices.indices, train_transform)
    val_dataset = SubsetDataset(full_dataset, val_indices.indices, val_transform)
    
    print(f"\n📊 Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4
    )
    
    # Создаём модель
    model = CharCNN(num_classes=NUM_CLASSES)
    
    if num_gpus > 1:
        print(f"\n🚀 DataParallel на {num_gpus} GPU!")
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ Параметров: {total_params:,}")
    
    # Оптимизатор и функция потерь
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()  # Для multiclass!
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    print(f"\n🚀 ОБУЧЕНИЕ ({EPOCHS} эпох)")
    print("="*70)
    
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        nonchar_correct = 0
        nonchar_total = 0
        space_correct = 0
        space_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                
                # Nonchar
                nonchar_mask = (labels == NONCHAR_IDX)
                nonchar_total += nonchar_mask.sum().item()
                nonchar_correct += ((preds == labels) & nonchar_mask).sum().item()
                
                # Space
                space_mask = (labels == SPACE_IDX)
                space_total += space_mask.sum().item()
                space_correct += ((preds == labels) & space_mask).sum().item()
        
        val_acc = 100 * val_correct / val_total
        nonchar_acc = 100 * nonchar_correct / nonchar_total if nonchar_total > 0 else 0
        space_acc = 100 * space_correct / space_total if space_total > 0 else 0
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            
            if num_gpus > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'nonchar_acc': nonchar_acc,
                'space_acc': space_acc,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'nonchar_idx': NONCHAR_IDX,
                'space_idx': SPACE_IDX,
                'num_classes': NUM_CLASSES,
            }, MODEL_PATH)
            marker = " 💾"
        else:
            marker = ""
        
        print(f"   Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Nonchar: {nonchar_acc:.1f}% | Space: {space_acc:.1f}% | Best: {best_acc:.2f}% (epoch {best_epoch}){marker}")
    
    print(f"\n✅ ГОТОВО! Модель: {MODEL_PATH}")
    print(f"   Лучшая точность: {best_acc:.2f}%")
    print(f"   Модель определяет {NUM_CLASSES} классов:")
    print(f"      {NUM_CHAR_CLASSES} символов + NONCHAR + SPACE")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    train()
