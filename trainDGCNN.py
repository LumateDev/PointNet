"""
Скрипт обучения DGCNN для семантической сегментации LiDAR данных
Оптимизирован под облака точек из LAS файлов
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import traceback
from datetime import datetime
import json

# Импорт модели
from modelDGCNN import DGCNN_Segmentation, DGCNN_Segmentation_V2

# Импорт датасета из существующего файла
try:
    from dataset import LASDataset
except ImportError:
    print("❌ Не найден файл dataset.py! Убедитесь, что он в той же папке.")
    sys.exit(1)


class ClassMapper:
    """
    Маппер классов для преобразования реальных ID классов в индексы модели
    """
    def __init__(self):
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def fit(self, unique_classes):
        """Создает маппинг на основе уникальных классов"""
        unique_classes = sorted(unique_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        print(f"📋 Маппинг классов создан:")
        for cls, idx in self.class_to_idx.items():
            print(f"   Класс {cls} → Индекс {idx}")
        return len(unique_classes)
    
    def map_labels(self, labels):
        """Преобразует метки классов в индексы"""
        mapped = np.copy(labels)
        for cls, idx in self.class_to_idx.items():
            mapped[labels == cls] = idx
        return mapped


def find_las_file():
    """Находит LAS файл в различных возможных местах"""
    possible_paths = [
        'Univer2019.las',
        'datasets/raw/NEONDSSampleLiDARPointCloud.las',
        'datasets/raw/Univer2019.las',
        'datasets/unlabeled/Univer2019.las',
    ]
    
    # Проверяем конкретные пути
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Найден LAS файл: {path}")
            return path
    
    # Ищем в папках
    folders_to_check = ['.', 'datasets/raw', 'datasets/unlabeled']
    
    for folder in folders_to_check:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith('.las'):
                    full_path = os.path.join(folder, file)
                    print(f"✅ Найден LAS файл: {full_path}")
                    return full_path
    
    return None


def setup_logging():
    """Настройка логирования"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_dgcnn_{timestamp}.log'
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    
    return log_file


def analyze_dataset_classes(dataset, num_samples=500):
    """
    Анализирует классы в датасете
    Возвращает уникальные классы для создания маппинга
    """
    print(f"\n📊 Анализ классов в датасете (выборка {num_samples} блоков)...")
    all_labels = []
    
    sample_size = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for i in tqdm(indices, desc="Сбор классов"):
        try:
            _, labels = dataset[i]
            all_labels.append(labels.numpy())
        except:
            continue
    
    if not all_labels:
        print("❌ Не удалось загрузить данные!")
        return None
    
    all_labels = np.concatenate(all_labels)
    unique_classes = np.unique(all_labels)
    
    print(f"🔍 Найденные классы: {unique_classes}")
    
    # Статистика
    for cls in unique_classes:
        count = np.sum(all_labels == cls)
        percentage = 100.0 * count / len(all_labels)
        print(f"   Класс {cls}: {count} точек ({percentage:.2f}%)")
    
    return unique_classes


def calculate_class_weights(dataset, class_mapper, num_samples=1000):
    """Вычисление весов классов для balanced loss"""
    print("\n⚖️  Вычисление весов классов...")
    all_labels = []
    
    sample_size = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for i in tqdm(indices, desc="Анализ весов"):
        try:
            _, labels = dataset[i]
            # Маппинг меток
            labels_mapped = class_mapper.map_labels(labels.numpy())
            all_labels.append(labels_mapped)
        except:
            continue
    
    if not all_labels:
        print("❌ Не удалось загрузить данные!")
        num_classes = len(class_mapper.class_to_idx)
        return torch.ones(num_classes)
    
    all_labels = np.concatenate(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    
    num_classes = len(class_mapper.class_to_idx)
    weights = np.ones(num_classes)
    
    # Inverse frequency weighting
    for idx, count in zip(unique, counts):
        weights[int(idx)] = 1.0 / (count + 1e-6)
    
    # Нормализация
    weights = weights / weights.sum() * num_classes
    
    print(f"⚖️  Веса классов:")
    for idx in range(num_classes):
        orig_class = class_mapper.idx_to_class[idx]
        print(f"   Класс {orig_class} (индекс {idx}): {weights[idx]:.4f}")
    
    return torch.FloatTensor(weights)


class LASDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper для LASDataset с маппингом классов
    """
    def __init__(self, base_dataset, class_mapper):
        self.base_dataset = base_dataset
        self.class_mapper = class_mapper
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        points, labels = self.base_dataset[idx]
        
        # Маппинг меток
        labels_np = labels.numpy()
        labels_mapped = self.class_mapper.map_labels(labels_np)
        labels = torch.from_numpy(labels_mapped).long()
        
        return points, labels


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Одна эпоха обучения с поддержкой mixed precision"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_points = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (points, labels) in enumerate(pbar):
        try:
            points = points.to(device).float()
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    pred = model(points)
                    pred_flat = pred.contiguous().view(-1, pred.size(-1))
                    labels_flat = labels.view(-1)
                    loss = criterion(pred_flat, labels_flat)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(points)
                pred_flat = pred.contiguous().view(-1, pred.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(pred_flat, labels_flat)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Метрики
            pred_choice = pred_flat.argmax(dim=1)
            correct = (pred_choice == labels_flat).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_points += labels_flat.size(0)
            
            # Обновление прогресс-бара
            current_acc = 100.0 * correct / labels_flat.size(0)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
        except Exception as e:
            print(f"\n❌ Ошибка на батче {batch_idx}: {e}")
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_acc = 100.0 * total_correct / total_points if total_points > 0 else 0.0
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device, class_mapper, phase="валидации"):
    """Валидация модели"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_points = 0
    
    num_classes = len(class_mapper.class_to_idx)
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(tqdm(val_loader, desc=phase)):
            try:
                points = points.to(device).float()
                labels = labels.to(device).long()
                
                pred = model(points)
                pred_flat = pred.contiguous().view(-1, pred.size(-1))
                labels_flat = labels.view(-1)
                
                loss = criterion(pred_flat, labels_flat)
                
                pred_choice = pred_flat.argmax(dim=1)
                correct = (pred_choice == labels_flat).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_points += labels_flat.size(0)
                
                # Per-class accuracy
                for c in range(num_classes):
                    mask = labels_flat == c
                    if mask.sum() > 0:
                        class_correct[c] += (pred_choice[mask] == labels_flat[mask]).sum().item()
                        class_total[c] += mask.sum().item()
                
            except Exception as e:
                print(f"\n❌ Ошибка при валидации батча {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    avg_acc = 100.0 * total_correct / total_points if total_points > 0 else 0.0
    
    # Детальная статистика
    print(f"\n📊 СТАТИСТИКА {phase.upper()}:")
    print("="*60)
    print(f"{'Оригинальный класс':<20} {'Индекс':<10} {'Точек':<12} {'Точность %':<12}")
    print("-"*60)
    
    for idx in range(num_classes):
        orig_class = class_mapper.idx_to_class[idx]
        if class_total[idx] > 0:
            class_acc = 100.0 * class_correct[idx] / class_total[idx]
            print(f"{orig_class:<20} {idx:<10} {int(class_total[idx]):<12} {class_acc:<12.2f}")
        else:
            print(f"{orig_class:<20} {idx:<10} {'-':<12} {'-':<12}")
    
    print("-"*60)
    print(f"{'ИТОГО':<20} {'':<10} {total_points:<12} {avg_acc:<12.2f}")
    print("="*60)
    
    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, history, config, filename):
    """Сохранение checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'history': history,
        'config': config
    }
    torch.save(checkpoint, filename)
    print(f"💾 Checkpoint сохранен: {filename}")


def save_training_plots(history, save_path='checkpoints/training_dgcnn.png'):
    """Сохранение графиков обучения"""
    try:
        import matplotlib.pyplot as plt
        
        # Убедимся, что все списки имеют одинаковую длину для тренировочных данных
        # и корректное соотношение для валидационных данных
        train_len = len(history['train_loss'])
        val_len = len(history['val_loss'])
        
        # Если val_loss/val_acc содержит начальное значение, берем train_len точек
        if val_len == train_len + 1:  # есть начальное значение
            epochs = list(range(train_len + 1))  # +1 для начальной эпохи
            val_epochs = epochs[1:]  # без начальной точки
        else:
            epochs = list(range(train_len))
            val_epochs = epochs
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(epochs[1:], history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        ax1.plot(val_epochs, history['val_loss'][1:], label='Val Loss', marker='s', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('DGCNN: Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs[1:], history['train_acc'], label='Train Acc', marker='o', linewidth=2)
        ax2.plot(val_epochs, history['val_acc'][1:], label='Val Acc', marker='s', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('DGCNN: Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ График сохранен: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Не удалось сохранить графики: {e}")


def main():
    try:
        # Фиксация random seed
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Настройка логирования
        log_file = setup_logging()
        print(f"📝 Логи сохраняются в: {log_file}\n")
        
        # ============= ПАРАМЕТРЫ =============
        NUM_POINTS = 4096
        BATCH_SIZE = 8
        EPOCHS = 10
        LEARNING_RATE = 0.001
        BLOCK_SIZE = 50.0
        STRIDE = 25.0
        K_NEIGHBORS = 20
        EMB_DIMS = 1024
        DROPOUT = 0.5
        USE_MIXED_PRECISION = True  # Mixed precision для ускорения
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Устройство: {device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Поиск LAS файла
        print("\n🔍 Поиск LAS файла...")
        las_file_path = find_las_file()
        
        if las_file_path is None:
            print("❌ LAS файл не найден!")
            sys.exit(1)
        
        print(f"✅ Используется файл: {las_file_path}")
        
        # Создание датасета
        print("\n📦 Создание датасета...")
        full_dataset = LASDataset(
            las_file_path,
            num_points=NUM_POINTS,
            block_size=BLOCK_SIZE,
            stride=STRIDE,
            train=True
        )
        
        if len(full_dataset) == 0:
            print("❌ Датасет пуст!")
            sys.exit(1)
        
        print(f"📊 Всего блоков: {len(full_dataset)}")
        
        # Анализ классов и создание маппинга
        unique_classes = analyze_dataset_classes(full_dataset)
        if unique_classes is None:
            print("❌ Не удалось проанализировать классы!")
            sys.exit(1)
        
        class_mapper = ClassMapper()
        num_classes = class_mapper.fit(unique_classes)
        
        print(f"\n🎯 Количество классов для модели: {num_classes}")
        
        # Обертка датасета с маппингом
        full_dataset = LASDatasetWrapper(full_dataset, class_mapper)
        
        # Разделение на train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📚 Train: {len(train_dataset)} блоков, Val: {len(val_dataset)} блоков")
        
        # DataLoaders
        print("\n🔄 Создание DataLoaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Создание модели
        print("\n🧠 Создание DGCNN модели...")
        model = DGCNN_Segmentation(
            num_classes=num_classes,
            k=K_NEIGHBORS,
            emb_dims=EMB_DIMS,
            dropout=DROPOUT
        ).to(device)
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Параметров модели: {total_params:,}")
        print(f"🎯 Обучаемых параметров: {trainable_params:,}")
        
        # Вычисление весов классов
        class_weights = calculate_class_weights(full_dataset.base_dataset, class_mapper)
        class_weights = class_weights.to(device)
        
        # Loss и optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler(device) if USE_MIXED_PRECISION and torch.cuda.is_available() else None
        
        # Конфигурация для сохранения
        config = {
            'model': 'DGCNN_Segmentation',
            'num_classes': num_classes,
            'num_points': NUM_POINTS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'k_neighbors': K_NEIGHBORS,
            'emb_dims': EMB_DIMS,
            'dropout': DROPOUT,
            'block_size': BLOCK_SIZE,
            'stride': STRIDE,
            'las_file': las_file_path,
            'class_mapping': {int(k): int(v) for k, v in class_mapper.class_to_idx.items()}
        }
        
        # Создание папки для checkpoints
        os.makedirs('checkpoints', exist_ok=True)
        
        # Сохранение конфигурации
        with open('checkpoints/dgcnn_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("💾 Конфигурация сохранена: checkpoints/dgcnn_config.json")
        
        # Начальная валидация
        print("\n🔍 Начальная валидация...")
        initial_val_loss, initial_val_acc = validate(
            model, val_loader, criterion, device, class_mapper, "начальной валидации"
        )
        
        # История обучения
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [initial_val_loss],
            'val_acc': [initial_val_acc]
        }
        
        best_acc = initial_val_acc
        best_epoch = 0
        
        # Обучение
        print("\n" + "="*60)
        print("🚀 НАЧАЛО ОБУЧЕНИЯ DGCNN")
        print("="*60)
        print(f"  • Модель: DGCNN with EdgeConv")
        print(f"  • Классов: {num_classes}")
        print(f"  • Эпох: {EPOCHS}")
        print(f"  • Batch size: {BATCH_SIZE}")
        print(f"  • Learning rate: {LEARNING_RATE}")
        print(f"  • K neighbors: {K_NEIGHBORS}")
        print(f"  • Mixed precision: {USE_MIXED_PRECISION and torch.cuda.is_available()}")
        print("="*60)
        
        for epoch in range(1, EPOCHS + 1):
            print(f"\n{'='*60}")
            print(f"📅 Эпоха {epoch}/{EPOCHS}")
            print(f"{'='*60}")
            print(f"📚 Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, scaler
            )
            
            print(f"\n📈 Train → Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            
            # Validation
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, class_mapper, f"валидации эпохи {epoch}"
            )
            
            print(f"📉 Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Обновление истории
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Сохранение лучшей модели
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                save_checkpoint(
                    model, optimizer, epoch, val_acc, val_loss, history, config,
                    'checkpoints/best_dgcnn.pth'
                )
                print(f"🏆 Новая лучшая модель! Val Acc: {val_acc:.2f}%")
            
            # Сохранение последней модели
            save_checkpoint(
                model, optimizer, epoch, val_acc, val_loss, history, config,
                'checkpoints/last_dgcnn.pth'
            )
            
            # Сохранение графиков каждые 5 эпох
            if epoch % 5 == 0:
                save_training_plots(history)
            
            # Scheduler step
            scheduler.step()
        
        # Финальное сохранение
        print("\n" + "="*60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("="*60)
        print(f"\n🏆 Лучшая модель:")
        print(f"   • Эпоха: {best_epoch}")
        print(f"   • Точность: {best_acc:.2f}%")
        print(f"   • Файл: checkpoints/best_dgcnn.pth")
        print(f"\n📊 Финальные графики:")
        save_training_plots(history)
        
        print(f"\n📋 Маппинг классов:")
        for orig_class, idx in class_mapper.class_to_idx.items():
            print(f"   Класс {orig_class} → Индекс {idx}")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()