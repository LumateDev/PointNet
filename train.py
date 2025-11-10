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
from model import PointNet2SemSeg
from dataset import LASDataset

def find_las_file():
    """Находит LAS файл в различных возможных местах"""
    possible_paths = [
        'datasets/raw/NEONDSSampleLiDARPointCloud.las',
        'Univer2019.las',
        'datasets/raw/Univer2019.las',
        'datasets/unlabeled/Univer2019.las',
    ]
    
    # Сначала проверяем конкретные пути
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Найден LAS файл: {path}")
            return path
    
    # Если конкретные пути не нашлись, ищем в папках
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
    log_file = f'logs/training_{timestamp}.log'
    
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

def calculate_class_weights(dataset):
    """Вычисление весов классов для balanced loss"""
    print("⚖️  Вычисление весов классов...")
    all_labels = []
    
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for i in tqdm(indices, desc="Анализ классов"):
        try:
            _, labels = dataset[i]
            all_labels.append(labels.numpy())
        except Exception as e:
            continue
    
    if not all_labels:
        print("❌ Не удалось загрузить данные для вычисления весов!")
        return torch.ones(8)
    
    all_labels = np.concatenate(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    
    print(f"📊 Найдены классы: {unique}")
    print(f"📊 Количество точек по классам: {counts}")
    
    # Правильное вычисление весов с защитой от деления на ноль
    weights = 1.0 / (counts + 1e-6)  # Добавляем epsilon для избежания деления на 0
    weights = weights / weights.sum() * len(weights)  # Нормализация
    
    weight_dict = {int(u): float(w) for u, w in zip(unique, weights)}
    print(f"⚖️  Веса классов: {weight_dict}")
    
    # Создание тензора весов для всех классов (0-7)
    weight_tensor = torch.ones(8)
    for class_id, weight in weight_dict.items():
        weight_tensor[class_id] = weight
    
    return weight_tensor

def analyze_dataset_distribution(dataset, dataset_name="датасете"):
    """Анализ распределения классов в датасете"""
    print(f"\n📊 Анализ распределения классов в {dataset_name}...")
    all_labels = []
    
    for i in tqdm(range(min(500, len(dataset))), desc="Сбор статистики"):
        try:
            _, labels = dataset[i]
            all_labels.append(labels.numpy())
        except:
            continue
    
    if all_labels:
        all_labels = np.concatenate(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        
        print(f"📈 Распределение классов в {dataset_name}:")
        total_points = len(all_labels)
        for class_id, count in zip(unique, counts):
            percentage = 100.0 * count / total_points
            print(f"   Класс {class_id + 1}: {count} точек ({percentage:.2f}%)")
        
        print(f"📊 Всего уникальных классов: {len(unique)}")
        print(f"📊 Всего точек в выборке: {total_points}")
        
        return unique
    return []

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_points = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (points, labels) in enumerate(pbar):
        try:
            points, labels = points.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Прямой проход
            pred = model(points)  # (B, N, num_classes)
            
            # Проверка на NaN в предсказаниях
            if torch.isnan(pred).any():
                print(f"⚠️  Обнаружены NaN в предсказаниях на батче {batch_idx}")
                continue
            
            # Вычисление loss
            pred = pred.contiguous().view(-1, pred.size(-1))
            labels = labels.view(-1)
            loss = criterion(pred, labels)
            
            # Проверка на NaN в loss
            if torch.isnan(loss):
                print(f"⚠️  Обнаружен NaN в loss на батче {batch_idx}")
                continue
            
            # Обратный проход
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Метрики
            pred_choice = pred.argmax(dim=1)
            correct = (pred_choice == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_points += labels.size(0)
            
            # Обновление прогресс-бара
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / labels.size(0):.2f}%'
            })
            
        except Exception as e:
            print(f"\n❌ Ошибка на батче {batch_idx}: {e}")
            continue
    
    if len(train_loader) == 0:
        print("❌ Нет данных для обучения!")
        return 0.0, 0.0
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * total_correct / total_points if total_points > 0 else 0.0
    
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device, phase="валидации"):
    """Валидация модели с улучшенной диагностикой"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_points = 0
    
    class_correct = torch.zeros(8, device=device)
    class_total = torch.zeros(8, device=device)
    
    print(f"\n🔍 Детальная диагностика {phase}...")
    
    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(tqdm(val_loader, desc=phase)):
            try:
                points, labels = points.to(device), labels.to(device)
                
                pred = model(points)
                pred = pred.contiguous().view(-1, pred.size(-1))
                labels_flat = labels.view(-1)
                
                loss = criterion(pred, labels_flat)
                
                pred_choice = pred.argmax(dim=1)
                correct = (pred_choice == labels_flat).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_points += labels_flat.size(0)
                
                # Per-class accuracy
                for c in range(8):
                    mask = labels_flat == c
                    if mask.sum() > 0:
                        class_correct[c] += (pred_choice[mask] == labels_flat[mask]).sum().item()
                        class_total[c] += mask.sum().item()
                        
            except Exception as e:
                print(f"\n❌ Ошибка при {phase} батча {batch_idx}: {e}")
                continue
    
    if len(val_loader) == 0:
        print(f"❌ Нет данных для {phase}!")
        return 0.0, 0.0
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100.0 * total_correct / total_points if total_points > 0 else 0.0
    
    # Детальная статистика
    print(f"\n📊 ДЕТАЛЬНАЯ СТАТИСТИКА {phase.upper()}:")
    print("="*50)
    print("Класс | Точек всего | Точность %")
    print("-"*50)
    
    any_class_processed = False
    total_val_points = 0
    
    for c in range(8):
        if class_total[c] > 0:
            any_class_processed = True
            class_acc = 100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
            print(f"  {c + 1}   | {int(class_total[c]):11d} | {class_acc:8.2f}%")
            total_val_points += int(class_total[c])
        else:
            print(f"  {c + 1}   | {'-':11} | {'-':8}")
    
    print("-"*50)
    print(f"  Всего | {total_val_points:11d} | {avg_acc:8.2f}%")
    
    if not any_class_processed:
        print("⚠️  В выборке не найдено ни одного класса!")
        print("💡 Проверьте разметку данных")
    
    print(f"\n📊 Общая точность: {avg_acc:.2f}%")
    print(f"📊 Общий loss: {avg_loss:.4f}")
    
    return avg_loss, avg_acc

def save_training_plots(history):
    """Сохранение графиков обучения"""
    try:
        import matplotlib.pyplot as plt
        
        epochs = list(range(len(history['train_loss'])))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xticks(epochs)
        
        # Accuracy
        ax2.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
        ax2.plot(epochs, history['val_acc'], label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xticks(epochs)
        
        plt.tight_layout()
        plt.savefig('checkpoints/training_history.png', dpi=300, bbox_inches='tight')
        print("✅ График сохранен: checkpoints/training_history.png")
        
    except Exception as e:
        print(f"⚠️  Не удалось сохранить графики: {e}")

def main():
    try:
        # Фиксация random seed для воспроизводимости
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Настройка логирования
        log_file = setup_logging()
        print(f"📝 Логи сохраняются в: {log_file}\n")
        
        # Параметры
        NUM_CLASSES = 8
        NUM_POINTS = 4096
        BATCH_SIZE = 4
        EPOCHS = 10
        LEARNING_RATE = 0.001
        BLOCK_SIZE = 50.0
        STRIDE = 25.0
        
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
            print("❌ Датасет пуст! Проверьте файл LAS")
            sys.exit(1)
        
        # Анализ распределения классов в полном датасете
        analyze_dataset_distribution(full_dataset, "полном датасете")
        
        # Разделение на train/val (80/20) с фиксированным seed
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        if train_size == 0 or val_size == 0:
            print("❌ Недостаточно данных для разделения на train/val!")
            sys.exit(1)
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Фиксируем seed
        )
        
        print(f"📚 Train: {len(train_dataset)} блоков, Val: {len(val_dataset)} блоков")
        
        # DataLoaders
        print("\n🔄 Создание DataLoaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Модель
        print("\n🧠 Создание модели...")
        model = PointNet2SemSeg(num_classes=NUM_CLASSES).to(device)
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Параметров модели: {total_params:,}")
        print(f"🎯 Обучаемых параметров: {trainable_params:,}")
        
        # Вычисление весов классов
        class_weights = calculate_class_weights(full_dataset)
        class_weights = class_weights.to(device)
        
        # Loss и optimizer - используем CrossEntropyLoss вместо NLLLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Создание папки для checkpoints
        os.makedirs('checkpoints', exist_ok=True)
        
        # Начальная валидация перед обучением
        print("\n🔍 Начальная валидация до обучения...")
        initial_val_loss, initial_val_acc = validate(model, val_loader, criterion, device, "начальной валидации")

        print(f"📊 Начальная точность: {initial_val_acc:.2f}%")
        
        # Обучение
        best_acc = initial_val_acc
        best_epoch = 0
        history = {
            'train_loss': [initial_val_loss],  # Используем val loss как начальный
            'train_acc': [initial_val_acc],    # Используем val acc как начальный
            'val_loss': [initial_val_loss],
            'val_acc': [initial_val_acc]
        }
        
        print(f"📊 Начальная точность: {initial_val_acc:.2f}%")
        
        print("\n🚀 Начало обучения...\n")
        print("="*60)
        print(f"Параметры обучения:")
        print(f"  • Эпох: {EPOCHS}")
        print(f"  • Batch size: {BATCH_SIZE}")
        print(f"  • Learning rate: {LEARNING_RATE}")
        print(f"  • Размер блока: {BLOCK_SIZE}m")
        print(f"  • Исходный файл: {las_file_path}")
        print("="*60)
        
        for epoch in range(1, EPOCHS + 1):
            print(f"\n{'='*60}")
            print(f"Эпоха {epoch}/{EPOCHS}")
            print(f"{'='*60}")
            
            try:
                # Train
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, epoch
                )
                
                print(f"\n📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Validation
                val_loss, val_acc = validate(model, val_loader, criterion, device, f"валидации эпохи {epoch}")
                print(f"📉 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Сохранение истории
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Сохранение лучшей модели
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'class_weights': class_weights,
                        'history': history,
                        'las_file_path': las_file_path
                    }, 'checkpoints/best_model.pth')
                    print(f"💾 Сохранена лучшая модель (Val Acc: {val_acc:.2f}%)")
                
                # Сохранение последней модели
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': history,
                    'las_file_path': las_file_path
                }, 'checkpoints/last_model.pth')
                
                # Learning rate scheduling
                scheduler.step()
                
            except Exception as e:
                print(f"\n❌ Ошибка на эпохе {epoch}:")
                print(traceback.format_exc())
                continue  # Продолжаем обучение вместо прерывания
        
        print(f"\n{'='*60}")
        print("  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"{'='*60}")
        print(f"\n🏆 Лучшая модель:")
        print(f"   • Эпоха: {best_epoch}")
        print(f"   • Точность: {best_acc:.2f}%")
        print(f"   • Файл: checkpoints/best_model.pth")
        
        # Сохранение графиков обучения
        print("\n📊 Сохранение графиков обучения...")
        save_training_plots(history)
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()