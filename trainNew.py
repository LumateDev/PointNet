import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys
import traceback
from datetime import datetime
import json
from pathlib import Path

# ✅ ИСПРАВЛЕНО: импортируем только PointNet2LiDAR
from modelNew import PointNet2LiDAR, WeightedCrossEntropyLoss, compute_class_weights
from dataset import LASDataset


class FocalLoss(nn.Module):
    """Focal Loss для борьбы с дисбалансом классов"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MetricsCalculator:
    """Калькулятор метрик для сегментации"""
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.total_correct = 0
        self.total_points = 0
        self.class_correct = np.zeros(self.num_classes)
        self.class_total = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """
        predictions: (N,) numpy array или tensor
        targets: (N,) numpy array или tensor
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        correct = (predictions == targets).sum()
        self.total_correct += correct
        self.total_points += len(targets)
        
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                self.class_correct[c] += (predictions[mask] == targets[mask]).sum()
                self.class_total[c] += mask.sum()
        
        for t, p in zip(targets, predictions):
            if t < self.num_classes and p < self.num_classes:
                self.confusion_matrix[int(t), int(p)] += 1
    
    def get_metrics(self):
        overall_acc = 100.0 * self.total_correct / self.total_points if self.total_points > 0 else 0.0
        
        class_acc = {}
        for c in range(self.num_classes):
            if self.class_total[c] > 0:
                class_acc[c] = 100.0 * self.class_correct[c] / self.class_total[c]
            else:
                class_acc[c] = 0.0
        
        valid_classes = [acc for acc in class_acc.values() if acc > 0]
        mean_class_acc = np.mean(valid_classes) if valid_classes else 0.0
        
        # IoU
        iou_per_class = {}
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            
            if tp + fp + fn > 0:
                iou_per_class[c] = tp / (tp + fp + fn)
            else:
                iou_per_class[c] = 0.0
        
        valid_ious = [iou for iou in iou_per_class.values() if iou > 0]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        
        return {
            'overall_acc': overall_acc,
            'mean_class_acc': mean_class_acc,
            'class_acc': class_acc,
            'mean_iou': mean_iou * 100,
            'iou_per_class': {k: v * 100 for k, v in iou_per_class.items()},
        }


class EarlyStopping:
    """Early stopping для предотвращения переобучения"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class Trainer:
    """Класс для обучения PointNet++ на LiDAR данных"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n🖥️  Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.setup_directories()
        self.setup_logging()
        
        self.model = self.build_model()
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001)
        )
        
        self.metrics = MetricsCalculator(num_classes=config['num_classes'])
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_miou': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_miou = 0.0
        self.best_epoch = 0
        
        self.writer = SummaryWriter(log_dir=self.config['log_dir'])
    
    def setup_directories(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"PointNet2LiDAR_{timestamp}"
        
        self.config['log_dir'] = f"runs/{self.run_name}"
        self.config['checkpoint_dir'] = f"checkpoints/{self.run_name}"
        
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
    
    def setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        log_file = f"logs/train_{self.run_name}.log"
        
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, 'w', encoding='utf-8')
            
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            
            def flush(self):
                pass
        
        sys.stdout = Logger(log_file)
        print(f"📝 Логи сохраняются: {log_file}")
    
    def build_model(self):
        """Создание модели PointNet2LiDAR"""
        print(f"\n{'='*70}")
        print("🧠 СОЗДАНИЕ МОДЕЛИ")
        print(f"{'='*70}")
        
        model = PointNet2LiDAR(
            num_classes=self.config['num_classes'],
            use_features=self.config.get('use_features', True),
            feature_dim=self.config.get('feature_dim', 3)
        )
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Модель: PointNet++ для LiDAR")
        print(f"   Классов: {self.config['num_classes']}")
        print(f"   Параметров: {total_params:,}")
        print(f"   Обучаемых: {trainable_params:,}")
        print(f"   Размер: {total_params * 4 / 1024 / 1024:.2f} MB")
        print(f"   Использование признаков: {self.config.get('use_features', True)}")
        
        if self.config.get('use_features', True):
            print(f"   Размерность признаков: {self.config.get('feature_dim', 3)}")
            print(f"   (intensity, return_number, number_of_returns)")
        
        return model
    
    def build_criterion(self):
        """Создание функции потерь"""
        print(f"\n{'='*70}")
        print("📉 ФУНКЦИЯ ПОТЕРЬ")
        print(f"{'='*70}")
        
        loss_type = self.config.get('loss_type', 'focal').lower()
        
        class_weights = None
        if self.config.get('use_class_weights', True):
            print("⚖️  Вычисление весов классов...")
            class_weights = self.calculate_class_weights()
            class_weights = class_weights.to(self.device)
            
            print("\n   Веса классов:")
            for i, w in enumerate(class_weights):
                print(f"   Класс {i}: {w:.4f}")
        
        if loss_type == 'focal':
            criterion = FocalLoss(
                alpha=class_weights,
                gamma=self.config.get('focal_gamma', 2.0)
            )
            print(f"\n✅ Loss: Focal Loss (gamma={self.config.get('focal_gamma', 2.0)})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"\n✅ Loss: Cross Entropy")
        
        return criterion
    
    def build_optimizer(self):
        """Создание оптимизатора"""
        print(f"\n{'='*70}")
        print("🎯 ОПТИМИЗАТОР")
        print(f"{'='*70}")
        
        optimizer_type = self.config.get('optimizer', 'adamw').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        
        print(f"✅ Optimizer: {optimizer_type.upper()}")
        print(f"   Learning Rate: {lr}")
        print(f"   Weight Decay: {weight_decay}")
        
        return optimizer
    
    def build_scheduler(self):
        """Создание scheduler'а"""
        scheduler_type = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
            print(f"\n✅ Scheduler: Cosine Annealing")
            print(f"   Min LR: {self.config.get('min_lr', 1e-6)}")
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
            print(f"\n✅ Scheduler: Step LR (step=20, gamma=0.5)")
        else:
            scheduler = None
            print(f"\n✅ Scheduler: None")
        
        return scheduler
    
    def calculate_class_weights(self):
        """Вычисление весов классов на основе распределения"""
        try:
            # Пытаемся создать датасет с признаками
            dataset = LASDataset(
                self.config['las_file'],
                num_points=self.config['num_points'],
                block_size=self.config['block_size'],
                stride=self.config.get('stride', self.config['block_size'] / 2),
                train=True,
                use_features=self.config.get('use_features', True)
            )
        except TypeError:
            # Если use_features не поддерживается
            dataset = LASDataset(
                self.config['las_file'],
                num_points=self.config['num_points'],
                block_size=self.config['block_size'],
                stride=self.config.get('stride', self.config['block_size'] / 2),
                train=True
            )
        
        all_labels = []
        sample_size = min(500, len(dataset))
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        print(f"   Анализ {sample_size} блоков...")
        
        for i in tqdm(indices, desc="   Сбор статистики", leave=False):
            try:
                data = dataset[i]
                if isinstance(data, tuple):
                    labels = data[1]
                else:
                    labels = data
                
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                
                all_labels.append(labels)
            except Exception as e:
                continue
        
        if not all_labels:
            print("   ⚠️  Не удалось загрузить данные, используем равные веса")
            return torch.ones(self.config['num_classes'])
        
        all_labels = np.concatenate(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        
        print(f"\n   Распределение классов:")
        total = len(all_labels)
        for cls, cnt in zip(unique, counts):
            percent = 100.0 * cnt / total
            print(f"   Класс {cls}: {cnt:,} точек ({percent:.2f}%)")
        
        # Вычисление весов
        weights = compute_class_weights(
            {int(cls): int(cnt) for cls, cnt in zip(unique, counts)},
            mode='effective'
        )
        
        # Создание тензора весов
        weight_tensor = torch.ones(self.config['num_classes'])
        for class_id, weight in zip(unique, weights):
            if class_id < self.config['num_classes']:
                weight_tensor[int(class_id)] = weight
        
        return weight_tensor
    
    def prepare_data(self):
        """Подготовка DataLoader'ов"""
        print(f"\n{'='*70}")
        print("📦 ПОДГОТОВКА ДАННЫХ")
        print(f"{'='*70}")
        
        try:
            full_dataset = LASDataset(
                self.config['las_file'],
                num_points=self.config['num_points'],
                block_size=self.config['block_size'],
                stride=self.config.get('stride', self.config['block_size'] / 2),
                train=True,
                use_features=self.config.get('use_features', True)
            )
        except TypeError:
            full_dataset = LASDataset(
                self.config['las_file'],
                num_points=self.config['num_points'],
                block_size=self.config['block_size'],
                stride=self.config.get('stride', self.config['block_size'] / 2),
                train=True
            )
        
        print(f"📚 Всего блоков: {len(full_dataset)}")
        
        # Проверка данных
        sample_data, sample_labels = full_dataset[0]
        print(f"   Размерность блока: {sample_data.shape}")
        print(f"   Размерность меток: {sample_labels.shape}")
        
        # Train/Val split
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"   Train: {train_size} блоков ({train_ratio*100:.0f}%)")
        print(f"   Val: {val_size} блоков ({(1-train_ratio)*100:.0f}%)")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"\n✅ DataLoaders созданы")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        self.metrics.reset()
        total_loss = 0
        
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{self.config["epochs"]}',
            ncols=100
        )
        
        for batch_idx, (points, labels) in enumerate(pbar):
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            pred = self.model(points)  # (B, N, num_classes)
            
            # Reshape
            pred_flat = pred.reshape(-1, pred.size(-1))
            labels_flat = labels.reshape(-1)
            
            # Loss
            loss = self.criterion(pred_flat, labels_flat)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Metrics
            pred_choice = pred_flat.argmax(dim=1)
            self.metrics.update(pred_choice, labels_flat)
            
            total_loss += loss.item()
            
            # Progress bar
            batch_acc = 100.0 * (pred_choice == labels_flat).float().mean().item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.1f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.metrics.get_metrics()
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Валидация"""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0
        
        for points, labels in tqdm(val_loader, desc='Validation', ncols=100, leave=False):
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            pred = self.model(points)
            pred_flat = pred.reshape(-1, pred.size(-1))
            labels_flat = labels.reshape(-1)
            
            loss = self.criterion(pred_flat, labels_flat)
            
            pred_choice = pred_flat.argmax(dim=1)
            self.metrics.update(pred_choice, labels_flat)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics.get_metrics()
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'history': self.history,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'best_val_miou': self.best_val_miou
        }
        
        # Последняя модель
        last_path = os.path.join(self.config['checkpoint_dir'], 'last_model.pth')
        torch.save(checkpoint, last_path)
        
        # Лучшая модель
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"   💾 Лучшая модель сохранена!")
    
    def train(self):
        """Основной цикл обучения"""
        print(f"\n{'='*70}")
        print("🚀 НАЧАЛО ОБУЧЕНИЯ")
        print(f"{'='*70}\n")
        
        # Сохранение конфига
        config_path = os.path.join(self.config['checkpoint_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"💾 Конфиг сохранен: {config_path}\n")
        
        # Подготовка данных
        train_loader, val_loader = self.prepare_data()
        
        # Обучение
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*70}")
            print(f"📅 Эпоха {epoch}/{self.config['epochs']}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            
            # Вывод
            print(f"\n📊 Результаты эпохи {epoch}:")
            print(f"{'─'*70}")
            print(f"📈 Train:      Loss={train_loss:.4f}  |  Acc={train_metrics['overall_acc']:.2f}%")
            print(f"📉 Validation: Loss={val_loss:.4f}  |  Acc={val_metrics['overall_acc']:.2f}%  |  mIoU={val_metrics['mean_iou']:.2f}%")
            
            # Per-class
            print(f"\n   По классам:")
            for cls_id in sorted(val_metrics['class_acc'].keys()):
                acc = val_metrics['class_acc'][cls_id]
                iou = val_metrics['iou_per_class'].get(cls_id, 0)
                print(f"   Класс {cls_id}: Acc={acc:5.2f}%  |  IoU={iou:5.2f}%")
            
            # TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_metrics['overall_acc'],
                'val': val_metrics['overall_acc']
            }, epoch)
            self.writer.add_scalar('mIoU', val_metrics['mean_iou'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['overall_acc'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['overall_acc'])
            self.history['val_miou'].append(val_metrics['mean_iou'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Сохранение
            is_best = val_metrics['overall_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['overall_acc']
                self.best_val_miou = val_metrics['mean_iou']
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if self.early_stopping(val_metrics['overall_acc']):
                print(f"\n⚠️  Early stopping на эпохе {epoch}")
                print(f"   Нет улучшений {self.early_stopping.patience} эпох")
                break
        
        # Финал
        print(f"\n{'='*70}")
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*70}")
        print(f"\n🏆 Лучший результат:")
        print(f"   Эпоха: {self.best_epoch}")
        print(f"   Accuracy: {self.best_val_acc:.2f}%")
        print(f"   mIoU: {self.best_val_miou:.2f}%")
        print(f"\n📁 Результаты:")
        print(f"   Чекпоинты: {self.config['checkpoint_dir']}")
        print(f"   TensorBoard: {self.config['log_dir']}")
        print(f"\n💡 Запустите TensorBoard:")
        print(f"   tensorboard --logdir=runs")
        
        self.writer.close()


def find_las_file():
    """Поиск LAS файла"""
    possible_paths = [
        'datasets/raw/NEONDSSampleLiDARPointCloud.las',
        'Univer2019.las',
        'datasets/raw/Univer2019.las',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Поиск в папках
    for folder in ['.', 'datasets/raw', 'datasets', 'data']:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith('.las'):
                    return os.path.join(folder, file)
    
    return None


def main():
    """Главная функция"""
    config = {
        # ========== МОДЕЛЬ ==========
        'num_classes': 4,  # Ваши классы: 1, 2, 5, 6
        
        # ========== PointNet++ параметры ==========
        'use_features': True,   # ✅ Использовать LiDAR признаки
        'feature_dim': 3,       # intensity, return_number, number_of_returns
        
        # ========== ДАННЫЕ ==========
        'num_points': 4096,     # Точек в блоке
        'block_size': 50.0,     # Размер блока (метры)
        'stride': 25.0,         # Шаг между блоками
        'train_ratio': 0.8,     # Train/Val split
        
        # ========== ОБУЧЕНИЕ ==========
        'batch_size': 8,        # Уменьшите до 4 если не хватает памяти
        'epochs': 10,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # ========== LOSS ==========
        'loss_type': 'focal',        # 'focal' или 'ce'
        'use_class_weights': True,   # ✅ Важно для дисбаланса классов
        'focal_gamma': 2.0,
        
        # ========== OPTIMIZER ==========
        'optimizer': 'adamw',   # 'adamw', 'adam', 'sgd'
        'scheduler': 'cosine',  # 'cosine', 'step', None
        'min_lr': 1e-6,
        
        # ========== EARLY STOPPING ==========
        'patience': 15,
        'min_delta': 0.001,
        
        # ========== ДРУГОЕ ==========
        'num_workers': 4,
        'seed': 42,
    }
    
    # Поиск LAS файла
    las_file = find_las_file()
    if las_file is None:
        print("\n❌ LAS файл не найден!")
        print("\n💡 Положите .las файл в одну из папок:")
        print("   • datasets/raw/")
        print("   • текущая директория")
        return
    
    config['las_file'] = las_file
    
    # Информация
    print(f"\n{'='*70}")
    print(f"  POINTNET++ LIDAR SEMANTIC SEGMENTATION")
    print(f"{'='*70}")
    print(f"\n📁 Датасет: {las_file}")
    print(f"🎯 Классов: {config['num_classes']}")
    print(f"📊 Точек в блоке: {config['num_points']}")
    print(f"📦 Размер блока: {config['block_size']}m (шаг {config['stride']}m)")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🔄 Эпох (макс): {config['epochs']}")
    print(f"💡 Признаки: {'XYZ + intensity + returns' if config['use_features'] else 'Только XYZ'}")
    
    # Seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Обучение
    trainer = Trainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        print("   Сохранение текущего состояния...")
        trainer.save_checkpoint(0, {}, is_best=False)
        print("   ✅ Сохранено")
    except Exception as e:
        print(f"\n\n❌ Ошибка во время обучения:")
        print(f"   {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()