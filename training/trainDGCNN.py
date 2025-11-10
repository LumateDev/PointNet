"""
training/trainDGCNN.py
Скрипт обучения DGCNN для семантической сегментации LiDAR данных

Структура:
    logs/DGCNN/{timestamp}/
    checkpoints/DGCNN/{timestamp}/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import traceback

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modelDGCNN import DGCNN_LiDAR
from utils.dataset import LASDataset
from utils.losses import FocalLoss, compute_class_weights
from utils.metrics import SegmentationMetrics


# ==================== КОНФИГУРАЦИЯ ====================

def get_config():
    """Конфигурация обучения"""
    config = {
        # ========== ДАННЫЕ ==========
        'las_file': 'datasets/raw/NEONDSSampleLiDARPointCloud.las',
        'dataset_config': 'configs/datasets/neon_sample.yaml',
        'num_points': 4096,
        'block_size': 50.0,
        'stride': 25.0,
        'use_features': True,
        'feature_dim': 3,  # intensity, return_number, number_of_returns
        'normalize': True,
        'train_ratio': 0.8,
        
        # ========== МОДЕЛЬ ==========
        'model_name': 'DGCNN',
        'num_classes': 4,
        'k_neighbors': 20,
        'dropout': 0.5,
        
        # ========== ОБУЧЕНИЕ ==========
        'batch_size': 8,
        'epochs': 15,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # ========== LOSS ==========
        'loss_type': 'focal',  # 'focal', 'ce', 'dice'
        'use_class_weights': True,
        'focal_gamma': 2.0,
        'weight_mode': 'effective',  # 'inverse', 'effective', 'sqrt'
        
        # ========== OPTIMIZER & SCHEDULER ==========
        'optimizer': 'adamw',  # 'adam', 'adamw', 'sgd'
        'scheduler': 'cosine',  # 'cosine', 'step', 'plateau', None
        'min_lr': 1e-6,
        'patience': 10,  # для ReduceLROnPlateau
        
        # ========== EARLY STOPPING ==========
        'early_stopping': True,
        'early_stopping_patience': 15,
        'early_stopping_delta': 0.001,
        
        # ========== AUGMENTATION ==========
        'augment_train': True,
        'augment_val': False,
        
        # ========== ДРУГОЕ ==========
        'num_workers': 4,
        'pin_memory': True,
        'save_freq': 5,  # Сохранять каждые N эпох
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    return config


# ==================== EARLY STOPPING ====================

class EarlyStopping:
    """Early stopping для предотвращения переобучения"""
    
    def __init__(self, patience=15, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"   ⚠️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0


# ==================== TRAINER ====================

class DGCNNTrainer:
    """Класс для обучения DGCNN"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Создание директорий
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"DGCNN_{self.timestamp}"
        
        self.log_dir = Path('logs') / 'DGCNN' / self.run_name
        self.checkpoint_dir = Path('checkpoints') / 'DGCNN' / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # История обучения
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_miou': [],
            'val_loss': [],
            'val_acc': [],
            'val_miou': [],
            'lr': []
        }
        
        # Лучшие метрики
        self.best_val_acc = 0.0
        self.best_val_miou = 0.0
        self.best_epoch = 0
        
        # Метрики
        self.train_metrics = SegmentationMetrics(num_classes=config['num_classes'])
        self.val_metrics = SegmentationMetrics(num_classes=config['num_classes'])
        
        # Early stopping
        if config['early_stopping']:
            self.early_stopping = EarlyStopping(
                patience=config['early_stopping_patience'],
                delta=config['early_stopping_delta']
            )
        else:
            self.early_stopping = None
        
        # Инициализация
        self._print_header()
        self._set_seed()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._save_config()
    
    def _print_header(self):
        """Печать заголовка"""
        print("\n" + "="*80)
        print(f"{'🚀 DGCNN LIDAR SEMANTIC SEGMENTATION':^80}")
        print("="*80)
        print(f"\n📅 Timestamp: {self.timestamp}")
        print(f"📁 Logs: {self.log_dir}")
        print(f"💾 Checkpoints: {self.checkpoint_dir}")
        print(f"🖥️  Device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _set_seed(self):
        """Установка seed для воспроизводимости"""
        seed = self.config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"\n🎲 Random seed: {seed}")
    
    def _setup_data(self):
        """Подготовка данных"""
        print("\n" + "="*80)
        print("📦 ПОДГОТОВКА ДАННЫХ")
        print("="*80)
        
        # Проверка файла
        if not os.path.exists(self.config['las_file']):
            raise FileNotFoundError(f"LAS file not found: {self.config['las_file']}")
        
        # ЗАГРУЗКА КОНФИГУРАЦИИ ДАТАСЕТА
        from utils.dataset_config import DatasetConfig, auto_detect_config
        
        dataset_config_path = self.config.get('dataset_config', None)
        
        if dataset_config_path and os.path.exists(dataset_config_path):
            # Загрузка из явно указанного файла
            dataset_config = DatasetConfig(dataset_config_path)
            print(f"📋 Загружена конфигурация: {dataset_config_path}")
        else:
            # Автоопределение по имени файла
            dataset_config = auto_detect_config(self.config['las_file'])
            
            if dataset_config is None:
                # Fallback: используем NEON по умолчанию
                print("⚠️  Конфигурация не найдена, используется NEON по умолчанию")
                neon_config = Path('configs/datasets/neon_sample.yaml')
                if neon_config.exists():
                    dataset_config = DatasetConfig(neon_config)
                else:
                    raise FileNotFoundError(
                        "Dataset config not found! Create configs/datasets/neon_sample.yaml"
                    )
        
        # Вывод информации о конфигурации
        dataset_config.print_info()
        
        # Обновляем num_classes из конфигурации датасета
        self.config['num_classes'] = dataset_config.num_classes
        self.train_metrics = SegmentationMetrics(num_classes=dataset_config.num_classes)
        self.val_metrics = SegmentationMetrics(num_classes=dataset_config.num_classes)
        
        # Создание полного датасета
        full_dataset = LASDataset(
            las_file=self.config['las_file'],
            num_points=self.config['num_points'],
            block_size=self.config['block_size'],
            stride=self.config['stride'],
            use_features=self.config['use_features'],
            normalize=self.config['normalize'],
            augment=False,
            dataset_config=dataset_config  # 🆕 ПЕРЕДАЕМ КОНФИГУРАЦИЮ
        )
        
        # Train/Val split
        train_size = int(self.config['train_ratio'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        generator = torch.Generator().manual_seed(self.config['seed'])
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        # Включаем аугментации для train
        if self.config['augment_train']:
            full_dataset.augment = True
        
        print(f"\n📊 Разбиение данных:")
        print(f"   Train: {train_size} блоков ({self.config['train_ratio']*100:.0f}%)")
        print(f"   Val:   {val_size} блоков ({(1-self.config['train_ratio'])*100:.0f}%)")
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        print(f"\n✅ DataLoaders созданы:")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches:   {len(self.val_loader)}")
        
        # Вычисление весов классов
        if self.config['use_class_weights']:
            print(f"\n⚖️  Вычисление весов классов...")
            class_dist = full_dataset.get_class_distribution()
            self.class_weights = compute_class_weights(
                class_dist,
                mode=self.config['weight_mode'],
                device=self.device
            )
            print(f"\n   Веса классов ({self.config['weight_mode']} mode):")
            for i, w in enumerate(self.class_weights):
                class_name = dataset_config.get_class_name(i)
                print(f"   Класс {i} ({class_name}): {w:.4f}")
        else:
            self.class_weights = None
        
        # Сохраняем конфигурацию датасета для использования в других местах
        self.dataset_config = dataset_config    


    def _setup_model(self):
        """Создание модели и loss"""
        print("\n" + "="*80)
        print("🧠 СОЗДАНИЕ МОДЕЛИ")
        print("="*80)
        
        # Модель
        self.model = DGCNN_LiDAR(
            num_classes=self.config['num_classes'],
            k=self.config['k_neighbors'],
            use_features=self.config['use_features'],
            feature_dim=self.config['feature_dim'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Модель: DGCNN")
        print(f"   Классов: {self.config['num_classes']}")
        print(f"   K соседей: {self.config['k_neighbors']}")
        print(f"   Параметров: {total_params:,}")
        print(f"   Обучаемых: {trainable_params:,}")
        print(f"   Размер: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Loss function
        print(f"\n📉 Loss function: {self.config['loss_type'].upper()}")
        
        if self.config['loss_type'] == 'focal':
            self.criterion = FocalLoss(
                alpha=self.class_weights,
                gamma=self.config['focal_gamma']
            )
            print(f"   Gamma: {self.config['focal_gamma']}")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        print(f"   Class weights: {self.config['use_class_weights']}")
    
    def _setup_optimizer(self):
        """Создание optimizer и scheduler"""
        print("\n" + "="*80)
        print("🎯 OPTIMIZER & SCHEDULER")
        print("="*80)
        
        # Optimizer
        lr = self.config['learning_rate']
        wd = self.config['weight_decay']
        
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=(0.9, 0.999)
            )
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        else:  # sgd
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=wd
            )
        
        print(f"\n✅ Optimizer: {self.config['optimizer'].upper()}")
        print(f"   Learning Rate: {lr}")
        print(f"   Weight Decay: {wd}")
        
        # Scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['min_lr']
            )
            print(f"\n✅ Scheduler: Cosine Annealing")
            print(f"   Min LR: {self.config['min_lr']}")
        
        elif self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
            print(f"\n✅ Scheduler: Step LR")
        
        elif self.config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=self.config['patience'],
                verbose=True
            )
            print(f"\n✅ Scheduler: ReduceLROnPlateau")
        
        else:
            self.scheduler = None
            print(f"\n✅ Scheduler: None")
    
    def _save_config(self):
        """Сохранение конфигурации"""
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"\n💾 Конфигурация сохранена: {config_path}")
    
    def train_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config["epochs"]} [TRAIN]',
            ncols=100,
            ascii=True
        )
        
        for batch_idx, (points, labels) in enumerate(pbar):
            # Перенос на device
            points = points.to(self.device)  # (B, N, C)
            labels = labels.to(self.device)  # (B, N)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(points)  # (B, N, num_classes)
            
            # Reshape для loss
            outputs_flat = outputs.reshape(-1, self.config['num_classes'])  # (B*N, num_classes)
            labels_flat = labels.reshape(-1)  # (B*N,)
            
            # Loss
            loss = self.criterion(outputs_flat, labels_flat)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                preds = outputs_flat.argmax(dim=1)
                self.train_metrics.update(preds, labels_flat)
            
            total_loss += loss.item()
            
            # Progress bar
            if batch_idx % 10 == 0:
                current_metrics = self.train_metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_metrics["overall_acc"]:.1f}%'
                })
        
        # Средние метрики за эпоху
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.train_metrics.get_metrics()
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Валидация"""
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch}/{self.config["epochs"]} [VAL]  ',
            ncols=100,
            ascii=True,
            leave=False
        )
        
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(points)
            
            # Reshape
            outputs_flat = outputs.reshape(-1, self.config['num_classes'])
            labels_flat = labels.reshape(-1)
            
            # Loss
            loss = self.criterion(outputs_flat, labels_flat)
            total_loss += loss.item()
            
            # Metrics
            preds = outputs_flat.argmax(dim=1)
            self.val_metrics.update(preds, labels_flat)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.val_metrics.get_metrics()
        
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
            'best_val_miou': self.best_val_miou,
            'best_epoch': self.best_epoch
        }
        
        # Последний чекпоинт
        last_path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
        
        # Лучший чекпоинт
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"      💾 Best model saved! (Acc: {self.best_val_acc:.2f}%, mIoU: {self.best_val_miou:.2f}%)")
        
        # Периодические чекпоинты
        if epoch % self.config['save_freq'] == 0:
            epoch_path = self.checkpoint_dir / f'model_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def plot_history(self):
        """Построение графиков обучения"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'DGCNN Training History - {self.run_name}', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy History')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # mIoU
        axes[1, 0].plot(epochs, self.history['train_miou'], 'b-', label='Train mIoU', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_miou'], 'r-', label='Val mIoU', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mIoU (%)')
        axes[1, 0].set_title('mIoU History')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        plot_path = self.checkpoint_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 График сохранен: {plot_path}")
    
    def train(self):
        """Основной цикл обучения"""
        print("\n" + "="*80)
        print("🚀 НАЧАЛО ОБУЧЕНИЯ")
        print("="*80)
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*80}")
            print(f"📅 Эпоха {epoch}/{self.config['epochs']}")
            print(f"{'='*80}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate(epoch)
            
            # Вывод результатов
            print(f"\n📊 Результаты эпохи {epoch}:")
            print(f"   {'─'*76}")
            print(f"   📈 TRAIN | Loss: {train_loss:.4f} | Acc: {train_metrics['overall_acc']:6.2f}% | mIoU: {train_metrics['mean_iou']:6.2f}%")
            print(f"   📉 VAL   | Loss: {val_loss:.4f} | Acc: {val_metrics['overall_acc']:6.2f}% | mIoU: {val_metrics['mean_iou']:6.2f}%")
            
            # Per-class метрики

            print(f"\n   🎯 Per-Class Metrics (Validation):")
            for i in range(self.config['num_classes']):
                acc = val_metrics['class_acc'][i]
                iou = val_metrics['iou_per_class'][i]
                # 🆕 Используем названия из конфигурации датасета
                class_name = self.dataset_config.get_class_name(i)
                print(f"      {class_name}: Acc={acc:6.2f}% | IoU={iou:6.2f}%")
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['overall_acc'])
            self.history['train_miou'].append(train_metrics['mean_iou'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['overall_acc'])
            self.history['val_miou'].append(val_metrics['mean_iou'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_metrics['overall_acc'],
                'val': val_metrics['overall_acc']
            }, epoch)
            self.writer.add_scalars('mIoU', {
                'train': train_metrics['mean_iou'],
                'val': val_metrics['mean_iou']
            }, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Проверка на лучшую модель
            is_best = val_metrics['overall_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['overall_acc']
                self.best_val_miou = val_metrics['mean_iou']
                self.best_epoch = epoch
            
            # Сохранение чекпоинта
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['overall_acc'])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_metrics['overall_acc'], epoch)
                if self.early_stopping.early_stop:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                    print(f"   No improvement for {self.early_stopping.patience} epochs")
                    break
        
        # Завершение обучения
        self._finish_training()
    
    def _finish_training(self):
        """Завершение обучения"""
        print("\n" + "="*80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("="*80)
        
        print(f"\n🏆 Лучший результат:")
        print(f"   Эпоха: {self.best_epoch}")
        print(f"   Accuracy: {self.best_val_acc:.2f}%")
        print(f"   mIoU: {self.best_val_miou:.2f}%")
        
        print(f"\n📁 Результаты сохранены:")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print(f"   Logs: {self.log_dir}")
        
        # Построение графиков
        self.plot_history()
        
        # Закрытие TensorBoard
        self.writer.close()
        
        print(f"\n💡 Запустите TensorBoard для просмотра:")
        print(f"   tensorboard --logdir=logs/DGCNN")
        print("\n" + "="*80)


# ==================== MAIN ====================

def main():
    """Главная функция"""
    
    # Парсинг аргументов (опционально)
    parser = argparse.ArgumentParser(description='Train DGCNN for LiDAR Segmentation')
    parser.add_argument('--las_file', type=str, default=None, help='Path to LAS file')
    parser.add_argument('--dataset_config', type=str, default=None, help='Path to dataset config YAML')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    args = parser.parse_args()
    
    # Конфигурация
    config = get_config()
    
     # Переопределение из аргументов
    if args.las_file is not None:
        config['las_file'] = args.las_file
    if args.dataset_config is not None:
        config['dataset_config'] = args.dataset_config
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['learning_rate'] = args.lr
    
    # Проверка наличия файла
    if not os.path.exists(config['las_file']):
        print(f"\n❌ Ошибка: LAS файл не найден: {config['las_file']}")
        print("\n💡 Убедитесь, что файл находится в правильной папке:")
        print("   datasets/raw/NEONDSSampleLiDARPointCloud.las")
        return
    
    # Создание тренера
    try:
        trainer = DGCNNTrainer(config)
        trainer.train()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
        print("   Последний чекпоинт сохранен")
    
    except Exception as e:
        print(f"\n\n❌ Ошибка во время обучения:")
        print(f"   {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()