"""
utils/losses.py
Функции потерь для семантической сегментации с дисбалансом классов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов
    Фокусируется на сложных примерах
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        Args:
            alpha: веса классов (tensor размера num_classes)
            gamma: фокусирующий параметр (обычно 2.0)
            reduction: 'mean' или 'sum'
            ignore_index: индекс для игнорирования
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B*N, num_classes) - логиты
            targets: (B*N,) - метки классов
        """
        # Cross entropy
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        # p_t
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = focal_term * ce_loss
        
        # Alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Получаем веса для каждого примера
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss для сегментации
    Хорошо работает с несбалансированными классами
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets, num_classes=4):
        """
        Args:
            inputs: (B*N, num_classes) - логиты
            targets: (B*N,) - метки
        """
        # Softmax
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss
        return 1.0 - dice.mean()


def compute_class_weights(class_counts, mode='effective', device='cpu'):
    """
    Вычисление весов классов для балансировки
    
    Args:
        class_counts: dict {класс: количество} или list/array
        mode: 'inverse', 'effective', 'sqrt'
        device: устройство для тензора
    
    Returns:
        weights: tensor с весами для каждого класса
    """
    if isinstance(class_counts, dict):
        # Сортируем по ключу класса
        sorted_items = sorted(class_counts.items())
        counts = np.array([count for _, count in sorted_items])
    else:
        counts = np.array(class_counts)
    
    if mode == 'inverse':
        # Обратно пропорционально количеству
        weights = 1.0 / (counts + 1e-6)
    
    elif mode == 'effective':
        # Effective number of samples (для сильного дисбаланса)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    
    elif mode == 'sqrt':
        # Квадратный корень (менее агрессивная балансировка)
        weights = 1.0 / np.sqrt(counts + 1e-6)
    
    else:
        weights = np.ones_like(counts, dtype=np.float32)
    
    # Нормализация
    weights = weights / weights.sum() * len(weights)
    
    return torch.FloatTensor(weights).to(device)


if __name__ == '__main__':
    print("🧪 Тестирование функций потерь\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Параметры
    num_classes = 4
    B, N = 2, 4096
    
    # Дисбаланс классов (как в вашем датасете)
    class_distribution = {
        0: 855675,    # Class 1 (12.95%)
        1: 4145402,   # Class 2 (62.72%)
        2: 1569036,   # Class 5 (23.74%)
        3: 39716      # Class 6 (0.60%)
    }
    
    # Веса классов
    weights = compute_class_weights(class_distribution, mode='effective', device=device)
    print("⚖️ Веса классов:")
    for i, w in enumerate(weights):
        count = list(class_distribution.values())[i]
        percent = 100.0 * count / sum(class_distribution.values())
        print(f"   Класс {i}: вес={w:.4f} (встречается {percent:.2f}%)")
    
    # Тест Focal Loss
    print("\n📉 Focal Loss:")
    inputs = torch.randn(B * N, num_classes).to(device)
    targets = torch.randint(0, num_classes, (B * N,)).to(device)
    
    focal_loss = FocalLoss(alpha=weights, gamma=2.0)
    loss_focal = focal_loss(inputs, targets)
    print(f"   Loss: {loss_focal.item():.4f}")
    
    # Тест Dice Loss
    print("\n🎲 Dice Loss:")
    dice_loss = DiceLoss()
    loss_dice = dice_loss(inputs, targets, num_classes=num_classes)
    print(f"   Loss: {loss_dice.item():.4f}")
    
    print("\n✅ Все тесты пройдены!")