"""
modelDGCNN.py
Dynamic Graph CNN для семантической сегментации LiDAR данных
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def knn(x, k):
    """
    K-Nearest Neighbors в пространстве признаков
    Args:
        x: (B, C, N) - точки
        k: число соседей
    Returns:
        idx: (B, N, k) - индексы k ближайших соседей
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x**2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Построение графовых признаков (EdgeConv)
    Args:
        x: (B, C, N)
        k: число соседей
        idx: предвычисленные индексы соседей
        dim9: использовать ли 9D признаки (xyz + признаки)
    Returns:
        feature: (B, 2C, N, k) - признаки ребер графа
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (B, N, k)
        else:
            idx = knn(x[:, 6:], k=k)  # Используем только xyz для KNN
    
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, C)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, N, k, C)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # (B, 2C, N, k) - [разность соседей, центральная точка]
    
    return feature


# ==================== DGCNN LAYERS ====================

class EdgeConvBlock(nn.Module):
    """
    Edge Convolution Block
    Обрабатывает графовые признаки: для каждой точки агрегирует информацию от k соседей
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        """
        x: (B, C, N)
        return: (B, out_channels, N)
        """
        x = get_graph_feature(x, k=self.k)  # (B, 2C, N, k)
        x = self.conv(x)  # (B, out_channels, N, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, out_channels, N)
        return x


# ==================== DGCNN MODEL ====================

class DGCNN_LiDAR(nn.Module):
    """
    DGCNN для семантической сегментации LiDAR данных
    
    Архитектура:
    1. Feature Extraction: 4 EdgeConv блока с увеличением каналов
       - EdgeConv1: in → 64 (k=20)
       - EdgeConv2: 64 → 64 (k=20)
       - EdgeConv3: 64 → 128 (k=20)
       - EdgeConv4: 128 → 256 (k=20)
    
    2. Global Feature Aggregation: Conv1d для объединения всех уровней
       - Concat всех признаков: 64+64+128+256 = 512
       - Conv1d: 512 → 1024
    
    3. Decoder: MLP для классификации каждой точки
       - Concat локальных + глобальных: 1024+512 = 1536
       - Conv1d: 1536 → 512 → 256 → 128
       - Output: 128 → num_classes
    
    Функции активации: LeakyReLU(0.2)
    Dropout: 0.5 перед финальным слоем
    """
    
    def __init__(self, num_classes=4, k=20, use_features=True, feature_dim=3, dropout=0.5):
        """
        Args:
            num_classes: количество классов сегментации
            k: число соседей для KNN
            use_features: использовать ли дополнительные признаки (intensity, returns)
            feature_dim: размерность доп. признаков
            dropout: вероятность dropout
        """
        super().__init__()
        
        self.k = k
        self.num_classes = num_classes
        self.use_features = use_features
        
        # Входная размерность: 3 (xyz) + feature_dim (intensity, returns, etc.)
        input_channels = 3 + (feature_dim if use_features else 0)
        
        # ========== ENCODER: Edge Convolutions ==========
        # Блок 1: input → 64
        self.edgeconv1 = EdgeConvBlock(input_channels, 64, k=k)
        
        # Блок 2: 64 → 64
        self.edgeconv2 = EdgeConvBlock(64, 64, k=k)
        
        # Блок 3: 64 → 128
        self.edgeconv3 = EdgeConvBlock(64, 128, k=k)
        
        # Блок 4: 128 → 256
        self.edgeconv4 = EdgeConvBlock(128, 256, k=k)
        
        # ========== GLOBAL FEATURE AGGREGATION ==========
        # Объединяем все уровни: 64 + 64 + 128 + 256 = 512
        self.conv_global = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # ========== DECODER: Segmentation Head ==========
        # Concat: global (1024) + local (512) = 1536
        self.conv_decode1 = nn.Sequential(
            nn.Conv1d(1536, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv_decode2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.dp = nn.Dropout(p=dropout)
        
        self.conv_decode3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Финальный классификатор
        self.conv_out = nn.Conv1d(128, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, C) где C = 3 + feature_dim (xyz + intensity, returns, etc.)
        Returns:
            (B, N, num_classes) - логиты для каждого класса
        """
        B, N, C = x.shape
        
        # Transpose для Conv1d/Conv2d: (B, N, C) → (B, C, N)
        x = x.transpose(2, 1).contiguous()  # (B, C, N)
        
        # ========== ENCODER ==========
        x1 = self.edgeconv1(x)      # (B, 64, N)
        x2 = self.edgeconv2(x1)     # (B, 64, N)
        x3 = self.edgeconv3(x2)     # (B, 128, N)
        x4 = self.edgeconv4(x3)     # (B, 256, N)
        
        # Объединяем все уровни признаков
        x_local = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        
        # ========== GLOBAL FEATURES ==========
        x_global = self.conv_global(x_local)  # (B, 1024, N)
        
        # Max pooling для глобального контекста
        x_global_max = x_global.max(dim=-1, keepdim=True)[0]  # (B, 1024, 1)
        x_global_max = x_global_max.repeat(1, 1, N)  # (B, 1024, N)
        
        # ========== DECODER ==========
        # Concatenate локальные + глобальные признаки
        x_concat = torch.cat((x_local, x_global_max), dim=1)  # (B, 1536, N)
        
        x = self.conv_decode1(x_concat)  # (B, 512, N)
        x = self.conv_decode2(x)         # (B, 256, N)
        x = self.dp(x)
        x = self.conv_decode3(x)         # (B, 128, N)
        x = self.conv_out(x)             # (B, num_classes, N)
        
        # Transpose обратно: (B, num_classes, N) → (B, N, num_classes)
        x = x.transpose(2, 1).contiguous()
        
        return x


# ==================== SUMMARY FUNCTION ====================

def model_summary(model, input_size=(2, 4096, 6), device='cuda'):
    """
    Вывод информации о модели
    """
    model = model.to(device)
    
    print("=" * 80)
    print(f"{'DGCNN MODEL SUMMARY':^80}")
    print("=" * 80)
    
    # Параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Параметры модели:")
    print(f"   Всего параметров: {total_params:,}")
    print(f"   Обучаемых: {trainable_params:,}")
    print(f"   Размер модели: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Архитектура
    print(f"\n🏗️  Архитектура:")
    print(f"   Классов: {model.num_classes}")
    print(f"   K соседей: {model.k}")
    print(f"   Использование признаков: {model.use_features}")
    
    print(f"\n🔧 Слои:")
    print(f"   ┌─ ENCODER (Edge Convolutions):")
    print(f"   │  ├─ EdgeConv1: input → 64 channels")
    print(f"   │  ├─ EdgeConv2: 64 → 64 channels")
    print(f"   │  ├─ EdgeConv3: 64 → 128 channels")
    print(f"   │  └─ EdgeConv4: 128 → 256 channels")
    print(f"   │")
    print(f"   ├─ GLOBAL AGGREGATION:")
    print(f"   │  └─ Conv1d: 512 → 1024 channels")
    print(f"   │")
    print(f"   └─ DECODER (Segmentation Head):")
    print(f"      ├─ Conv1d: 1536 → 512 channels")
    print(f"      ├─ Conv1d: 512 → 256 channels")
    print(f"      ├─ Dropout: p=0.5")
    print(f"      ├─ Conv1d: 256 → 128 channels")
    print(f"      └─ Output: 128 → {model.num_classes} classes")
    
    print(f"\n⚡ Функции активации: LeakyReLU(negative_slope=0.2)")
    print(f"   Нормализация: BatchNorm2d/BatchNorm1d")
    
    # Тест forward pass
    print(f"\n🧪 Тестирование forward pass...")
    x = torch.randn(input_size).to(device)
    with torch.no_grad():
        out = model(x)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   ✅ Forward pass успешен!")
    
    print("=" * 80)


# ==================== TESTING ====================

if __name__ == '__main__':
    print("\n🧪 Тестирование DGCNN для LiDAR сегментации\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Параметры
    num_classes = 4
    k = 20
    use_features = True
    feature_dim = 3  # intensity, return_number, number_of_returns
    
    # Создание модели
    model = DGCNN_LiDAR(
        num_classes=num_classes,
        k=k,
        use_features=use_features,
        feature_dim=feature_dim,
        dropout=0.5
    )
    
    # Summary
    input_channels = 3 + (feature_dim if use_features else 0)
    model_summary(model, input_size=(2, 4096, input_channels), device=device)
    
    # Тест backward
    print(f"\n🔬 Тестирование backward pass...")
    model.train()
    x = torch.randn(2, 4096, input_channels).to(device)
    labels = torch.randint(0, num_classes, (2, 4096)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out.view(-1, num_classes), labels.view(-1))
    loss.backward()
    optimizer.step()
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✅ Backward pass успешен!")
    
    print("\n" + "=" * 80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 80)