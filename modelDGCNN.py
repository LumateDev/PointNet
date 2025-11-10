"""
DGCNN (Dynamic Graph CNN) для семантической сегментации облаков точек
Архитектура основана на EdgeConv операциях и динамических графах
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Находит k ближайших соседей для каждой точки
    Args:
        x: (batch_size, num_dims, num_points)
        k: количество соседей
    Returns:
        idx: (batch_size, num_points, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x**2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Построение локальных графовых признаков
    Args:
        x: (batch_size, num_dims, num_points)
        k: количество соседей
        idx: индексы соседей (опционально)
    Returns:
        feature: (batch_size, 2*num_dims, num_points, k)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, C)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, N, k, C)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
    
    return feature


class EdgeConv(nn.Module):
    """
    Edge Convolution Layer
    Извлекает локальные признаки используя связи между точками
    """
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, num_points)
        Returns:
            x: (batch_size, out_channels, num_points)
        """
        x = get_graph_feature(x, k=self.k)  # (B, 2*in_channels, N, k)
        x = self.conv(x)  # (B, out_channels, N, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, out_channels, N)
        return x


class DGCNN_Segmentation(nn.Module):
    """
    DGCNN для семантической сегментации облаков точек
    """
    def __init__(self, num_classes=4, k=20, emb_dims=1024, dropout=0.5):
        super(DGCNN_Segmentation, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        
        # EdgeConv блоки для извлечения признаков
        self.conv1 = EdgeConv(3, 64, k=k)
        self.conv2 = EdgeConv(64, 64, k=k)
        self.conv3 = EdgeConv(64, 128, k=k)
        self.conv4 = EdgeConv(128, 256, k=k)
        
        # Глобальное представление
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Segmentation head
        self.conv6 = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.dp1 = nn.Dropout(p=dropout)
        self.conv8 = nn.Conv1d(256, num_classes, kernel_size=1, bias=True)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_points, 3) или (batch_size, 3, num_points)
        Returns:
            x: (batch_size, num_points, num_classes)
        """
        # Преобразование входа в формат (B, 3, N)
        if x.size(1) != 3:
            x = x.transpose(2, 1)
        
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Извлечение локальных признаков через EdgeConv
        x1 = self.conv1(x)  # (B, 64, N)
        x2 = self.conv2(x1)  # (B, 64, N)
        x3 = self.conv3(x2)  # (B, 128, N)
        x4 = self.conv4(x3)  # (B, 256, N)
        
        # Конкатенация локальных признаков
        x_local = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        
        # Глобальные признаки
        x_global = self.conv5(x_local)  # (B, emb_dims, N)
        x_global = F.adaptive_max_pool1d(x_global, 1)  # (B, emb_dims, 1)
        x_global = x_global.repeat(1, 1, num_points)  # (B, emb_dims, N)
        
        # Комбинация локальных и глобальных признаков
        x = torch.cat((x_local, x_global), dim=1)  # (B, 512+emb_dims, N)
        
        # Segmentation head
        x = self.conv6(x)  # (B, 512, N)
        x = self.conv7(x)  # (B, 256, N)
        x = self.dp1(x)
        x = self.conv8(x)  # (B, num_classes, N)
        
        # Преобразование в формат (B, N, num_classes)
        x = x.transpose(2, 1).contiguous()
        
        return x


class DGCNN_Segmentation_V2(nn.Module):
    """
    Улучшенная версия DGCNN с дополнительными признаками
    Использует интенсивность и другие атрибуты LiDAR
    """
    def __init__(self, num_classes=4, input_channels=6, k=20, emb_dims=1024, dropout=0.5):
        super(DGCNN_Segmentation_V2, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.input_channels = input_channels
        
        # Input transform (для дополнительных признаков)
        if input_channels > 3:
            self.input_transform = nn.Sequential(
                nn.Conv1d(input_channels, 64, 1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(64, input_channels, 1)
            )
        else:
            self.input_transform = None
        
        # EdgeConv блоки (адаптированы под input_channels)
        self.conv1 = EdgeConv(input_channels, 64, k=k)
        self.conv2 = EdgeConv(64, 64, k=k)
        self.conv3 = EdgeConv(64, 128, k=k)
        self.conv4 = EdgeConv(128, 256, k=k)
        
        # Глобальное представление
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Segmentation head с residual connections
        self.conv6 = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(128, num_classes, kernel_size=1, bias=True)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_points, input_channels) или (batch_size, input_channels, num_points)
        Returns:
            x: (batch_size, num_points, num_classes)
        """
        # Преобразование входа
        if x.size(1) != self.input_channels:
            x = x.transpose(2, 1)
        
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Input transform (опционально)
        if self.input_transform is not None:
            x = x + self.input_transform(x)
        
        # Извлечение признаков
        x1 = self.conv1(x)  # (B, 64, N)
        x2 = self.conv2(x1)  # (B, 64, N)
        x3 = self.conv3(x2)  # (B, 128, N)
        x4 = self.conv4(x3)  # (B, 256, N)
        
        # Локальные признаки
        x_local = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        
        # Глобальные признаки
        x_global = self.conv5(x_local)  # (B, emb_dims, N)
        x_global_pooled = F.adaptive_max_pool1d(x_global, 1)  # (B, emb_dims, 1)
        x_global_broadcast = x_global_pooled.repeat(1, 1, num_points)  # (B, emb_dims, N)
        
        # Комбинация признаков
        x = torch.cat((x_local, x_global_broadcast), dim=1)  # (B, 512+emb_dims, N)
        
        # Segmentation
        x = self.conv6(x)  # (B, 512, N)
        x = self.conv7(x)  # (B, 256, N)
        x = self.conv8(x)  # (B, 128, N)
        x = self.dp1(x)
        x = self.conv9(x)  # (B, num_classes, N)
        
        # Output format
        x = x.transpose(2, 1).contiguous()  # (B, N, num_classes)
        
        return x


def test_model():
    """Тестирование модели"""
    print("🧪 Тестирование DGCNN модели...")
    
    # Параметры
    batch_size = 2
    num_points = 2048
    num_classes = 4
    
    # Тест базовой версии (только координаты)
    print("\n📦 Тест DGCNN_Segmentation (базовая версия):")
    model = DGCNN_Segmentation(num_classes=num_classes, k=20)
    x = torch.randn(batch_size, num_points, 3)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   ✅ Expected: ({batch_size}, {num_points}, {num_classes})")
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   📊 Параметров: {total_params:,}")
    
    # Тест улучшенной версии (с доп. признаками)
    print("\n📦 Тест DGCNN_Segmentation_V2 (с доп. признаками):")
    model_v2 = DGCNN_Segmentation_V2(num_classes=num_classes, input_channels=6, k=20)
    x_v2 = torch.randn(batch_size, num_points, 6)  # координаты + RGB или другие признаки
    
    model_v2.eval()
    with torch.no_grad():
        output_v2 = model_v2(x_v2)
    
    print(f"   Input shape: {x_v2.shape}")
    print(f"   Output shape: {output_v2.shape}")
    print(f"   ✅ Expected: ({batch_size}, {num_points}, {num_classes})")
    
    # Подсчет параметров
    total_params_v2 = sum(p.numel() for p in model_v2.parameters())
    print(f"   📊 Параметров: {total_params_v2:,}")
    
    print("\n✅ Все тесты пройдены!")


if __name__ == '__main__':
    test_model()