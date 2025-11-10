import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def square_distance(src, dst):
    """
    Вычисление квадратов расстояний между точками
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Выбор точек по индексам
    points: (B, N, C)
    idx: (B, S) или (B, S, K)
    return: (B, S, C) или (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS)
    xyz: (B, N, 3)
    npoint: число точек для сэмплирования
    return: (B, npoint) индексы выбранных точек
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball Query - поиск точек в радиусе
    radius: радиус поиска
    nsample: максимальное число соседей
    xyz: (B, N, 3) все точки
    new_xyz: (B, S, 3) центры запросов
    return: (B, S, nsample) индексы соседей
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


# ==================== POINTNET++ MODULES ====================

class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Module - базовый блок PointNet++
    Выполняет: Sampling -> Grouping -> PointNet
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3  # +3 для относительных координат
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        xyz: (B, N, 3)
        points: (B, N, C) или None
        return: new_xyz (B, npoint, 3), new_points (B, npoint, mlp[-1])
        """
        B, N, _ = xyz.shape
        
        if self.group_all:
            # Глобальный pooling
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = points.unsqueeze(1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            # FPS sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            
            # Ball query grouping
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
            
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz_norm
        
        # (B, npoint, nsample, C) -> (B, C, nsample, npoint)
        new_points = new_points.permute(0, 3, 2, 1)
        
        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation Module - upsampling для декодера
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Интерполяция признаков ОТ xyz1 К xyz2
        xyz1: (B, N, 3) - точки источника (МЕНЬШЕ точек)
        xyz2: (B, M, 3) - точки назначения (БОЛЬШЕ точек)
        points1: (B, N, C1) - признаки источника
        points2: (B, M, C2) - признаки назначения (skip connection)
        return: (B, M, mlp[-1])
        """
        B, N, C = xyz1.shape
        _, M, _ = xyz2.shape
        
        if N == 1:
            # Глобальные признаки - просто повторяем
            interpolated_points = points1.repeat(1, M, 1)
        else:
            # 3-NN интерполяция
            dists = square_distance(xyz2, xyz1)  # (B, M, N)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # (B, M, 3)
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(
                index_points(points1, idx) * weight.view(B, M, 3, 1),
                dim=2
            )
        
        # Concatenate со skip connection
        if points2 is not None:
            new_points = torch.cat([interpolated_points, points2], dim=-1)
        else:
            new_points = interpolated_points
        
        # MLP
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = new_points.permute(0, 2, 1)
        return new_points


class PointNet2LiDAR(nn.Module):
    """
    PointNet++ для семантической сегментации LiDAR данных
    Поддерживает дополнительные признаки: intensity, return_number, etc.
    """
    def __init__(self, num_classes=4, use_features=True, feature_dim=3):
        """
        Args:
            num_classes: количество классов (для вашего датасета: 4)
            use_features: использовать ли дополнительные признаки
            feature_dim: размерность дополнительных признаков
                        (intensity, return_number, number_of_returns)
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_features = use_features
        
        # Входной канал: 0 (только xyz) или feature_dim (с доп. признаками)
        in_channel = feature_dim if use_features else 0
        
        # ========== ENCODER ==========
        # Уровень 1: 2048 -> 1024 точки
        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.1,
            nsample=32,
            in_channel=in_channel,
            mlp=[32, 32, 64]
        )
        
        # Уровень 2: 1024 -> 256 точек
        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=64,
            mlp=[64, 64, 128]
        )
        
        # Уровень 3: 256 -> 64 точки
        self.sa3 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=128,
            mlp=[128, 128, 256]
        )
        
        # Уровень 4: 64 -> 16 точек
        self.sa4 = PointNetSetAbstraction(
            npoint=16,
            radius=0.8,
            nsample=32,
            in_channel=256,
            mlp=[256, 256, 512]
        )
        
        # ========== DECODER ==========
        # FP4: 16 -> 64 точки (512 + 256 = 768)
        self.fp4 = PointNetFeaturePropagation(
            in_channel=768,
            mlp=[256, 256]
        )
        
        # FP3: 64 -> 256 точек (256 + 128 = 384)
        self.fp3 = PointNetFeaturePropagation(
            in_channel=384,
            mlp=[256, 256]
        )
        
        # FP2: 256 -> 1024 точки (256 + 64 = 320)
        self.fp2 = PointNetFeaturePropagation(
            in_channel=320,
            mlp=[256, 128]
        )
        
        # FP1: 1024 -> 2048 точек (128 + feature_dim = 128+3)
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + (feature_dim if use_features else 0),
            mlp=[128, 128, 128]
        )
        
        # ========== CLASSIFIER ==========
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3+C) где первые 3 канала - координаты XYZ,
                 остальные C каналов - признаки (intensity, etc.)
        Returns:
            (B, N, num_classes) - логиты для каждого класса
        """
        B, N, C = xyz.shape
        
        # Разделяем координаты и признаки
        l0_xyz = xyz[:, :, :3].contiguous()  # (B, N, 3)
        
        if self.use_features and C > 3:
            l0_points = xyz[:, :, 3:].contiguous()  # (B, N, feature_dim)
        else:
            l0_points = None
        
        # ========== ENCODER ==========
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # (B, 1024, 3), (B, 1024, 64)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 256, 3), (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 64, 3), (B, 64, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 16, 3), (B, 16, 512)
        
        # ========== DECODER ==========
        # ⚠️ ВАЖНО: интерполяция идет ОТ меньшего К большему!
        l3_points = self.fp4(l4_xyz, l3_xyz, l4_points, l3_points)  # (B, 64, 256)
        l2_points = self.fp3(l3_xyz, l2_xyz, l3_points, l2_points)  # (B, 256, 256)
        l1_points = self.fp2(l2_xyz, l1_xyz, l2_points, l1_points)  # (B, 1024, 128)
        l0_points = self.fp1(l1_xyz, l0_xyz, l1_points, l0_points)  # (B, N, 128)
        
        # ========== CLASSIFIER ==========
        x = l0_points.permute(0, 2, 1)  # (B, 128, N)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)  # (B, num_classes, N)
        x = x.permute(0, 2, 1)  # (B, N, num_classes)
        
        return x


# ==================== ФУНКЦИЯ ПОТЕРЬ С ВЕСАМИ ====================

class WeightedCrossEntropyLoss(nn.Module):
    """
    Взвешенная Cross Entropy для несбалансированных классов
    """
    def __init__(self, class_weights=None, ignore_index=-1):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, N, num_classes)
            targets: (B, N) - метки классов
        """
        B, N, C = predictions.shape
        
        # Flatten
        predictions = predictions.reshape(-1, C)
        targets = targets.reshape(-1)
        
        # Cross Entropy
        loss = F.cross_entropy(
            predictions,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )
        
        return loss


def compute_class_weights(class_counts, mode='inverse'):
    """
    Вычисление весов классов для балансировки
    
    Args:
        class_counts: dict {класс: количество} или list с количеством для каждого класса
        mode: 'inverse' или 'effective'
    """
    if isinstance(class_counts, dict):
        counts = np.array(list(class_counts.values()))
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
    
    # Нормализация
    weights = weights / weights.sum() * len(weights)
    
    return torch.FloatTensor(weights)


# ==================== ТЕСТИРОВАНИЕ ====================

def test_pointnet2():
    """Тестирование PointNet++ для LiDAR"""
    print("🧪 Тестирование PointNet++ для LiDAR\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}\n")
    
    # ========== Параметры вашего датасета ==========
    num_classes = 4  # Классы: 1, 2, 5, 6
    B, N = 2, 2048
    
    # Классы из вашего файла
    class_distribution = {
        1: 855675,    # Unclassified (12.95%)
        2: 4145402,   # Ground (62.72%)
        5: 1569036,   # High Vegetation (23.74%)
        6: 39716      # Building (0.60%)
    }
    
    # Вычисляем веса классов
    class_weights = compute_class_weights(class_distribution, mode='effective')
    print("⚖️  Веса классов (для балансировки):")
    for i, (cls, count) in enumerate(class_distribution.items()):
        percent = count / sum(class_distribution.values()) * 100
        print(f"   Класс {cls}: вес={class_weights[i]:.4f} (встречается {percent:.2f}%)")
    print()
    
    # ========== ТЕСТ 1: Только XYZ ==========
    print("=" * 70)
    print("📊 ТЕСТ 1: PointNet++ (только XYZ координаты)")
    print("=" * 70)
    
    model_xyz = PointNet2LiDAR(num_classes=num_classes, use_features=False).to(device)
    x_xyz = torch.randn(B, N, 3).to(device)
    
    try:
        out = model_xyz(x_xyz)
        print(f"✅ Forward pass успешен")
        print(f"   Input:  {x_xyz.shape}")
        print(f"   Output: {out.shape}")
        
        total_params = sum(p.numel() for p in model_xyz.parameters())
        print(f"   Параметров: {total_params:,}")
        print(f"   Размер: {total_params * 4 / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print()
    
    # ========== ТЕСТ 2: XYZ + Features ==========
    print("=" * 70)
    print("📊 ТЕСТ 2: PointNet++ (XYZ + intensity + returns)")
    print("=" * 70)
    
    # XYZ + 3 признака (intensity, return_number, number_of_returns)
    feature_dim = 3
    model_features = PointNet2LiDAR(
        num_classes=num_classes,
        use_features=True,
        feature_dim=feature_dim
    ).to(device)
    
    x_features = torch.randn(B, N, 3 + feature_dim).to(device)
    
    try:
        out = model_features(x_features)
        print(f"✅ Forward pass успешен")
        print(f"   Input:  {x_features.shape} (XYZ + {feature_dim} features)")
        print(f"   Output: {out.shape}")
        
        total_params = sum(p.numel() for p in model_features.parameters())
        print(f"   Параметров: {total_params:,}")
        print(f"   Размер: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Тест функции потерь
        print("\n📉 Тестирование функции потерь:")
        targets = torch.randint(0, num_classes, (B, N)).to(device)
        
        criterion = WeightedCrossEntropyLoss(class_weights.to(device))
        loss = criterion(out, targets)
        print(f"   Loss (weighted): {loss.item():.4f}")
        
        # Backpropagation
        loss.backward()
        print(f"   ✅ Backpropagation успешен")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Все тесты завершены!")
    print("=" * 70)
    
    # ========== Рекомендации ==========
    print("\n💡 РЕКОМЕНДАЦИИ ДЛЯ ОБУЧЕНИЯ:\n")
    print("1️⃣  Используйте взвешенную функцию потерь (класс 6 встречается редко!)")
    print("2️⃣  Добавьте признаки: intensity, return_number, number_of_returns")
    print("3️⃣  Нормализуйте признаки перед обучением:")
    print("    - XYZ: центрируйте по среднему блока")
    print("    - Intensity: нормализуйте в [0, 1]")
    print("    - Returns: можно оставить как есть (уже малые значения)")
    print("4️⃣  Learning rate: начните с 1e-3 с cosine scheduler")
    print("5️⃣  Batch size: 8-16 (в зависимости от памяти GPU)")
    print("6️⃣  Data augmentation: rotation, jittering, scaling")


if __name__ == '__main__':
    test_pointnet2()