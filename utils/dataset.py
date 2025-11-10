"""
utils/dataset.py
Датасет для загрузки и обработки LiDAR данных из LAS файлов
Совместимость с laspy 2.x
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import laspy
from pathlib import Path
from tqdm import tqdm
import pickle
import os


class LASDataset(Dataset):
    """
    Датасет для LiDAR данных из LAS файлов
    Разбивает облако точек на блоки фиксированного размера
    """
    
    def __init__(
        self,
        las_file,
        num_points=4096,
        block_size=50.0,
        stride=None,
        use_features=True,
        normalize=True,
        augment=False,
        cache_dir='cache'
    ):
        """
        Args:
            las_file: путь к LAS файлу
            num_points: количество точек в блоке
            block_size: размер блока в метрах
            stride: шаг между блоками (по умолчанию block_size/2)
            use_features: использовать ли доп. признаки (intensity, returns)
            normalize: нормализовать ли признаки
            augment: применять ли аугментации
            cache_dir: папка для кэширования
        """
        self.las_file = las_file
        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size / 2
        self.use_features = use_features
        self.normalize = normalize
        self.augment = augment
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Маппинг классов: 1,2,5,6 -> 0,1,2,3
        self.class_mapping = {1: 0, 2: 1, 5: 2, 6: 3}
        self.num_classes = 4
        
        print(f"\n📂 Загрузка LAS файла: {las_file}")
        self._load_data()
        self._create_blocks()
        
        print(f"✅ Датасет готов:")
        print(f"   • Блоков: {len(self.blocks)}")
        print(f"   • Точек в блоке: {self.num_points}")
        print(f"   • Размер блока: {self.block_size}m")
        print(f"   • Stride: {self.stride}m")
        print(f"   • Признаки: {'XYZ + intensity + returns' if use_features else 'Только XYZ'}")
    
    def _load_data(self):
        """Загрузка данных из LAS файла"""
        # Проверка кэша
        cache_file = self.cache_dir / f"{Path(self.las_file).stem}_preprocessed.pkl"
        
        if cache_file.exists():
            print(f"   📦 Загрузка из кэша: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.points = cached['points']
                self.labels = cached['labels']
                self.features = cached['features']
                self.bounds = cached['bounds']
            return
        
        # Загрузка LAS
        print(f"   📖 Чтение LAS файла...")
        las = laspy.read(self.las_file)
        
        # ========== ИСПРАВЛЕНИЕ: Совместимость с laspy 2.x ==========
        # Координаты
        xyz = np.vstack([
            np.array(las.x, dtype=np.float32),
            np.array(las.y, dtype=np.float32),
            np.array(las.z, dtype=np.float32)
        ]).T
        
        # Метки классов
        labels = np.array(las.classification, dtype=np.int32)
        
        # Дополнительные признаки
        features = {}
        
        # Intensity
        try:
            if hasattr(las, 'intensity'):
                features['intensity'] = np.array(las.intensity, dtype=np.float32)
        except:
            print("   ⚠️  Intensity не найден")
        
        # Return number
        try:
            if hasattr(las, 'return_number') or hasattr(las, 'return_num'):
                return_num = las.return_number if hasattr(las, 'return_number') else las.return_num
                features['return_number'] = np.array(return_num, dtype=np.float32)
        except:
            print("   ⚠️  Return number не найден")
        
        # Number of returns
        try:
            if hasattr(las, 'number_of_returns') or hasattr(las, 'num_returns'):
                num_returns = las.number_of_returns if hasattr(las, 'number_of_returns') else las.num_returns
                features['number_of_returns'] = np.array(num_returns, dtype=np.float32)
        except:
            print("   ⚠️  Number of returns не найден")
        
        print(f"   • Всего точек: {len(xyz):,}")
        print(f"   • Классы: {np.unique(labels)}")
        print(f"   • Признаков: {len(features)}")
        for key in features.keys():
            print(f"     - {key}: min={features[key].min():.2f}, max={features[key].max():.2f}")
        
        # Маппинг классов
        labels_mapped = np.full_like(labels, -1)
        for original, mapped in self.class_mapping.items():
            labels_mapped[labels == original] = mapped
        
        # Удаляем точки с неизвестными классами
        valid_mask = labels_mapped >= 0
        xyz = xyz[valid_mask]
        labels_mapped = labels_mapped[valid_mask]
        for key in features:
            features[key] = features[key][valid_mask]
        
        print(f"   • После фильтрации: {len(xyz):,} точек")
        
        # Распределение классов
        unique_labels, counts = np.unique(labels_mapped, return_counts=True)
        print(f"\n   📊 Распределение классов:")
        total = len(labels_mapped)
        for label, count in zip(unique_labels, counts):
            percent = 100.0 * count / total
            print(f"      Класс {label}: {count:,} точек ({percent:.2f}%)")
        
        # Нормализация координат (центрирование по минимуму)
        self.bounds = {
            'x_min': xyz[:, 0].min(),
            'y_min': xyz[:, 1].min(),
            'z_min': xyz[:, 2].min(),
            'x_max': xyz[:, 0].max(),
            'y_max': xyz[:, 1].max(),
            'z_max': xyz[:, 2].max(),
        }
        
        print(f"\n   🌍 Диапазоны координат:")
        print(f"      X: {self.bounds['x_min']:.2f} → {self.bounds['x_max']:.2f} (Δ={self.bounds['x_max']-self.bounds['x_min']:.2f}m)")
        print(f"      Y: {self.bounds['y_min']:.2f} → {self.bounds['y_max']:.2f} (Δ={self.bounds['y_max']-self.bounds['y_min']:.2f}m)")
        print(f"      Z: {self.bounds['z_min']:.2f} → {self.bounds['z_max']:.2f} (Δ={self.bounds['z_max']-self.bounds['z_min']:.2f}m)")
        
        xyz[:, 0] -= self.bounds['x_min']
        xyz[:, 1] -= self.bounds['y_min']
        
        self.points = xyz
        self.labels = labels_mapped
        self.features = features
        
        # Кэширование
        print(f"\n   💾 Сохранение в кэш: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'points': self.points,
                'labels': self.labels,
                'features': self.features,
                'bounds': self.bounds
            }, f)
    
    def _create_blocks(self):
        """Разбиение облака точек на блоки"""
        cache_file = self.cache_dir / f"{Path(self.las_file).stem}_blocks_{self.block_size}_{self.stride}.pkl"
        
        if cache_file.exists():
            print(f"\n   📦 Загрузка блоков из кэша: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.blocks = pickle.load(f)
            return
        
        print(f"\n   🔨 Создание блоков...")
        
        x_min, y_min = 0, 0
        x_max = self.bounds['x_max'] - self.bounds['x_min']
        y_max = self.bounds['y_max'] - self.bounds['y_min']
        
        blocks = []
        
        x_start = x_min
        pbar = tqdm(desc="   Создание блоков", unit="row")
        
        while x_start < x_max:
            y_start = y_min
            while y_start < y_max:
                # Маска точек в блоке
                mask = (
                    (self.points[:, 0] >= x_start) &
                    (self.points[:, 0] < x_start + self.block_size) &
                    (self.points[:, 1] >= y_start) &
                    (self.points[:, 1] < y_start + self.block_size)
                )
                
                indices = np.where(mask)[0]
                
                # Пропускаем пустые блоки или слишком маленькие
                if len(indices) >= 100:  # Минимум 100 точек
                    blocks.append({
                        'indices': indices,
                        'x_start': x_start,
                        'y_start': y_start
                    })
                
                y_start += self.stride
            x_start += self.stride
            pbar.update(1)
        
        pbar.close()
        
        self.blocks = blocks
        print(f"   • Создано блоков: {len(blocks)}")
        
        # Статистика блоков
        block_sizes = [len(b['indices']) for b in blocks]
        print(f"   • Точек в блоке: min={min(block_sizes)}, max={max(block_sizes)}, avg={np.mean(block_sizes):.0f}")
        
        # Кэширование
        print(f"   💾 Сохранение блоков в кэш...")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.blocks, f)
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        """
        Возвращает блок точек с признаками и метками
        Returns:
            points: (num_points, 3+feature_dim) - xyz + признаки
            labels: (num_points,) - метки классов
        """
        block = self.blocks[idx]
        indices = block['indices']
        
        # Получаем точки блока
        block_points = self.points[indices].copy()
        block_labels = self.labels[indices].copy()
        
        # Центрирование блока
        centroid = block_points[:, :2].mean(axis=0)
        block_points[:, 0] -= centroid[0]
        block_points[:, 1] -= centroid[1]
        
        # Сэмплирование/дополнение до num_points
        if len(block_points) >= self.num_points:
            # Random sampling
            choice = np.random.choice(len(block_points), self.num_points, replace=False)
        else:
            # Repeat points
            choice = np.random.choice(len(block_points), self.num_points, replace=True)
        
        block_points = block_points[choice]
        block_labels = block_labels[choice]
        
        # Добавление признаков
        if self.use_features:
            feature_list = []
            
            # Intensity
            if 'intensity' in self.features:
                intensity = self.features['intensity'][indices][choice]
                if self.normalize:
                    intensity = intensity / 255.0  # Нормализация в [0, 1]
                feature_list.append(intensity.reshape(-1, 1))
            
            # Return number
            if 'return_number' in self.features:
                return_num = self.features['return_number'][indices][choice]
                feature_list.append(return_num.reshape(-1, 1))
            
            # Number of returns
            if 'number_of_returns' in self.features:
                num_returns = self.features['number_of_returns'][indices][choice]
                feature_list.append(num_returns.reshape(-1, 1))
            
            if feature_list:
                features = np.concatenate(feature_list, axis=1)
                block_points = np.concatenate([block_points, features], axis=1)
        
        # Аугментации
        if self.augment:
            block_points = self._augment(block_points)
        
        # Нормализация координат
        if self.normalize:
            block_points[:, :3] = self._normalize_coords(block_points[:, :3])
        
        return torch.FloatTensor(block_points), torch.LongTensor(block_labels)
    
    def _normalize_coords(self, coords):
        """Нормализация координат в [-1, 1]"""
        centroid = coords.mean(axis=0)
        coords = coords - centroid
        max_dist = np.max(np.sqrt(np.sum(coords**2, axis=1)))
        if max_dist > 0:
            coords = coords / max_dist
        return coords
    
    def _augment(self, points):
        """
        Аугментации для LiDAR данных
        - Random rotation вокруг Z
        - Random scaling
        - Random jittering
        """
        # Rotation вокруг Z оси
        if np.random.random() > 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            points[:, :3] = points[:, :3] @ rotation_matrix.T
        
        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            points[:, :3] *= scale
        
        # Jittering
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.02, points[:, :3].shape)
            points[:, :3] += noise
        
        return points
    
    def get_class_distribution(self):
        """Получить распределение классов в датасете"""
        all_labels = []
        print("\n📊 Анализ распределения классов в блоках...")
        
        # Выборка для ускорения
        sample_size = min(100, len(self))
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        for i in tqdm(indices, desc="Подсчет классов"):
            _, labels = self[i]
            all_labels.append(labels.numpy())
        
        all_labels = np.concatenate(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        
        distribution = {}
        total = len(all_labels)
        
        print("\nРаспределение классов (выборка):")
        for cls, count in zip(unique, counts):
            percent = 100.0 * count / total
            distribution[int(cls)] = int(count)
            print(f"   Класс {cls}: {count:,} точек ({percent:.2f}%)")
        
        return distribution


if __name__ == '__main__':
    # Тестирование датасета
    print("🧪 Тестирование LASDataset\n")
    
    las_file = 'datasets/raw/NEONDSSampleLiDARPointCloud.las'
    
    if not os.path.exists(las_file):
        print(f"❌ Файл не найден: {las_file}")
        exit(1)
    
    # Создание датасета
    dataset = LASDataset(
        las_file=las_file,
        num_points=4096,
        block_size=50.0,
        stride=25.0,
        use_features=True,
        normalize=True,
        augment=True
    )
    
    # Тест загрузки
    print("\n🧪 Тест загрузки блока:")
    points, labels = dataset[0]
    print(f"   Points shape: {points.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Points range: [{points.min():.3f}, {points.max():.3f}]")
    print(f"   Unique labels: {labels.unique().tolist()}")
    
    # Распределение классов
    distribution = dataset.get_class_distribution()
    
    print("\n✅ Тест завершен!")