import argparse
import os
import platform
import sys
from typing import Optional, Tuple

import laspy
import numpy as np
import torch


SEPARATOR = "=" * 60


def diagnose_environment() -> None:
    """Выводит информацию о доступном окружении."""
    print(SEPARATOR)
    print("ДИАГНОСТИКА ОКРУЖЕНИЯ")
    print(SEPARATOR)
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Python версия: {sys.version}")
    print(f"Платформа: {platform.platform()}")
    print(f"Архитектура: {platform.architecture()}")

    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA доступен: {cuda_available}")
    if cuda_available:
        print(f"Количество GPU: {torch.cuda.device_count()}")
        device_id = torch.cuda.current_device()
        print(f"Текущий GPU: {device_id}")
        print(f"Имя GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"CUDA версия: {torch.version.cuda}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Всего GPU памяти: {gpu_memory:.2f} GB")
    else:
        print("CUDA недоступна — будет использоваться CPU")

    cpu_threads = torch.get_num_threads()
    print(f"Количество CPU потоков: {cpu_threads}")


def analyze_las_dataset(las_file: str) -> Optional[laspy.LasData]:
    """Анализирует LAS-файл и выводит его характеристики."""
    print(f"\n{SEPARATOR}")
    print(f"АНАЛИЗ LAS ДАТАСЕТА: {las_file}")
    print(SEPARATOR)

    try:
        las = laspy.read(las_file)
    except FileNotFoundError:
        print(f"Файл {las_file} не найден. Пропускаем анализ датасета.")
        return None
    except Exception as exc:
        print(f"Ошибка при чтении LAS файла: {exc}")
        return None

    print(f"Формат файла: {las.header.version}")
    print(f"Количество точек: {len(las.points)}")
    print(f"Количество атрибутов: {len(las.point_format.dimensions)}")

    print("\nДоступные атрибуты:")
    for dim in las.point_format.dimensions:
        print(f"  - {dim.name}")

    x_range = (np.min(las.x), np.max(las.x))
    y_range = (np.min(las.y), np.max(las.y))
    z_range = (np.min(las.z), np.max(las.z))
    print("\nДиапазоны координат:")
    print(f"  X: {x_range[0]:.2f} - {x_range[1]:.2f}")
    print(f"  Y: {y_range[0]:.2f} - {y_range[1]:.2f}")
    print(f"  Z: {z_range[0]:.2f} - {z_range[1]:.2f}")

    if hasattr(las, "classification"):
        classifications = las.classification
        unique_classes, counts = np.unique(classifications, return_counts=True)
        print("\nКлассы разметки:")
        for cls, count in zip(unique_classes, counts):
            percentage = count / len(classifications) * 100
            print(f"  Класс {cls}: {count} точек ({percentage:.2f}%)")
        print(f"\nВсего уникальных классов: {len(unique_classes)}")
        print(f"Диапазон классов: {unique_classes.min()} - {unique_classes.max()}")

    print("\nДополнительные атрибуты:")
    if hasattr(las, "intensity"):
        print(f"  Intensity: {np.min(las.intensity)} - {np.max(las.intensity)}")
    if hasattr(las, "return_number"):
        unique_returns, return_counts = np.unique(las.return_number, return_counts=True)
        print(f"  Return numbers: {dict(zip(unique_returns, return_counts))}")
    if hasattr(las, "number_of_returns"):
        unique_returns, return_counts = np.unique(las.number_of_returns, return_counts=True)
        print(f"  Number of returns: {dict(zip(unique_returns, return_counts))}")

    file_size_mb = os.path.getsize(las_file) / (1024**2)
    print(f"\nРазмер файла: {file_size_mb:.2f} MB")
    return las


def test_data_loading(las_file: str, sample_size: int = 10_000) -> bool:
    """Проверяет базовую загрузку и преобразование точек."""
    print(f"\n{SEPARATOR}")
    print(f"ТЕСТ ЗАГРУЗКИ ДАННЫХ (первые {sample_size} точек)")
    print(SEPARATOR)

    try:
        las = laspy.read(las_file)
    except FileNotFoundError:
        print(f"Файл {las_file} не найден. Пропускаем тест загрузки.")
        return False
    except Exception as exc:
        print(f"Ошибка при тесте загрузки: {exc}")
        return False

    total_points = len(las.points)
    if total_points == 0:
        print("Файл не содержит точек.")
        return False

    if total_points > sample_size:
        indices = np.random.choice(total_points, sample_size, replace=False)
        points = np.vstack((las.x[indices], las.y[indices], las.z[indices])).T
        if hasattr(las, "classification"):
            labels = las.classification[indices]
    else:
        points = np.vstack((las.x, las.y, las.z)).T
        if hasattr(las, "classification"):
            labels = las.classification

    print(f"Загружено точек: {len(points)}")
    print(f"Форма точек: {points.shape}")

    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_distance = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    print(f"Центроид: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
    print(f"Максимальное расстояние от центра: {max_distance:.2f}")

    if "labels" in locals():
        unique_labels, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
        print(f"Уникальные метки в сэмпле: {distribution}")

    points_tensor = torch.as_tensor(points, dtype=torch.float32).T
    print(f"Успешно преобразовано в тензор PyTorch: {points_tensor.shape}")
    return True


def suggest_training_params(las_file: str) -> Optional[Tuple[int, int, int]]:
    """Формирует рекомендации по параметрам обучения."""
    print(f"\n{SEPARATOR}")
    print("РЕКОМЕНДАЦИИ ДЛЯ ОБУЧЕНИЯ")
    print(SEPARATOR)

    try:
        las = laspy.read(las_file)
    except FileNotFoundError:
        print(f"Файл {las_file} не найден. Пропускаем рекомендации.")
        return None
    except Exception as exc:
        print(f"Ошибка при анализе параметров: {exc}")
        return None

    total_points = len(las.points)

    if hasattr(las, "classification"):
        unique_classes = np.unique(las.classification)
        num_classes = len(unique_classes)
        min_class = unique_classes.min()
        max_class = unique_classes.max()
    else:
        num_classes = 1
        min_class = max_class = 0

    print(f"Общее количество точек: {total_points:,}")
    print(f"Количество классов: {num_classes}")
    print(f"Диапазон классов: {min_class} - {max_class}")

    if total_points < 100_000:
        num_points = 2_048
        batch_size = 4
    elif total_points < 1_000_000:
        num_points = 4_096
        batch_size = 2
    else:
        num_points = 4_096
        batch_size = 1

    print("\nРекомендуемые параметры:")
    print(f"  - Размер блока (num_points): {num_points}")
    print(f"  - Размер батча: {batch_size}")
    print(f"  - Количество классов для модели: {num_classes}")

    estimated_memory = (num_points * 3 * 4 * batch_size) / (1024**2)
    print(f"  - Оценка памяти на один батч: ~{estimated_memory:.2f} MB")
    print(f"  - Использовать GPU: {'Да' if torch.cuda.is_available() else 'Не обязательно (CUDA недоступна)'}")
    return num_points, batch_size, num_classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Диагностика окружения и датасета для PointNet")
    parser.add_argument(
        "las_file",
        nargs="?",
        default="Univer2019.las",
        help="Путь к LAS-файлу (по умолчанию Univer2019.las в корне проекта)",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Пропустить анализ датасета и ограничиться диагностикой окружения",
    )
    args = parser.parse_args()

    diagnose_environment()

    if args.skip_data:
        return

    las_data = analyze_las_dataset(args.las_file)
    if las_data is None:
        return

    test_success = test_data_loading(args.las_file)
    if not test_success:
        return

    recommendations = suggest_training_params(args.las_file)
    if recommendations is None:
        return

    num_points, batch_size, num_classes = recommendations
    print(f"\n{SEPARATOR}")
    print("СВОДКА")
    print(SEPARATOR)
    print(f"Файл: {args.las_file}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    print(f"Рекомендуемый размер блока: {num_points}")
    print(f"Рекомендуемый размер батча: {batch_size}")
    print(f"Количество классов: {num_classes}")

    if not torch.cuda.is_available():
        print("\n⚠️  ВНИМАНИЕ: CUDA недоступна!")
        print("   Обучение будет происходить на CPU, что может быть медленнее.")
        print("   Рассмотрите установку PyTorch с поддержкой CUDA.")


if __name__ == "__main__":
    main()
