import torch
import laspy
import numpy as np
from tqdm import tqdm
from model import PointNet2SemSeg
import traceback
import sys

def predict_las_file_with_comparison(input_las_unlabeled, reference_las, model_path, output_las, num_points=4096, batch_size=8):
    """
    Применение обученной модели к неразмеченному облаку точек и сравнение с эталоном
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Устройство: {device}")
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели из {model_path}...")
    try:
        model = PointNet2SemSeg(num_classes=8).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        print(traceback.format_exc())
        return
    
    # Загрузка неразмеченного LAS файла
    print(f"\n📂 Загрузка неразмеченного файла {input_las_unlabeled}...")
    try:
        las_unlabeled = laspy.read(input_las_unlabeled)
        points = np.vstack([las_unlabeled.x, las_unlabeled.y, las_unlabeled.z]).T.astype(np.float32)
        num_total_points = len(points)
        print(f"✅ Загружено {num_total_points} точек")
    except Exception as e:
        print(f"❌ Ошибка загрузки неразмеченного файла: {e}")
        print(traceback.format_exc())
        return
    
    # Загрузка эталонного LAS файла для сравнения
    print(f"\n🔍 Загрузка эталонного файла {reference_las} для сравнения...")
    try:
        las_reference = laspy.read(reference_las)
        reference_classification = las_reference.classification
        print(f"✅ Эталонная классификация загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки эталонного файла: {e}")
        print(traceback.format_exc())
        return
    
    # Проверка, что файлы содержат одинаковое количество точек
    if len(reference_classification) != num_total_points:
        print(f"⚠️  Внимание: количество точек не совпадает!")
        print(f"   Неразмеченный файл: {num_total_points}")
        print(f"   Эталонный файл: {len(reference_classification)}")
        print("   Используем минимальное количество точек")
        min_points = min(num_total_points, len(reference_classification))
        points = points[:min_points]
        reference_classification = reference_classification[:min_points]
        num_total_points = min_points
    else:
        print(f"✅ Количество точек совпадает: {num_total_points}")
    
    # Подготовка массива для предсказаний
    predictions = np.zeros(num_total_points, dtype=np.uint8)
    
    # Нормализация координат
    coord_min = np.min(points, axis=0)
    coord_max = np.max(points, axis=0)
    
    # Разбиение на блоки
    print("\n📦 Разбиение на блоки...")
    block_size = 50.0
    stride = 50.0  # Без перекрытия для предсказания
    
    blocks = []
    block_indices = []
    
    x_min, y_min, _ = coord_min
    x_max, y_max, _ = coord_max
    
    x_blocks = int(np.ceil((x_max - x_min) / stride))
    y_blocks = int(np.ceil((y_max - y_min) / stride))
    
    for i in range(x_blocks):
        for j in range(y_blocks):
            x_start = x_min + i * stride
            y_start = y_min + j * stride
            x_end = x_start + block_size
            y_end = y_start + block_size
            
            mask = (points[:, 0] >= x_start) & (points[:, 0] <= x_end) & \
                   (points[:, 1] >= y_start) & (points[:, 1] <= y_end)
            
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                blocks.append(points[indices])
                block_indices.append(indices)
    
    print(f"✅ Создано {len(blocks)} блоков")
    
    # Предсказание для каждого блока
    print("\n🔮 Предсказание классов...")
    
    with torch.no_grad():
        for block_idx, (block_pts, indices) in enumerate(tqdm(zip(blocks, block_indices), total=len(blocks))):
            # Фильтруем индексы, чтобы не выйти за пределы
            valid_indices = [idx for idx in indices if idx < num_total_points]
            if len(valid_indices) == 0:
                continue
                
            indices = np.array(valid_indices)
            
            # Обработка блока частями по num_points
            block_pts_filtered = points[indices]
            num_block_points = len(block_pts_filtered)
            
            if num_block_points == 0:
                continue
            
            # Центрирование блока
            centroid = np.mean(block_pts_filtered, axis=0)
            block_pts_centered = block_pts_filtered - centroid
            
            # Нормализация по максимальному расстоянию
            max_dist = np.max(np.sqrt(np.sum(block_pts_centered ** 2, axis=1)))
            if max_dist > 0:
                block_pts_centered = block_pts_centered / max_dist
            
            # Если точек больше num_points, обрабатываем частями
            if num_block_points > num_points:
                block_predictions = np.zeros(num_block_points, dtype=np.int64)
                
                # Разбиваем на части
                num_batches = int(np.ceil(num_block_points / num_points))
                
                for b in range(num_batches):
                    start_idx = b * num_points
                    end_idx = min((b + 1) * num_points, num_block_points)
                    
                    # Если последняя часть меньше num_points, дополняем
                    if end_idx - start_idx < num_points:
                        # Дополняем повторением случайных точек
                        current_pts = block_pts_centered[start_idx:end_idx]
                        num_missing = num_points - len(current_pts)
                        random_indices = np.random.choice(len(current_pts), num_missing, replace=True)
                        padded_pts = np.vstack([current_pts, current_pts[random_indices]])
                        
                        # Предсказание
                        pts_tensor = torch.from_numpy(padded_pts).float().unsqueeze(0).transpose(1, 2).to(device)
                        pred = model(pts_tensor)
                        pred_labels = pred.argmax(dim=2).cpu().numpy()[0]
                        
                        # Берем только реальные точки
                        block_predictions[start_idx:end_idx] = pred_labels[:len(current_pts)]
                    else:
                        pts_tensor = torch.from_numpy(block_pts_centered[start_idx:end_idx]).float().unsqueeze(0).transpose(1, 2).to(device)
                        pred = model(pts_tensor)
                        pred_labels = pred.argmax(dim=2).cpu().numpy()[0]
                        block_predictions[start_idx:end_idx] = pred_labels
            else:
                # Если точек меньше num_points, дополняем
                num_missing = num_points - num_block_points
                random_indices = np.random.choice(num_block_points, num_missing, replace=True)
                padded_pts = np.vstack([block_pts_centered, block_pts_centered[random_indices]])
                
                # Предсказание
                pts_tensor = torch.from_numpy(padded_pts).float().unsqueeze(0).transpose(1, 2).to(device)
                pred = model(pts_tensor)
                pred_labels = pred.argmax(dim=2).cpu().numpy()[0]
                
                # Берем только реальные точки
                block_predictions = pred_labels[:num_block_points]
            
            # Сохраняем предсказания (переводим 0-7 обратно в 1-8)
            predictions[indices] = block_predictions + 1
    
    # Вычисление точности по сравнению с эталоном
    print(f"\n📊 Вычисление точности по сравнению с эталоном...")
    
    # Приведение эталонных меток к numpy массиву и к диапазону 1-8
    ref_labels = np.array(reference_classification).astype(np.int64)
    
    # Фильтрация точек, где эталонная классификация > 0 (имеет значение)
    valid_mask = ref_labels > 0
    valid_predictions = predictions[valid_mask]
    valid_reference = ref_labels[valid_mask]
    
    if len(valid_reference) > 0:
        # Вычисление общей точности
        total_accuracy = np.mean(valid_predictions == valid_reference) * 100
        print(f"🎯 Общая точность: {total_accuracy:.2f}%")
        
        # Вычисление точности по классам
        print(f"\n📊 Точность по классам:")
        unique_classes = np.unique(np.concatenate([valid_reference, valid_predictions]))
        unique_classes = unique_classes[unique_classes > 0]  # Исключаем нулевые метки
        
        for class_id in sorted(unique_classes):
            class_mask = valid_reference == class_id
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(valid_predictions[class_mask] == valid_reference[class_mask]) * 100
                ref_count = np.sum(class_mask)
                pred_count = np.sum(valid_predictions[class_mask] == class_id)
                print(f"   Класс {class_id}: {class_accuracy:.2f}% (эталон: {ref_count}, предсказано: {pred_count})")
    else:
        print("❌ Нет валидных эталонных меток для вычисления точности")
    
    # Сохранение результата
    print(f"\n💾 Сохранение результата в {output_las}...")
    
    # Создаем новый LAS файл с предсказаниями
    output_las_data = laspy.LasData(las_unlabeled.header)
    output_las_data.x = las_unlabeled.x
    output_las_data.y = las_unlabeled.y
    output_las_data.z = las_unlabeled.z
    
    # Копируем другие атрибуты если они есть
    if hasattr(las_unlabeled, 'intensity'):
        output_las_data.intensity = las_unlabeled.intensity
    if hasattr(las_unlabeled, 'return_number'):
        output_las_data.return_number = las_unlabeled.return_number
    if hasattr(las_unlabeled, 'number_of_returns'):
        output_las_data.number_of_returns = las_unlabeled.number_of_returns
    
    # Добавляем предсказанную классификацию
    output_las_data.classification = predictions
    
    # Сохраняем
    output_las_data.write(output_las)
    
    print("✅ Готово!")
    
    # Статистика предсказаний
    print("\n📊 Статистика предсказанных классов:")
    unique, counts = np.unique(predictions, return_counts=True)
    for class_id, count in zip(unique, counts):
        percentage = 100.0 * count / num_total_points
        print(f"   Класс {class_id}: {count} точек ({percentage:.2f}%)")

def main():
    # Параметры
    # INPUT_LAS_UNLABELED = 'datasets/unlabeled/Univer2019.las'  # Неразмеченный файл для предсказания
    # REFERENCE_LAS = 'datasets/raw/Univer2019.las'              # Эталонный файл для сравнения
    INPUT_LAS_UNLABELED = 'datasets/unlabeled/NEONDSSampleLiDARPointCloud.las'  # Неразмеченный файл для предсказания
    REFERENCE_LAS = 'datasets/raw/Univer2019.las'              # Эталонный файл для сравнения
    OUTPUT_LAS = 'predicted_with_comparison.las'               # Выходной файл с предсказаниями
    MODEL_PATH = 'checkpoints/best_model.pth'                  # Путь к обученной модели
    
    try:
        predict_las_file_with_comparison(INPUT_LAS_UNLABELED, REFERENCE_LAS, MODEL_PATH, OUTPUT_LAS)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        print(traceback.format_exc())
    finally:
        input("\nНажмите Enter для выхода...")  # Добавляем паузу в конце

if __name__ == '__main__':
    main()