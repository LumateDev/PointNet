import numpy as np
import open3d as o3d
import laspy
import os
import xml.etree.ElementTree as ET
import sys

def get_label_mapping_from_xml(xml_file):
    """Пытается извлечь mapping из Colors.xml (если он содержит индексы классов)."""
    label_mapping = {}
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Предположим, структура XML такая:
        # <colors>
        #   <color name="Ground" r="255" g="0" b="0" index="0"/>
        #   ...
        # </colors>
        for color_elem in root.findall('color'): # Адаптируйте под реальную структуру XML
            name = color_elem.get('name')
            index = color_elem.get('index')
            if index is not None and name is not None:
                try:
                    label_mapping[int(index)] = name
                except ValueError:
                    continue # Если index не число
        print(f"ℹ️  Найдено mapping из XML: {label_mapping}")
        return label_mapping
    except Exception as e:
        print(f"⚠️  Не удалось прочитать mapping из {xml_file}: {e}")
        return {}

def convert_ply_to_las_with_labels(ply_file_path, las_file_path, label_mapping=None):
    """
    Конвертирует .ply в .las, пытаясь извлечь разметку из .ply.
    Если разметка не найдена, заполняет её нулями.
    """
    try:
        # Загрузка .ply файла
        pcd = o3d.io.read_point_cloud(ply_file_path)

        # Попытка получить *все* атрибуты точечного облака через open3d
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None

        print(f"✅ Загружено {len(points)} точек из {ply_file_path}")

        # --- Извлечение разметки из .ply ---
        # Open3D не всегда напрямую предоставляет *все* свойства .ply (например, scalar_Label).
        # Для более точного чтения нужно использовать `plyfile`.
        # Установите: pip install plyfile
        import plyfile

        labels = None
        try:
            with open(ply_file_path, 'rb') as f:
                plydata = plyfile.PlyData.read(f)

            # Поиск потенциальных колонок с метками
            for element in plydata.elements:
                if element.name == 'vertex':
                    for prop in element.properties:
                        prop_name = prop.name
                        if 'label' in prop_name.lower() or 'class' in prop_name.lower():
                            print(f"🔍 Найден потенциальный атрибут разметки: {prop_name}")
                            labels = element[prop_name]
                            # Проверяем, является ли он числовым (например, int)
                            if not np.issubdtype(labels.dtype, np.integer):
                                print(f"⚠️  Атрибут {prop_name} не является целочисленным, пропускаем.")
                                labels = None
                            else:
                                print(f"✅ Извлечено {len(labels)} меток из атрибута '{prop_name}'.")
                                # Ограничиваем значения, чтобы они соответствовали .las (0-255 для Classification)
                                # Согласно Toronto-3D, классы 0-8 (9 классов). .las classification поле обычно uint8.
                                labels = np.clip(labels, 0, 255).astype(np.uint8)
                                # Выводим уникальные метки
                                unique_labels = np.unique(labels)
                                print(f"📊 Уникальные метки в файле: {unique_labels}")
                                if label_mapping:
                                    print("📊 Соответствие меток:")
                                    for ul in unique_labels:
                                        name = label_mapping.get(ul, f"Неизвестный (ID {ul})")
                                        print(f"   - ID {ul}: {name}")
                            break # Берём первую найденную подходящую колонку
                    if labels is not None:
                        break # Берём первую найденную подходящую колонку

        except ImportError:
            print("⚠️  Библиотека 'plyfile' не установлена. Попробуйте: pip install plyfile")
        except Exception as e:
            print(f"❌ Ошибка при чтении .ply с помощью plyfile: {e}")

        if labels is None:
            print(f"⚠️  В файле {ply_file_path} не найдена разметка классов. Используем нули.")
            labels = np.zeros(len(points), dtype=np.uint8) # Создаём массив нулей для разметки
        # --- Конец извлечения разметки ---

        # Создание .las файла (формат 3 - позволяет хранить Classification)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.x_scale = 0.01
        header.y_scale = 0.01
        header.z_scale = 0.01

        # Рассчитываем смещения, чтобы избежать переполнения
        header.x_offset = np.floor(np.min(points[:, 0])) - (np.min(points[:, 0]) % header.x_scale)
        header.y_offset = np.floor(np.min(points[:, 1])) - (np.min(points[:, 1]) % header.y_scale)
        header.z_offset = np.floor(np.min(points[:, 2])) - (np.min(points[:, 2]) % header.z_scale)

        las = laspy.LasData(header)

        # Запись координат
        las.x = (points[:, 0] - header.x_offset) / header.x_scale
        las.y = (points[:, 1] - header.y_offset) / header.y_scale
        las.z = (points[:, 2] - header.z_offset) / header.z_scale

        # Запись разметки
        las.classification = labels

        # Сохранение .las файла
        las.write(las_file_path)
        print(f"✅ Конвертация завершена. Файл сохранен: {las_file_path}")

    except Exception as e:
        print(f"❌ Ошибка при конвертации {ply_file_path}: {e}")

def main():
    # --- Изменяем пути ---
    # Путь к папке с .ply файлами (относительно текущей директории)
    input_base_folder = os.path.join(os.getcwd(), "datasets", "ply")
    # Папка для сохранения .las файлов (относительно текущей директории)
    output_folder = os.path.join(os.getcwd(), "datasets", "raw")
    # --- Конец изменения путей ---

    # Проверяем, существует ли базовая папка с .ply файлами
    if not os.path.exists(input_base_folder):
        print(f"❌ Базовая папка с .ply файлами не найдена: {input_base_folder}")
        print(f"💡 Убедитесь, что папка 'datasets/ply' существует в текущей директории.")
        sys.exit(1)

    # Создаем папку для .las файлов, если её нет
    os.makedirs(output_folder, exist_ok=True)

    # Путь к файлу с описанием классов (если есть) в текущей директории
    xml_file_path = os.path.join(os.getcwd(), "Colors.xml")
    # Альтернатива: искать в базовой папке с .ply или её подпапках
    # xml_file_path = os.path.join(input_base_folder, "Colors.xml")
    # Или еще альтернатива: искать в той же папке, где лежит .ply файл (см. ниже)
    label_mapping = get_label_mapping_from_xml(xml_file_path)

    # Рекурсивный поиск .ply файлов
    ply_files_found = False
    for root, dirs, files in os.walk(input_base_folder):
        for filename in files:
            if filename.lower().endswith('.ply'):
                ply_files_found = True
                input_ply_path = os.path.join(root, filename)
                # Имя выходного .las файла (без .ply)
                output_las_name = os.path.splitext(filename)[0] + ".las"
                # Сохраняем в output_folder, сохраняя имя файла
                output_las_path = os.path.join(output_folder, output_las_name)

                print(f"\n🔄 Обработка файла: {input_ply_path}")
                # Попробуем найти Colors.xml в той же папке, что и .ply файл
                local_xml_path = os.path.join(root, "Colors.xml")
                local_label_mapping = label_mapping
                if os.path.exists(local_xml_path):
                     local_label_mapping = get_label_mapping_from_xml(local_xml_path)

                convert_ply_to_las_with_labels(input_ply_path, output_las_path, local_label_mapping)

    if not ply_files_found:
        print(f"❌ В папке {input_base_folder} и её подпапках не найдено файлов .ply")
    else:
        print(f"\n🎉 Конвертация завершена! .las файлы сохранены в: {output_folder}")

if __name__ == "__main__":
    main()
