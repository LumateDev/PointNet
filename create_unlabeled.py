import os
import laspy

def list_las_files(folder):
    if not os.path.exists(folder):
        print(f"❌ Папка {folder} не существует.")
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith('.las')]

def main():
    raw_dir = 'datasets/raw'
    out_dir = 'datasets/unlabeled'
    
    files = list_las_files(raw_dir)
    if not files:
        print("Нет .las файлов в datasets/raw/")
        return

    print("Доступные файлы:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    
    try:
        idx = int(input("Выберите номер файла: ")) - 1
        if idx < 0 or idx >= len(files):
            print("Неверный номер.")
            return
    except ValueError:
        print("Введите число.")
        return

    input_path = os.path.join(raw_dir, files[idx])
    output_path = os.path.join(out_dir, files[idx])
    os.makedirs(out_dir, exist_ok=True)

    print(f"Очистка {input_path} → {output_path}...")
    
    # Чтение исходного файла
    with laspy.open(input_path) as f:
        las = f.read()

    # Определяем совместимую версию: LAS 1.2 — минимальная поддерживаемая
    target_version = "1.2"
    point_format = las.header.point_format

    # Создаём новый LAS с поддерживаемой версией
    new_las = laspy.create(point_format=point_format, file_version=target_version)
    
    # Копируем координаты
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z

    #tets comment

    # Копируем другие поля, кроме классификации
    for dim_name in las.point_format.dimension_names:
        if 'class' in dim_name.lower():
            continue  # пропускаем любые поля с "class"
        if hasattr(las, dim_name):
            try:
                setattr(new_las, dim_name, getattr(las, dim_name))
            except Exception as e:
                print(f"⚠️  Не удалось скопировать поле '{dim_name}': {e}")

    new_las.write(output_path)
    print(f"✅ Сохранено: {output_path}")

if __name__ == '__main__':
    main()