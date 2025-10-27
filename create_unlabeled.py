import argparse
import os
from typing import Optional

import laspy


def create_unlabeled_copy(source_path: str, destination_path: str) -> Optional[str]:
    """Создает копию LAS-файла с координатами без разметки."""
    if not os.path.exists(source_path):
        print(f"❌ Исходный файл не найден: {source_path}")
        return None

    try:
        las = laspy.read(source_path)
    except Exception as exc:
        print(f"❌ Не удалось прочитать {source_path}: {exc}")
        return None

    header = laspy.LasHeader(point_format=las.header.point_format.id, version="1.2")
    unlabeled_las = laspy.LasData(header)
    unlabeled_las.x = las.x
    unlabeled_las.y = las.y
    unlabeled_las.z = las.z

    try:
        unlabeled_las.write(destination_path)
    except Exception as exc:
        print(f"❌ Ошибка записи {destination_path}: {exc}")
        return None

    print(f"✅ Создан файл {destination_path} (версия 1.2, только координаты)")
    return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Создает версию LAS-файла без разметки")
    parser.add_argument(
        "source",
        nargs="?",
        default="Univer2019.las",
        help="Исходный LAS-файл с разметкой",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        default="unlabeled.las",
        help="Итоговый LAS-файл без разметки",
    )
    args = parser.parse_args()

    created_file = create_unlabeled_copy(args.source, args.destination)
    if created_file is not None:
        print("Готово! Можно использовать файл для инференса или псевдоразметки.")


if __name__ == "__main__":
    main()
