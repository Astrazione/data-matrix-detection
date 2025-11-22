import os
import shutil
import random
from glob import glob
import yaml
from pathlib import Path

BASE_DIR = Path("DATASET_V2")
OUTPUT_DIR = Path("yolo_dataset")
RATIO = 0.8  # 80% на обучение, 20% на валидацию

dirs = [
    OUTPUT_DIR / "images" / "train",
    OUTPUT_DIR / "images" / "val",
    OUTPUT_DIR / "labels" / "train",
    OUTPUT_DIR / "labels" / "val"
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

image_paths = list(BASE_DIR.glob("*.jpg")) + list(BASE_DIR.glob("*.jpeg")) + list(BASE_DIR.glob("*.png"))
image_paths = [str(p) for p in image_paths]
print(f"Найдено изображений: {len(image_paths)}")


valid_pairs = []
for img_path in image_paths:
    stem = Path(img_path).stem
    label_path = BASE_DIR / f"{stem}.txt"
    
    if label_path.exists():
        # Проверка содержимого аннотации
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if lines:  # Пропускаем пустые файлы
                valid_pairs.append((img_path, str(label_path)))
    else:
        print(f"Предупреждение: Нет аннотации для {img_path}")

print(f"Валидных пар (изображение+аннотация): {len(valid_pairs)}")


def fix_annotations(label_path, output_path):
    """Преобразует класс 15 в 0 и проверяет корректность координат"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        cls_id = int(parts[0])
        # ИСПРАВЛЕНИЕ: если QR-код должен быть классом 0
        # (в вашем примере был класс 15, что нестандартно)
        fixed_cls = 0 if cls_id == 15 else cls_id
        
        # Нормализация координат в [0,1]
        coords = [float(x) for x in parts[1:5]]
        coords = [max(0.0, min(1.0, x)) for x in coords]  # Обрезка значений
        
        fixed_lines.append(f"{fixed_cls} " + " ".join(f"{x:.6f}" for x in coords))
    
    with open(output_path, 'w') as f:
        f.write("\n".join(fixed_lines))


random.seed(42)
random.shuffle(valid_pairs)
split_idx = int(len(valid_pairs) * RATIO)
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

print(f"Обучающая выборка: {len(train_pairs)} изображений")
print(f"Валидационная выборка: {len(val_pairs)} изображений")

# 7. Копирование файлов
def copy_files(pairs, img_dest, label_dest):
    for img_path, label_path in pairs:
        # Копирование изображения
        shutil.copy(img_path, img_dest / Path(img_path).name)
        
        # Исправление и копирование аннотации
        new_label_path = label_dest / Path(label_path).name
        fix_annotations(label_path, new_label_path)

copy_files(train_pairs, dirs[0], dirs[2])  # train
copy_files(val_pairs, dirs[1], dirs[3])   # val

# 8. Создание dataset.yaml
dataset_yaml = {
    'path': str(OUTPUT_DIR.resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'names': {
        0: 'qr_code'  # Единственный класс
    }
}

with open(OUTPUT_DIR / 'dataset.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)

print("✅ Датасет успешно подготовлен!")
print(f"Конфигурационный файл создан: {OUTPUT_DIR}/dataset.yaml")

# 9. Проверка качества данных (опционально)
def validate_dataset():
    """Проверяет баланс классов и корректность аннотаций"""
    from collections import Counter
    class_counter = Counter()
    
    for label_path in (OUTPUT_DIR / "labels" / "train").glob("*.txt"):
        with open(label_path, 'r') as f:
            for line in f:
                cls_id = int(line.split()[0])
                class_counter[cls_id] += 1
    
    print("\nСтатистика классов в обучающей выборке:")
    for cls_id, count in class_counter.items():
        print(f"  Класс {cls_id}: {count} bounding boxes")

validate_dataset()