from ultralytics import YOLO

# 1. Загрузка предобученной модели YOLOv8s
model = YOLO('yolov8s.pt')  # Можно использовать 'yolov8n.pt' для более легкой версии

results = model.train(
    data='dataset.yaml',     # Путь к вашему YAML-файлу с датасетом
    epochs=100,              # Количество эпох
    batch=16,                # Размер батча (уменьшите при нехватке памяти)
    imgsz=640,               # Размер изображений для обучения
    name='qr_detector',      # Название проекта
    patience=15,             # Ранняя остановка при отсутствии прогресса
    optimizer='AdamW',       # Оптимизатор (можно использовать 'SGD')
    lr0=0.01,                # Начальный learning rate
    lrf=0.01,                # Финальный learning rate
    dropout=0.2,             # Dropout для регуляризации
    augment=True,            # Включение аугментаций (повороты, обрезка и т.д.)
    hsv_h=0.015,             # Аугментация: оттенок
    hsv_s=0.7,               # Аугментация: насыщенность
    hsv_v=0.4,               # Аугментация: яркость
    degrees=15.0,            # Максимальный угол поворота
    translate=0.1,           # Сдвиг изображения
    scale=0.5,               # Масштабирование
    fliplr=0.5,              # Вероятность горизонтального отражения
    mosaic=1.0,              # Вероятность использования мозаичной аугментации
    workers=8                # Количество потоков загрузки данных
)

metrics = model.val()

model.save('best_qr_detector.pt')

results = model.predict('test_image.jpg', save=True, conf=0.5)