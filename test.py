import cv2
from ultralytics import YOLO # type: ignore
from pathlib import Path

BASE_DIR = 'source'
MODEL_PATH = f'models/best-finetuned.pt'
IMAGE_PATH = f'{BASE_DIR}/test_images/test5.jpg'
OUTPUT_DIR = f'{BASE_DIR}/prediction_results'

Path(OUTPUT_DIR).mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)

results = model(
    IMAGE_PATH,
    conf=0.3,        # порог уверенности (можно понизить для QR-кодов)
    iou=0.45,         # порог NMS IoU
    save=True,        # сохранить изображение с bounding boxes
    save_txt=False,   # не сохранять txt-файлы (можно включить при желании)
    project=OUTPUT_DIR,
    name='output',    # папка внутри OUTPUT_DIR
    exist_ok=True     # перезаписывать, если уже существует
)

for result in results:
    saved_image_path = result.save_dir + '/' + Path(result.path).name
    print(f"Результат сохранён в: {saved_image_path}")

    img_with_boxes = result.plot()  # возвращает numpy array (BGR)
    cv2.imshow("Prediction", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()