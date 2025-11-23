from ultralytics import YOLO # type: ignore
import torch

DATASET_PATH = 'source/yolo_dataset/dataset.yaml'

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обучение на устройстве: {device}")

    model = YOLO('models/yolo12s.pt')
    results = model.train(
        data=DATASET_PATH,
        epochs=100,
        batch=16,
        name='data_matrix_detector-12',
        project='models',
        augment=True,
        degrees=25.0,
        translate=0.05,
        scale=0.1,
        amp=False
    )

if __name__ == '__main__':
    train()
