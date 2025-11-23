from ultralytics import YOLO # type: ignore
import torch

DATASET_PATH = 'source/finetuning_dataset/data.yaml'

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Дообучение (!!!) на устройстве: {device}")

    model = YOLO('models/data_matrix_detector-12/weights/best.pt')
    results = model.train(
        data=DATASET_PATH,
        epochs=50,
        batch=16,
        name='data_matrix_detector-12-finetuned',
        project='models',
        lr0=0.001,
        lrf=0.0002,
        amp=False
    )

if __name__ == '__main__':
    train()
