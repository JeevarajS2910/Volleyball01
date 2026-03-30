import os
from ultralytics import YOLO

def train():
    # Load a model
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"Weights not found at {model_path}, downloading...")
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    else:
        model = YOLO(model_path)

    # Use the local data.yaml
    data_yaml = r"c:\Volleyball01\volleyball.yolov8\data.yaml"

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=2,  # reduced from 4 to avoid WinError 1455
        project="YOLOv8/runs",
        name="volleyball_train",
        device=0,
        exist_ok=True,
        workers=0  # force scalar dataloader
    )

    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == "__main__":
    train()
