"""
Train YOLOv26 model on the volleyball.yolo26 dataset.
- Splits training data into train/val (85/15) if val doesn't exist
- Creates a corrected data.yaml with absolute paths
- Trains with yolo26n.pt pretrained weights
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO


def split_dataset(base_dir: str, val_ratio: float = 0.15, seed: int = 42):
    """Split train data into train/val if val doesn't exist."""
    base = Path(base_dir)
    train_images = base / "train" / "images"
    train_labels = base / "train" / "labels"
    val_images = base / "valid" / "images"
    val_labels = base / "valid" / "labels"

    if val_images.exists() and len(list(val_images.glob("*"))) > 0:
        print(f"Validation set already exists with {len(list(val_images.glob('*')))} images. Skipping split.")
        return

    # Get all image files
    image_files = sorted([
        f for f in train_images.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    ])

    if not image_files:
        raise FileNotFoundError(f"No images found in {train_images}")

    print(f"Found {len(image_files)} images in training set")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(image_files)
    val_count = int(len(image_files) * val_ratio)
    val_files = image_files[:val_count]

    print(f"Moving {val_count} images to validation set ({val_ratio*100:.0f}%)")

    # Create val directories
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    moved = 0
    for img_file in val_files:
        # Move image
        dst_img = val_images / img_file.name
        shutil.move(str(img_file), str(dst_img))

        # Move corresponding label
        label_file = train_labels / (img_file.stem + ".txt")
        if label_file.exists():
            dst_label = val_labels / label_file.name
            shutil.move(str(label_file), str(dst_label))
        moved += 1

    remaining_train = len(list(train_images.glob("*")))
    print(f"Split complete: {remaining_train} train / {moved} val images")


def create_data_yaml(base_dir: str):
    """Create a corrected data.yaml with absolute paths."""
    base = Path(base_dir)
    data_yaml_path = base / "data.yaml"

    data_config = {
        'train': str(base / "train" / "images").replace("\\", "/"),
        'val': str(base / "valid" / "images").replace("\\", "/"),
        'nc': 6,
        'names': ['1', '2', 'ball', 'player', 'player 1', 'referee']
    }

    # Backup original
    backup = base / "data_original.yaml"
    if not backup.exists() and data_yaml_path.exists():
        shutil.copy2(str(data_yaml_path), str(backup))
        print(f"Backed up original data.yaml to {backup}")

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Updated data.yaml with absolute paths:")
    print(f"  train: {data_config['train']}")
    print(f"  val:   {data_config['val']}")
    print(f"  classes ({data_config['nc']}): {data_config['names']}")

    return str(data_yaml_path)


def train():
    """Train the YOLOv26 model."""
    # Dynamically resolve paths to support both local Windows and Lightning AI (Linux)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "volleyball.yolo26")
    weights = os.path.join(script_dir, "yolo26n.pt")
    project_dir = os.path.join(script_dir, "YOLOv26", "runs")

    # Step 1: Split dataset
    print("=" * 60)
    print("STEP 1: Preparing dataset splits")
    print("=" * 60)
    split_dataset(base_dir)

    # Step 2: Fix data.yaml
    print("\n" + "=" * 60)
    print("STEP 2: Creating corrected data.yaml")
    print("=" * 60)
    data_yaml = create_data_yaml(base_dir)

    # Step 3: Load model and train
    print("\n" + "=" * 60)
    print("STEP 3: Starting YOLOv26 training")
    print("=" * 60)

    if not os.path.exists(weights):
        print(f"ERROR: Pretrained weights not found at {weights}")
        return

    model = YOLO(weights)
    print(f"Loaded model: {weights}")

    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=2,
        project=project_dir,
        name="volleyball_train",
        device=0,
        exist_ok=True,
        workers=0,          # Avoid Windows multiprocessing issues
        patience=50,        # Early stopping patience
        save_period=10,     # Save checkpoint every 10 epochs
        pretrained=True,
        amp=True,           # Mixed precision for faster training
        mixup=0.1,          # Augmentation
        mosaic=1.0,
        close_mosaic=10,    # Disable mosaic last 10 epochs
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Results saved to: {results.save_dir}")
    print("=" * 60)

    # Print final metrics
    print("\nBest weights: ", os.path.join(results.save_dir, "weights", "best.pt"))


if __name__ == "__main__":
    train()
