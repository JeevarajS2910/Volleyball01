"""
Train YOLOv7 model on the volleyball.yolov7pytorch dataset.
- Splits training data into train/val (85/15) if val doesn't exist
- Creates a corrected data.yaml with absolute paths
- Clones WongKinYiu/yolov7 repo if empty
- Downloads yolov7.pt pretrained weights
- Creates a custom yolov7.yaml with nc=6
- Starts training subprocess
"""

import os
import shutil
import random
import yaml
import subprocess
import urllib.request
from pathlib import Path

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

    random.seed(seed)
    random.shuffle(image_files)
    val_count = int(len(image_files) * val_ratio)
    val_files = image_files[:val_count]

    print(f"Moving {val_count} images to validation set ({val_ratio*100:.0f}%)")

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    moved = 0
    for img_file in val_files:
        dst_img = val_images / img_file.name
        shutil.move(str(img_file), str(dst_img))

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

    backup = base / "data_original.yaml"
    if not backup.exists() and data_yaml_path.exists():
        shutil.copy2(str(data_yaml_path), str(backup))
        print(f"Backed up original data.yaml to {backup}")

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Updated data.yaml with absolute paths")
    return str(data_yaml_path)

def prepare_yolov7(script_dir):
    """Clone YOLOv7, download weights, create custom cfg."""
    yolo_dir = os.path.join(script_dir, "YOLOv7", "yolov7")
    
    # 1. Clone repository if empty
    if not os.path.exists(yolo_dir) or not os.listdir(yolo_dir):
        print("Cloning YOLOv7 repository...")
        subprocess.run(["git", "clone", "https://github.com/WongKinYiu/yolov7.git", yolo_dir], check=True)
        # Install requirements
        req_file = os.path.join(yolo_dir, "requirements.txt")
        if os.path.exists(req_file):
            subprocess.run(["pip", "install", "-r", req_file], check=False)
            
    # 2. Download pretrained weights
    weights_path = os.path.join(script_dir, "yolov7.pt")
    if not os.path.exists(weights_path):
        print("Downloading yolov7.pt...")
        url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        urllib.request.urlretrieve(url, weights_path)
        print("Downloaded yolov7.pt")

    # 3. Create custom config with nc=6
    base_cfg_path = os.path.join(yolo_dir, "cfg", "training", "yolov7.yaml")
    custom_cfg_path = os.path.join(script_dir, "YOLOv7", "yolov7_custom.yaml")
    
    if os.path.exists(base_cfg_path):
        with open(base_cfg_path, 'r') as f:
            lines = f.readlines()
            
        with open(custom_cfg_path, 'w') as f:
            for line in lines:
                if line.startswith("nc:"):
                    f.write("nc: 6  # number of classes\n")
                else:
                    f.write(line)
        print(f"Created custom config at {custom_cfg_path}")
    else:
        print(f"Warning: Base config not found at {base_cfg_path}")
        
    return yolo_dir, weights_path, custom_cfg_path

def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "volleyball.yolov7pytorch")
    
    print("=" * 60)
    print("STEP 1: Preparing dataset splits")
    split_dataset(base_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Creating corrected data.yaml")
    data_yaml = create_data_yaml(base_dir)

    print("\n" + "=" * 60)
    print("STEP 3: Preparing YOLOv7 Repository & Weights")
    yolo_dir, weights, custom_cfg = prepare_yolov7(script_dir)

    print("\n" + "=" * 60)
    print("STEP 4: Starting YOLOv7 training")
    
    train_script = os.path.join(yolo_dir, "train.py")
    project_dir = os.path.join(script_dir, "YOLOv7", "runs")
    
    cmd = [
        "python", train_script,
        "--workers", "0",
        "--device", "0",
        "--batch-size", "2",
        "--data", data_yaml,
        "--img", "640", "640",
        "--cfg", custom_cfg,
        "--weights", weights,
        "--name", "volleyball_train",
        "--hyp", os.path.join(yolo_dir, "data", "hyp.scratch.p5.yaml"),
        "--epochs", "100",
        "--project", project_dir
    ]
    
    print("Running command:", " ".join(cmd))
    
    # Run in the yolov7 directory so relative paths in its scripts work correctly
    subprocess.run(cmd, cwd=yolo_dir, check=True)

if __name__ == "__main__":
    train()
