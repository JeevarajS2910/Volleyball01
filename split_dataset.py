import os
import random
import shutil

# Paths
base_dir = r"c:\Volleyball01\volleyball.yolov8"
train_img_dir = os.path.join(base_dir, "train", "images")
train_lbl_dir = os.path.join(base_dir, "train", "labels")

valid_img_dir = os.path.join(base_dir, "valid", "images")
valid_lbl_dir = os.path.join(base_dir, "valid", "labels")
test_img_dir = os.path.join(base_dir, "test", "images")
test_lbl_dir = os.path.join(base_dir, "test", "labels")

# Create dirs
for d in [valid_img_dir, valid_lbl_dir, test_img_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# List all images
images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
random.shuffle(images)

# Split counts (80/10/10 of 1480 = 1184 / 148 / 148)
val_count = 148
test_count = 148

val_images = images[:val_count]
test_images = images[val_count:val_count + test_count]

def move_files(file_list, src_img, src_lbl, dst_img, dst_lbl):
    for img_name in file_list:
        lbl_name = img_name.replace('.jpg', '.txt')
        # Move image
        shutil.move(os.path.join(src_img, img_name), os.path.join(dst_img, img_name))
        # Move label
        src_lbl_path = os.path.join(src_lbl, lbl_name)
        if os.path.exists(src_lbl_path):
            shutil.move(src_lbl_path, os.path.join(dst_lbl, lbl_name))

print(f"Moving {len(val_images)} images to validation set...")
move_files(val_images, train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir)

print(f"Moving {len(test_images)} images to test set...")
move_files(test_images, train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir)

print("Split completed successfully!")
