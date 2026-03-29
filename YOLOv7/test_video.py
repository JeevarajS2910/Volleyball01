import cv2
import math
import os
import sys
import torch
import numpy as np

# ────────────────────────── CONFIGURATION ──────────────────────────
CURRENT_FILE_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(CURRENT_FILE_PATH)
BASE_DIR = os.path.dirname(SCRIPT_DIR)
YOLOV7_REPO = os.path.join(SCRIPT_DIR, "yolov7")

# Add YOLOv7 repo to sys.path
if YOLOV7_REPO not in sys.path:
    sys.path.append(YOLOV7_REPO)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Path to the trained YOLOv7 model weights
MODEL_PATH = os.path.join(SCRIPT_DIR, "runs", "volleyball_train", "weights", "best.pt")

# Path to the input video
VIDEO_PATH = os.path.join(BASE_DIR, "Input vido.mp4")

# Output video path
OUTPUT_VIDEO_PATH = os.path.join(SCRIPT_DIR, "runs", "inference_output.mp4")

# Heuristic thresholds for action recognition
ACTION_DISTANCE_THRESHOLD = 150  
# ───────────────────────────────────────────────────────────────────

def calculate_distance(box1, box2):
    c1_x = (box1[0] + box1[2]) / 2
    c1_y = (box1[1] + box1[3]) / 2
    c2_x = (box2[0] + box2[2]) / 2
    c2_y = (box2[1] + box2[3]) / 2
    return math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

def recognize_action(player_box, ball_box, previous_ball_y):
    if ball_box is None:
        return "standing"
    
    dist = calculate_distance(player_box, ball_box)
    
    if dist < ACTION_DISTANCE_THRESHOLD:
        ball_center_y = (ball_box[1] + ball_box[3]) / 2
        if previous_ball_y is not None and ball_center_y < previous_ball_y:
            return "shooting / hitting"
        else:
            return "receiving"
            
    return "standing"

def run_inference():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found at: {VIDEO_PATH}")
        sys.exit(1)

    print(f"✅ Loading YOLOv7 model from {MODEL_PATH}")
    model = attempt_load(MODEL_PATH, map_location=device)
    model.eval()

    print(f"✅ Opening video {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Error opening video file.")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"🎬 Processing video. Output will be saved to {OUTPUT_VIDEO_PATH}")
    
    frame_count = 0
    previous_ball_y = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        img0 = frame.copy()
        # Padded resize
        img = letterbox(img0, 640, stride=32)[0]
        # Convert HWC to CHW, BGR to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        frame_count += 1
        
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                ball_box = None
                player_boxes = []
                
                for *xyxy, conf, cls in det:
                    cls_id = int(cls)
                    if cls_id == 0:
                        ball_box = [float(x) for x in xyxy]
                    elif cls_id in [1, 2]:
                        player_boxes.append(([float(x) for x in xyxy], cls_id))
                
                for (p_box, cls_id) in player_boxes:
                    color = (255, 0, 0) if cls_id == 1 else (0, 0, 255) 
                    action = recognize_action(p_box, ball_box, previous_ball_y)
                    x1, y1, x2, y2 = map(int, p_box)
                    cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                    label = f"Player {cls_id}: {action}"
                    cv2.putText(img0, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                if ball_box is not None:
                    x1, y1, x2, y2 = map(int, ball_box)
                    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img0, "Volleyball", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    previous_ball_y = (y1 + y2) / 2
                
        out.write(img0)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("✅ Video processing complete.")
    print(f"📂 Output saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    run_inference()
