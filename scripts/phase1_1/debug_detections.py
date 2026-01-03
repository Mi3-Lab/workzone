#!/usr/bin/env python3
"""Debug script to see what YOLO is detecting"""

from ultralytics import YOLO
import cv2
from pathlib import Path

# Load model and video
model = YOLO("weights/best.pt")
video_path = "data/demo/boston_workzone_short.mp4"

cap = cv2.VideoCapture(video_path)
frame_count = 0

print(f"🎬 Analyzing detections in: {video_path}\n")
print(f"📝 Model classes ({len(model.names)}):")
for class_id, class_name in model.names.items():
    print(f"   {class_id:2d}: {class_name}")
print()

# Process first 10 frames
detection_summary = {}
for frame_idx in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, verbose=False)[0]
    
    if len(results.boxes) > 0:
        print(f"Frame {frame_idx}: {len(results.boxes)} detections")
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            
            if cls_name not in detection_summary:
                detection_summary[cls_name] = []
            detection_summary[cls_name].append(conf)
            
            print(f"  • {cls_name:30s} conf={conf:.3f}")
    else:
        print(f"Frame {frame_idx}: NO DETECTIONS")
    print()

cap.release()

print("\n" + "="*80)
print("DETECTION SUMMARY (first 10 frames)")
print("="*80)
for cls_name, confs in sorted(detection_summary.items()):
    avg_conf = sum(confs) / len(confs)
    print(f"{cls_name:30s}: {len(confs):3d} detections, avg conf={avg_conf:.3f}")
