import argparse
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import time

# Target Class to Isolate
TARGET_CLASS = "Temporary Traffic Control Sign"

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO performance specifically on Workzone Signs")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="weights/yolo12s_hardneg_1280.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (lower to see weak detections)")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped images of detected signs")
    args = parser.parse_args()

    # Setup Paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input {input_path} not found.")
        return

    output_dir = Path("results/sign_evaluation") / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_crops:
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(exist_ok=True)

    # Load Model
    print(f"Loading model: {args.model}...")
    model = YOLO(args.model)
    
    # Verify Class Name
    class_id = None
    for idx, name in model.names.items():
        if name == TARGET_CLASS:
            class_id = idx
            break
    
    if class_id is None:
        print(f"⚠️ Warning: Model does not have exact class '{TARGET_CLASS}'.")
        print(f"Available classes: {model.names}")
        print("Using string matching fallback...")
    else:
        print(f"✅ Isolation Target: '{TARGET_CLASS}' (ID: {class_id})")

    # Open Video
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output Video
    out_vid_path = output_dir / f"{input_path.stem}_signs_only.mp4"
    writer = cv2.VideoWriter(str(out_vid_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing {input_path.name}...")
    
    stats = {
        "total_frames": 0,
        "frames_with_sign": 0,
        "total_signs": 0,
        "confidences": []
    }

    pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        stats["total_frames"] += 1
        
        # Inference
        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        
        has_sign = False
        annotated_frame = frame.copy()
        
        # Dim the background to highlight signs
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                conf = float(box.conf)
                
                # Check if it matches our target
                # STRICT MATCH: Only allow exact Class ID (15) or Exact Name
                # This excludes 'Temporary Traffic Control Sign: left arrow' etc.
                is_target = False
                if class_id is not None:
                    if cls_id == class_id:
                        is_target = True
                elif cls_name == TARGET_CLASS: # Fallback strict string match
                    is_target = True
                
                if is_target:
                    has_sign = True
                    stats["total_signs"] += 1
                    stats["confidences"].append(conf)
                    
                    # Draw Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 165, 255) # Orange
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"SIGN {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Save Crop
                    if args.save_crops:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_name = f"frame_{stats['total_frames']}_conf_{conf:.2f}.jpg"
                            cv2.imwrite(str(crops_dir / crop_name), crop)

        if has_sign:
            stats["frames_with_sign"] += 1
            # Add "DETECTED" banner
            cv2.rectangle(annotated_frame, (0, 0), (width, 60), (0, 165, 255), -1)
            cv2.putText(annotated_frame, f"WORKZONE SIGN DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        writer.write(annotated_frame)
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()

    # Summary
    print("\n" + "="*40)
    print("📊 SIGN DETECTION REPORT")
    print("="*40)
    print(f"Video: {input_path.name}")
    print(f"Frames with Signs: {stats['frames_with_sign']} / {stats['total_frames']} ({stats['frames_with_sign']/stats['total_frames']*100:.1f}%)")
    print(f"Total Signs Detected: {stats['total_signs']}")
    
    if stats['confidences']:
        avg_conf = sum(stats['confidences']) / len(stats['confidences'])
        max_conf = max(stats['confidences'])
        min_conf = min(stats['confidences'])
        print(f"Confidence - Avg: {avg_conf:.3f} | Max: {max_conf:.3f} | Min: {min_conf:.3f}")
    else:
        print("No signs detected.")
        
    print(f"\nResults saved to: {output_dir}")
    if args.save_crops:
        print(f"Crops saved to: {crops_dir}")
    print("="*40)

if __name__ == "__main__":
    main()
