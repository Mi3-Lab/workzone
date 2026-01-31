#!/usr/bin/env python3
"""
High-Performance Jetson Inference Script for Work Zone Detection.

This script mirrors the functionality of the Streamlit app but is optimized
for headless execution on Jetson Orin using TensorRT.

Usage:
    python tools/run_jetson_inference.py --config configs/jetson_config.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ============================================================
# JETSON ENVIRONMENT FIX (Before other imports)
# ============================================================
CUSPARSE_LT_PATH = Path(__file__).parent.parent / "libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib"
if CUSPARSE_LT_PATH.exists():
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if str(CUSPARSE_LT_PATH) not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{CUSPARSE_LT_PATH}:{current_ld_path}"
        # Note: In Python, changing os.environ["LD_LIBRARY_PATH"] doesn't affect 
        # the current process's loader. However, some libraries (like ctypes 
        # based ones) might respect it, or we might need to re-exec.
        # For now, we'll try to just proceed, but often a re-exec is needed.

import numpy as np
import torch
import yaml
import cv2
from ultralytics import YOLO

try:
    from workzone.utils.optimize_for_jetson import export_yolo_tensorrt
except ImportError:
    print("Warning: Could not import optimize_for_jetson. Auto-export might fail.")

# ============================================================ 
# CONFIGURATION & CONSTANTS
# ============================================================ 

WORKZONE_CLASSES_10 = {
    0: "Cone",
    1: "Drum",
    2: "Barricade",
    3: "Barrier",
    4: "Vertical Panel",
    5: "Work Vehicle",
    6: "Worker",
    7: "Arrow Board",
    8: "Temporary Traffic Control Message Board",
    9: "Temporary Traffic Control Sign",
}

# ============================================================ 
# HELPER FUNCTIONS (Ported from Streamlit Utils)
# ============================================================ 

def compute_simple_workzone_score(class_ids: np.ndarray) -> float:
    """Compute simple work zone score (0-1) based on class presence."""
    if class_ids.size == 0:
        return 0.0

    # Weight classes by importance
    weights = {
        0: 1.0,  # Cone
        1: 1.0,  # Drum
        2: 1.2,  # Barricade
        3: 1.2,  # Barrier
        4: 0.8,  # Vertical Panel
        5: 1.5,  # Work Vehicle
        6: 1.8,  # Worker
        7: 1.0,  # Arrow Board
        8: 1.3,  # Message Board
        9: 0.9,  # Sign
    }

    total_weight = sum(weights.get(cid, 0.5) for cid in class_ids)
    normalized_score = min(1.0, total_weight / 10.0)
    return normalized_score

def draw_workzone_banner(frame: np.ndarray, score: float) -> np.ndarray:
    """Draw work zone score banner on frame."""
    height, width = frame.shape[:2]

    # Create semi-transparent overlay
    overlay = frame.copy()
    banner_height = 100
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Status Text
    cv2.putText(frame, "Work Zone Score", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Score with color coding
    color = (0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.4 else (0, 0, 255)
    cv2.putText(frame, f"{score:.3f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Add Hardware Info (Jetson Specific)
    cv2.putText(frame, "Jetson Orin | TensorRT", (width - 250, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_engine_path(pt_path: Path) -> Path:
    """Get the expected engine path from the pt path."""
    return pt_path.with_suffix('.engine')

def ensure_engine_exists(config: dict) -> Path:
    """
    Checks if TensorRT engine exists. If not, attempts to export it.
    Returns path to the engine file.
    """
    pt_path = Path(config['model']['path'])
    engine_path = get_engine_path(pt_path)
    
    if engine_path.exists():
        print(f"✅ Found existing TensorRT engine: {engine_path}")
        return engine_path
    
    print(f"⚠️  TensorRT engine not found at {engine_path}")
    print(f"🚀 Starting export for {pt_path} (This may take a few minutes)...")
    
    try:
        success = export_yolo_tensorrt(
            model_path=str(pt_path),
            output_dir=str(pt_path.parent),
            half=config['hardware']['half'],
            imgsz=config['model']['imgsz'],
            workspace=config['hardware']['workspace']
        )
        if success and engine_path.exists():
            return engine_path
        else:
            raise RuntimeError("Export reported success but engine file missing.")
    except Exception as e:
        print(f"❌ Auto-export failed: {e}")
        print("Falling back to PyTorch model (slower)...")
        return pt_path

# ============================================================ 
# MAIN INFERENCE LOOP
# ============================================================ 

def process_video(video_path: Path, model: YOLO, output_dir: Path, config: dict):
    """Process a single video."""
    print(f"\nProcessing: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limits
    max_frames = config['video']['max_frames']
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # Output setup
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"annotated_{video_path.name}"
    # Use mp4v for compatibility, or avc1/h264 if available
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Stats
    frame_idx = 0
    start_time = time.time()
    inference_times = []
    
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    
    try:
        while cap.isOpened():
            if max_frames > 0 and frame_idx >= max_frames:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            t0 = time.time()
            
            # Inference
            # device=0 is important. verbose=False reduces console spam.
            results = model.predict(
                frame, 
                conf=config['model']['conf'],
                iou=config['model']['iou'],
                device=config['hardware']['device'],
                verbose=False,
                imgsz=config['model']['imgsz'] # Explicitly set img size for TRT consistency
            )
            result = results[0]
            
            # Track inference time
            inference_times.append(time.time() - t0)
            
            # Visualization
            # plotting is relatively expensive, but necessary for visual output
            annotated_frame = result.plot() 
            
            # Custom Scoring & Banner
            if result.boxes is not None:
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                score = compute_simple_workzone_score(cls_ids)
            else:
                score = 0.0
                
            annotated_frame = draw_workzone_banner(annotated_frame, score)
            
            # Write to output
            out.write(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                avg_fps = 1.0 / np.mean(inference_times[-30:])
                sys.stdout.write(f"\rProgress: {frame_idx}/{total_frames} | FPS: {avg_fps:.1f} | Score: {score:.2f}")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        cap.release()
        out.release()
        
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(f"\n\nDone! Average FPS: {avg_fps:.1f}")
    print(f"Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Jetson Orin Work Zone Inference")
    parser.add_argument("--config", default="configs/jetson_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Model Setup
    model_file = ensure_engine_exists(config)
    print(f"Loading model: {model_file}")
    model = YOLO(str(model_file))
    
    # 2. Input Setup
    input_path = Path(config['video']['input'])
    output_dir = Path(config['video']['output_dir'])
    
    videos = []
    if input_path.is_file():
        videos.append(input_path)
    elif input_path.is_dir():
        videos.extend(sorted(input_path.glob("*.mp4")))
        videos.extend(sorted(input_path.glob("*.avi")))
        videos.extend(sorted(input_path.glob("*.mov")))
    
    if not videos:
        print(f"❌ No videos found at {input_path}")
        sys.exit(1)
        
    print(f"Found {len(videos)} videos to process.")
    
    # 3. Processing Loop
    for vid in videos:
        process_video(vid, model, output_dir, config)

if __name__ == "__main__":
    main()
