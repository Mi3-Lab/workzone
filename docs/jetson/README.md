# Jetson Orin Workzone Fusion System

A high-performance, real-time work zone detection system optimized for NVIDIA Jetson Orin. This application combines **YOLOv12** object detection with **OpenCLIP** semantic verification and temporal state tracking.

## 🌟 Key Features

- **Semantic Fusion:** Combines object detection confidence with whole-frame semantic understanding (CLIP) to validate work zones.
- **State Machine:** Robust transition logic (`OUT` → `APPROACHING` → `INSIDE` → `EXITING`) prevents flickering detections.
- **Hardware Acceleration:** 
  - YOLOv12 export to **TensorRT (FP16)** for maximum throughput on Orin RT Cores.
  - Asynchronous video writing.
- **Zero-Overlap HUD:** Interface rendered in a dedicated top bar, preserving full video visibility.

## 🚀 Quick Start

### 1. Installation
Ensure you are on JetPack 6.0+ (Ubuntu 22.04).

```bash
# Setup environment (creates venv and installs dependencies)
./scripts/setup.sh
```

### 2. Run Inference
To process a video with visualization:

```bash
# Process a specific video
venv/bin/python scripts/jetson_app.py --input data/demo/boston.mp4 --show

# Process all videos in the configured folder
venv/bin/python scripts/jetson_app.py
```

### 3. Configuration (`configs/jetson_config.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `model.path` | `weights/yolo12s...pt` | Base YOLO model (auto-converts to .engine) |
| `model.imgsz` | `960` | Input resolution (lower = faster) |
| `fusion.use_clip` | `true` | Enable semantic verification |
| `video.stride` | `1` | Frame skip factor (increase for speed) |

## 🏗️ Architecture

### Pipeline Steps
1.  **Input:** Video frame capture.
2.  **YOLO Inference:** Detects cones, signs, workers.
3.  **Logic Score:** Calculates weighted score based on object counts.
4.  **EMA:** Smooths score over time.
5.  **CLIP Trigger:** If potential work zone detected, runs CLIP embedding to confirm.
6.  **Fusion:** Weighted average of YOLO + CLIP scores.
7.  **State Machine:** Updates global state (e.g., enters "INSIDE" only if score > 0.7 for 25 frames).
8.  **Render:** Draws HUD on padded frame and saves video.

### Optimization
- **TensorRT:** The script automatically exports `.pt` models to `.engine` on first run.
- **Memory:** CLIP model is cached locally in `weights/clip`.
- **Environment:** Automatically handles `LD_LIBRARY_PATH` for `libcusparseLt`.

## 📊 Outputs
Results are saved to `results/jetson/` as encoded MP4 files with the HUD overlay.
