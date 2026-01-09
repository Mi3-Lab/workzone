# Work Zone Detection System (Jetson Orin)

This repository contains the high-performance inference system for Work Zone Detection, optimized for NVIDIA Jetson Orin. It implements a multi-stage fusion pipeline combining object detection, semantic verification, and temporal smoothing.

## 🚀 Features

- **Object Detection:** YOLO11 (Optimized with TensorRT) for detecting cones, workers, and signs.
- **Semantic Fusion:** OpenCLIP integration to verify scene context (e.g., "is this actually a construction zone?").
- **Temporal Consistency:** Adaptive EMA (Exponential Moving Average) and a State Machine (`OUT` -> `APPROACHING` -> `INSIDE` -> `EXITING`) to prevent flicker.
- **Optimized for Jetson:**
  - `FP16` precision via TensorRT.
  - Custom environment handling for `libcusparseLt`.
  - Minimal overhead CLI HUD.

## 🛠️ Installation

1. **Environment Setup** (Assumes JetPack 6+):
   ```bash
   # Run the setup script to create venv and install dependencies
   ./scripts/setup.sh
   
   # Or manually:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download Models:**
   - Place your YOLO `.pt` model in `weights/`.
   - CLIP weights are downloaded automatically to `weights/clip` on first run (or run `scripts/setup_clip.py`).

## 🏃 Usage

### 1. Jetson Fusion App (Main)
The primary application for real-time inference and fusion.

```bash
# Run on all videos in data/demo
venv/bin/python scripts/jetson_app.py

# Run on a specific video with visualization
venv/bin/python scripts/jetson_app.py --input data/demo/charlotte.mp4 --show
```

### 2. Configuration
All parameters (thresholds, CLIP prompts, weights) are in `configs/jetson_config.yaml`.

```yaml
fusion:
  use_clip: true
  clip_trigger_th: 0.45  # Only run CLIP if YOLO confidence > 0.45
  weights_yolo:          # Semantic weights for state calculation
    channelization: 0.9
    workers: 0.8
```

## 🏗️ Architecture

1. **Detection (YOLO):** Runs at ~30-70 FPS (depending on resolution).
2. **Logic (Python):** Calculates a frame score based on object density and types.
3. **Verification (CLIP):** Triggers *only* when YOLO detects potential activity. Compares frame embedding against "road work" vs "normal road".
4. **State Machine:**
   - **OUT:** Score < 0.55
   - **APPROACHING:** Score > 0.55
   - **INSIDE:** Score > 0.70 for N frames.
   - **EXITING:** Score drops < 0.45 (persistent for visibility).

## ⚠️ Troubleshooting

- **`libcusparseLt` Error:** The app automatically handles `LD_LIBRARY_PATH`. If it fails, ensure `libcusparse_lt-linux-aarch64...` is present in the root.
- **Performance:** Set `imgsz: 960` in `jetson_config.yaml` for a balance of speed and accuracy.