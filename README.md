# Work Zone Detection System (Jetson Orin)

This repository contains the high-performance inference system for Work Zone Detection, optimized for NVIDIA Jetson Orin. It implements a multi-stage fusion pipeline combining object detection (YOLO), semantic verification (CLIP), and robust temporal state tracking.

## 🚀 Features

- **Object Detection:** YOLO12 (Optimized with TensorRT FP16) for detecting cones, workers, and signs.
- **Semantic Fusion:** OpenCLIP integration to verify scene context (e.g., "is this actually a construction zone?") with reduced stride for performance.
- **Robust State Machine:** Phase 2.1 logic with Adaptive EMA and Hysteresis (`OUT` -> `APPROACHING` -> `INSIDE` -> `EXITING`) to prevent false alerts and flickering.
- **Production Ready:**
  - **Threaded Video I/O:** Asynchronous writing for maximum FPS.
  - **Fast Resize:** Optimized pre-processing for CLIP.
  - **GUI Launcher:** Complete control panel for easy operation.

## 🛠️ Installation

1. **Environment Setup** (Assumes JetPack 6+):
   ```bash
   # Run the setup script to create venv and install dependencies
   ./scripts/setup.sh
   ```

2. **Download Models:**
   - Place your YOLO `.pt` model in `weights/` (default: `yolo12s_hardneg_1280.pt`).
   - CLIP weights are downloaded automatically to `weights/clip` on first run.

## 🏃 Usage

### 1. Jetson Launcher (GUI) - Recommended
The easiest way to configure and run the system.

**Production Mode (Stable):**
```bash
make workzone
```
Runs the verified Phase 2.1 pipeline (YOLO + CLIP + Adaptive EMA). Best for deployment.

**SOTA Experimental Mode (VLM Copilot):**
```bash
make workzone2
```
Runs the advanced hybrid engine with **Qwen2.5-VL Copilot**.
- **Async Threading:** VLM reasoning runs in parallel without blocking video.
- **Hybrid Fusion:** VLM semantic score blends with YOLO detections.
- **Enhanced HUD:** Visualizes "Copilot Check" status.

**Features:**
- **Stateless:** Resets to default settings every time you open it.
- **Import/Export:** Save your tuning presets to JSON files.
- **Hot Reload:** Adjust thresholds and weights while the video is running to see immediate effects.
- **Run/Stop:** Control the inference process directly.

### 2. CLI Usage (Advanced)
For headless operation or scripting.

```bash
# Run on all videos in data/demo
venv/bin/python scripts/jetson_app.py

# Run on a specific video with visualization
venv/bin/python scripts/jetson_app.py --input data/demo/charlotte.mp4 --show
```

## 🏗️ Architecture (Phase 2.1)

1. **Detection (YOLO):** Runs at ~30-70 FPS using TensorRT FP16.
2. **Scoring Logic:** Linear combination of semantic groups (Channelization, Workers, Vehicles) with fixed normalization.
3. **Adaptive Smoothing:** EMA Alpha adapts based on "Evidence" (Score + Object Density).
4. **Verification (CLIP):** Triggers when YOLO score > 0.20 to verify context.
5. **State Machine:**
   - **OUT:** Score < 0.25
   - **APPROACHING:** Score >= 0.25 (With hysteresis: drops to OUT only if < 0.20)
   - **INSIDE:** Score >= 0.42 for 6 frames.
   - **EXITING:** Score < 0.30 (Persistent for 20 frames).

## ⚙️ Configuration

The system uses `configs/jetson_config.yaml`.
- **Defaults:** The launcher loads from `configs/jetson_config_defaults.yaml`.
- **Runtime:** The app reads from `configs/jetson_config.yaml`.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enter_th` | `0.42` | Threshold to enter INSIDE state |
| `approach_th` | `0.25` | Threshold to trigger APPROACHING |
| `clip_trigger_th` | `0.20` | When to run CLIP verification |
| `ema_alpha` | `0.10` | Smoothing factor (lower = smoother) |

## ⚠️ Troubleshooting

- **`libcusparseLt` Error:** The app automatically handles `LD_LIBRARY_PATH`. If it fails, ensure the library folder exists in the root.
- **Performance:** Ensure you run `sudo jetson_clocks` before inference for maximum performance.
