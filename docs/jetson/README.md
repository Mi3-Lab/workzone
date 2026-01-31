# Jetson Orin Workzone Fusion System (SOTA Edition)

A high-performance, real-time work zone detection system optimized for NVIDIA Jetson Orin. This application combines **YOLOv12 (TensorRT)**, **OpenCLIP**, and **ResNet18 (Scene Context)** to achieve State-of-the-Art (SOTA) robustness in complex environments.

## 🌟 Key Features

### 🧠 SOTA Intelligence
- **Scene Context Adaptation:** Automatically detects the environment (`[HIGHWAY]`, `[URBAN]`, `[SUBURBAN]`) and adjusts detection sensitivity dynamically.
    - *Highway:* High trust in infrastructure (Cones/Barrels).
    - *Urban:* Low trust in cones (noise), high reliance on workers/signs.
- **Per-Cue Verification:** Each detected object is individually verified by CLIP.
    - **Contextual Rejection:** Distinguishes between "active cones on road" vs "inactive cones stacked on a truck", rejecting false positives.
- **Night Mode Boost:** Automatic low-light detection (`[NIGHT MODE]`) activates Gamma Correction and CLAHE to enhance visibility of reflective strips.

### ⚡ Performance & Architecture
- **Threaded Inference:** Producer-Consumer architecture separates AI processing (variable FPS) from UI rendering (stable 30 FPS), ensuring fluid video output without stutter.
- **Batch Processing:** CLIP processes crop verification in optimized batches on the GPU.
- **FP16 Acceleration:** Full FP16 support for YOLO, CLIP, and Scene Classifier on Orin Tensor Cores.

### 🎮 Control & UI
- **Jetson Launcher (GUI):** Complete control panel with Hot-Reload (adjust weights live).
- **Zero-Overlap HUD:** Interface rendered in a dedicated top bar.
- **Visual Feedback:** 
    - 🟩 Green Box: Verified Object.
    - 🟥 Red Box: Rejected Object (Contextual Mismatch).
    - 🟨 Yellow Box: Raw Detection (during throttling).

## 🚀 Quick Start

### 1. Launch the Controller
Use the dedicated make command to open the GUI launcher:

```bash
make workzone
```

### 2. Modes of Operation
- **Automation (Recommended):** Enable "Scene Context Adaptation" in the launcher. The system will auto-tune parameters based on the video content.
- **Manual:** Disable automation to manually set weights via sliders.

### 3. Speed Limit Detection (DLA Optimized) 🏎️
Specialized script running on the **DLA (Deep Learning Accelerator)** for speed limit verification.

```bash
# Run DLA Speed Limit Tracker
venv/bin/python3 tools/speed_limit_dla.py --input data/demo/boston.mp4 --show
```

## 🏗️ Technical Architecture

### 1. The Frame Processor (Producer Thread)
- **Input:** Reads frame (supports recursive search for mp4/avi/mov/mkv).
- **Night Boost:** Checks brightness (<60). If dark, applies CLAHE + Gamma 0.7.
- **Scene Context:** ResNet18 classifies scene every 30 frames.
- **YOLOv12:** Detects objects on the *enhanced* frame.
- **Per-Cue Verifier:** 
    - Extracts crops of detections.
    - Compares CLIP embedding against "Active" vs "Inactive" prompts.
    - Rejects objects that match "Inactive" (e.g., cones on truck).
- **Adaptive Fusion:** Calculates score using weights specific to the detected scene.

### 2. The Render Loop (Consumer Thread)
- **Stable FPS:** Renders HUD at steady 30 FPS.
- **Smoothing:** Uses Zero-Order Hold (repeats last valid state) if inference lags slightly, preventing video jitter.

## 📊 Configuration (`configs/jetson_config.yaml`)

The system auto-saves settings here. Key sections:

| Section | Parameter | Description |
| :--- | :--- | :--- |
| `scene_context` | `enabled` | Toggle SOTA scene adaptation |
| `fusion` | `use_per_cue` | Toggle individual object verification |
| `fusion` | `per_cue_th` | CLIP threshold for verification (default 0.05) |
| `fusion.weights_yolo` | `...` | Fallback weights for Manual Mode |

## 💾 Outputs
Annotated videos are saved to `results/jetson/` with timestamped filenames.
