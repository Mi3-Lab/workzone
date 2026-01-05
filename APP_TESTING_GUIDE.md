# 🎛️ Workzone Detection System - Calibration & Testing Guide

**Comprehensive guide for using the Streamlit calibration app and CLI to tune detection parameters.**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Streamlit Calibration App](#streamlit-calibration-app)
3. [Parameter Reference](#parameter-reference)
4. [CLI Batch Processing](#cli-batch-processing)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### Launch Streamlit App

```bash
# Activate venv
source .venv/bin/activate

# Launch app
streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py
```

The app opens in your browser at `http://localhost:8501`

---

## 🎬 Streamlit Calibration App

### Overview

The **Streamlit Calibration App** provides an interactive interface for:
- 📹 **Real-time preview** with live parameter adjustment
- 📊 **Batch processing** with comprehensive visualizations
- 💾 **Export** annotated videos + detailed CSV timelines
- 🎚️ **Calibration** of all detection and fusion parameters
- 🔬 **Explainability dashboards** with Phase 2.1 per-cue diagnostics

---

### Interface Sections

#### 1. **Sidebar: Configuration Panel** (Left)

All detection parameters are controlled from the sidebar.

##### 1.1 Model + Device

| Control | Description | Options |
|---------|-------------|---------|
| **Device** | Compute device | Auto / GPU (cuda) / CPU |

- **Auto**: Automatically selects GPU if available, otherwise CPU
- **GPU**: Forces CUDA (requires NVIDIA GPU with CUDA drivers)
- **CPU**: Forces CPU (slower, but works anywhere)

##### 1.2 YOLO Model

| Control | Description | Recommended |
|---------|-------------|-------------|
| **Model** | YOLO checkpoint selection | Hard-Negative Trained |

Options:
- **Hard-Negative Trained**: `weights/yolo12s_hardneg_1280.pt` (84.6% FP reduction) ✅
- **Fusion Baseline**: `weights/yolo12s_fusion_baseline.pt` (baseline model)
- **Upload Custom**: Upload your own `.pt` checkpoint

##### 1.3 Run Mode

| Mode | Description | Use Case |
|------|-------------|----------|
| **Live preview (real time)** | Stream processed frames to screen | Parameter tuning, visual debugging |
| **Batch (save outputs)** | Process entire video and save results | Final runs, CSV export, video annotation |

- **Live Preview**: Renders frames in real-time with adjustable playback speed. Great for quickly testing parameter changes. No outputs saved.
- **Batch Mode**: Processes full video, saves annotated MP4 + CSV timeline. Slower, but generates artifacts for analysis.

##### 1.4 Inference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Confidence** | 0.05 - 0.90 | 0.25 | YOLO confidence threshold. Lower = more detections (more sensitive, more FPs). Higher = fewer detections (more precise, may miss objects). |
| **IoU** | 0.10 - 0.90 | 0.70 | Non-Maximum Suppression IoU. Lower = fewer overlapping boxes. Higher = allows more overlap. |
| **Frame stride** | 1 - 30 | 2 | Process every Nth frame. 1 = every frame (slow, most accurate). 2-5 = good balance. 10+ = fast but may miss short events. |

**Tips**:
- Start with **conf=0.25**, **iou=0.70**, **stride=2** for general use.
- Increase **stride** to 5-10 for faster processing during calibration.
- Lower **conf** to 0.15-0.20 if missing small/distant objects.

##### 1.5 YOLO Semantic Weights

These weights control how strongly each object type contributes to the final YOLO score.

| Weight | Range | Default | Description |
|--------|-------|---------|-------------|
| **bias** | -1.0 - 0.5 | -0.35 | Baseline score offset. Negative values reduce false alarms when no objects present. |
| **channelization** | 0.0 - 2.0 | 0.9 | Weight for cones, drums, barriers, delineators. **Strong indicator** of work zones. |
| **workers** | 0.0 - 2.0 | 0.8 | Weight for construction workers in safety vests. **Reliable but may be occluded**. |
| **vehicles** | 0.0 - 2.0 | 0.5 | Weight for work trucks, construction vehicles. **Moderate indicator** (can appear outside work zones). |
| **ttc_signs** | 0.0 - 2.0 | 0.7 | Weight for temporary traffic control signs. **Good indicator** but sometimes distant. |
| **message_board** | 0.0 - 2.0 | 0.6 | Weight for arrow boards, message boards. **Reliable** when visible. |

**Tuning Strategy**:
- **Increase** weights for cues that are reliably present in your work zones.
- **Decrease** weights for cues prone to false positives (e.g., vehicles on regular roads).
- **Negative bias** (-0.3 to -0.4) helps reduce false alarms in empty scenes.

**Example Scenarios**:
- **Highway work zones**: Increase `channelization` and `ttc_signs` (common on highways).
- **Urban construction**: Increase `workers` and `vehicles` (more visible in urban settings).
- **Night scenes**: Increase `message_board` (illuminated boards are very visible).

##### 1.6 State Machine

Controls state transitions: **OUT** → **APPROACHING** → **INSIDE** → **EXITING** → **OUT**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Enter threshold** | 0.50 - 0.95 | 0.70 | Score required to transition APPROACHING → INSIDE. Higher = stricter entry (fewer false entries). |
| **Exit threshold** | 0.05 - 0.70 | 0.45 | Score must drop below this to start exiting. Lower = faster exits. |
| **Approach threshold** | 0.10 - 0.90 | 0.55 | Score to transition OUT → APPROACHING. Intermediate state before full entry. |
| **Min INSIDE frames** | 1 - 100 | 25 | Minimum frames to stay INSIDE before allowing exit. Prevents flicker. |
| **Min OUT frames** | 1 - 50 | 15 | Minimum frames to stay OUT/EXITING before re-entry. Prevents rapid re-entry. |

**State Machine Logic**:
```
OUT (score ≥ approach_th) → APPROACHING
APPROACHING (score ≥ enter_th) → INSIDE
INSIDE (score < exit_th AND inside_frames ≥ min_inside_frames) → EXITING
EXITING (out_frames ≥ min_out_frames) → OUT
```

**Tuning Strategy**:
- **High precision** (fewer false positives): Increase `enter_th` to 0.75-0.80, increase `min_inside_frames` to 30-40.
- **High recall** (catch all work zones): Lower `enter_th` to 0.60-0.65, lower `approach_th` to 0.50.
- **Reduce flicker**: Increase `min_inside_frames` and `min_out_frames`.

**Hysteresis**: `enter_th` > `exit_th` creates hysteresis (requires higher score to enter than to exit). This prevents rapid state oscillation.

##### 1.7 EMA + CLIP

**EMA (Exponential Moving Average)** smooths scores over time to reduce noise.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **EMA alpha** | 0.05 - 0.60 | 0.25 | Smoothing factor. **Lower** = slower response, smoother. **Higher** = faster response, noisier. |

**CLIP (Semantic Verification)** uses vision-language models to verify work zone semantics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Enable CLIP** | Checkbox | ✅ | Enable/disable CLIP verification. |
| **Positive prompt** | Text | "a road work zone with traffic cones, barriers, workers, construction signs" | Description of work zones. |
| **Negative prompt** | Text | "a normal road with no construction and no work zone" | Description of non-work zones. |
| **CLIP weight** | 0.0 - 0.8 | 0.35 | Fusion weight. 0 = no CLIP, 1 = only CLIP. Recommended: 0.3-0.4. |
| **CLIP trigger (YOLO ≥)** | 0.0 - 1.0 | 0.45 | Only apply CLIP when YOLO score ≥ this threshold. Saves computation. |

**How CLIP Works**:
1. YOLO produces initial score.
2. If score ≥ `clip_trigger_th`, run CLIP on the frame.
3. CLIP computes similarity to positive/negative prompts.
4. Fused score = `(1 - clip_weight) * yolo_score + clip_weight * clip_score`.

**Tuning Strategy**:
- **clip_weight = 0.35**: Good default balance.
- **clip_trigger_th = 0.45**: Apply CLIP only to frames with moderate YOLO confidence (saves GPU).
- **Increase clip_weight** (0.4-0.5) if YOLO is over-detecting false positives.
- **Decrease clip_weight** (0.2-0.3) if CLIP is too conservative.

**Prompt Engineering**:
- **Positive prompt**: Include specific visual cues (cones, workers, signs, barriers, orange vests, hard hats, construction vehicles).
- **Negative prompt**: Describe typical false positive scenes (regular roads, residential streets, parking lots, highways without construction).

##### 1.8 Orange-Cue Boost

**Orange Boost** detects orange pixels (common in work zones: cones, vests, signs) to boost confidence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Enable orange boost** | Checkbox | ✅ | Enable/disable orange pixel detection. |
| **Orange weight** | 0.0 - 0.6 | 0.25 | How much orange pixels boost the score. |
| **Trigger if YOLO_ema <** | 0.0 - 1.0 | 0.55 | Only apply boost when YOLO score is below this. Avoids over-boosting high-confidence detections. |

**HSV Parameters** (Advanced - expand to adjust):
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Hue low** | 0 - 179 | 5 | Lower bound of orange hue in HSV space. |
| **Hue high** | 0 - 179 | 25 | Upper bound of orange hue. |
| **Sat min** | 0 - 255 | 80 | Minimum saturation (filters out pale/washed out colors). |
| **Val min** | 0 - 255 | 50 | Minimum value (filters out very dark pixels). |
| **Center (ratio)** | 0.00 - 0.30 | 0.08 | Orange ratio at which boost is 0.5 (logistic center). |
| **Slope (k)** | 1.0 - 60.0 | 30.0 | Steepness of boost curve. Higher = sharper transition. |

**How It Works**:
1. Convert frame to HSV color space.
2. Count pixels within orange hue/sat/val ranges.
3. Compute orange ratio = (orange pixels) / (total pixels).
4. Map ratio to boost score via logistic function.
5. If YOLO score < `context_trigger_below`, fuse boost: `fused = (1 - orange_weight) * fused + orange_weight * orange_boost`.

**Tuning Strategy**:
- **Default settings work well** for most cases.
- **Increase orange_weight** (0.3-0.4) in scenes with lots of orange cues but weak YOLO detections.
- **Adjust HSV ranges** if orange objects are not being detected (check frame HSV values).

##### 1.9 Phase 1.4: Scene Context

**Scene Context** classifies the environment (Highway / Urban / Suburban) and adjusts thresholds accordingly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Enable Scene Context** | Checkbox | ✅ (if available) | Enable Phase 1.4 scene-adaptive thresholds. |

**How It Works**:
1. Classify scene as Highway, Urban, or Suburban using ResNet18.
2. Override state machine thresholds based on scene type:
   - **Highway**: Higher thresholds (fewer false positives on long stretches of highway).
   - **Urban**: Lower thresholds (more sensitive to detect smaller urban work zones).
   - **Suburban**: Balanced thresholds.
3. Thresholds are auto-adjusted every N frames (configurable).

**Scene-Specific Thresholds** (auto-applied):
| Scene | enter_th | exit_th | approach_th |
|-------|----------|---------|-------------|
| **Highway** | 0.75 | 0.50 | 0.60 |
| **Urban** | 0.65 | 0.40 | 0.50 |
| **Suburban** | 0.70 | 0.45 | 0.55 |

**When to Use**:
- ✅ Videos with **mixed environments** (e.g., highway → urban transitions).
- ✅ Want **adaptive thresholds** without manual tuning.
- ❌ Single-environment videos (manual threshold tuning is simpler).

##### 1.10 Phase 2.1: Per-Cue Verification + Motion Tracking

**Phase 2.1** adds fine-grained verification: per-cue CLIP confidences and motion plausibility.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Enable Per-Cue Verification + Motion Tracking** | Checkbox | ✅ (if available) | Enable Phase 2.1 features. |

**Phase 2.1 Features**:

1. **Per-Cue CLIP Verification**:
   - Instead of global CLIP score, compute CLIP confidence **separately** for each cue type:
     - Channelization (cones, drums, barriers)
     - Workers (construction workers in vests)
     - Vehicles (work trucks, construction vehicles)
     - Signs (temporary traffic signs)
     - Equipment (construction equipment)
   - Each cue gets its own positive/negative CLIP prompts.
   - Outputs 5 confidence scores (one per cue type).

2. **Motion Plausibility Tracking**:
   - Track objects across frames using centroid + IoU matching.
   - Validate trajectory consistency:
     - Channelization should be **stationary** (max speed 5 px/frame).
     - Workers may move **slowly** (max speed 20 px/frame).
     - Vehicles should be **parked/slow** (max speed 10 px/frame).
     - Signs/equipment should be **stationary** (max speed 3-5 px/frame).
   - Compute overall motion plausibility score (0-1).
   - Low plausibility = likely false positive (moving objects = not a stationary work zone).

**When to Use**:
- ✅ Need **fine-grained explainability** (which cues are firing?).
- ✅ Want to **filter motion-based false positives** (e.g., moving vehicles misclassified as work vehicles).
- ✅ Building **temporal attention models** (Phase 2.1 outputs feed into attention networks).
- ❌ Simple deployments where global CLIP is sufficient.

**Output**:
- CSV columns: `cue_conf_channelization`, `cue_conf_workers`, `cue_conf_vehicles`, `cue_conf_signs`, `cue_conf_equipment`, `motion_plausibility`.
- Dashboard plots: Per-cue confidences over time, motion plausibility timeline.

##### 1.11 OCR Text Extraction

**OCR** extracts text from message boards and traffic signs to boost confidence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Enable OCR** | Checkbox | ✅ (if available) | Enable OCR text extraction and classification. |

**How It Works**:
1. Detect message boards and traffic signs in YOLO output.
2. Crop bounding box + padding.
3. Run **EasyOCR** (GPU-accelerated) to extract text.
4. Classify text into categories:
   - **WORKZONE**: "WORK AHEAD", "ROAD WORK", "CONSTRUCTION ZONE", etc.
   - **SPEED**: "SPEED LIMIT 45", "REDUCE SPEED", etc.
   - **LANE**: "LEFT LANE CLOSED", "MERGE RIGHT", etc.
   - **CAUTION**: "CAUTION", "SLOW", etc.
   - **DIRECTION**: Arrows, "KEEP RIGHT", etc.
   - **UNCLEAR**: Noisy/unreadable text.
5. If high-confidence WORKZONE text (conf ≥ 0.70), boost fused score by up to 15%.

**Text Classification** (rule-based):
- **97.7% accuracy** on test set (43/44 correct).
- Filters out noise (63 unclear cases detected).
- Useful rate: 39% (up from 26% without classification).

**Tuning Strategy**:
- **Enable** if videos contain message boards with readable text.
- **Disable** if OCR is slow or videos have no text signs.
- OCR runs **every 2 frames** by default to reduce latency (configurable in CLI).

##### 1.12 Save Video

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Save video** | Checkbox | ✅ | Save annotated video output in batch mode. |

- Saves MP4 with:
  - YOLO bounding boxes
  - State banner (top of frame)
  - OCR text overlay (if enabled)
  - CLIP active indicator
- Effective FPS = `original_fps / stride`.

---

#### 2. **Main Area: Video Input & Processing** (Center/Right)

##### 2.1 Video Source

Choose video input source:

| Source | Description |
|--------|-------------|
| **Demo** | Select from pre-loaded demo videos in `data/03_demo/videos/`. |
| **Dataset** | Select from dataset videos in `data/videos_compressed/`. |
| **Upload** | Upload custom video (MP4, MOV, AVI). |

##### 2.2 Run Buttons

**Live Preview Mode**:
- Click **▶️ Start Live Preview** to begin real-time processing.
- Frames render in the app with current parameters.
- Adjust sliders in sidebar to see changes immediately.
- No outputs saved.

**Batch Mode**:
- Click **🚀 Process Video** to process full video.
- Outputs:
  - Annotated video: `<temp_dir>/<video_name>_calibrated.mp4`
  - Timeline CSV: `<temp_dir>/<video_name>_calibrated.csv`
- Visualizations appear after processing:
  - Score over time plot
  - State transition histogram
  - Explainability dashboard (OCR, CLIP, object counts, persistence counters)
  - Phase 2.1 plots (if enabled): per-cue confidences, motion plausibility
- Download buttons for CSV and video.

---

### Output CSV Schema

The timeline CSV contains per-frame metrics. Columns:

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int | Frame number (0-indexed). |
| `time_sec` | float | Timestamp in seconds. |
| `yolo_score` | float | Raw YOLO semantic score (0-1). |
| `yolo_ema` | float | EMA-smoothed YOLO score. |
| `fused_ema` | float | Final fused score after CLIP, OCR, orange boost. |
| `state` | str | State: OUT / APPROACHING / INSIDE / EXITING. |
| `inside_frames` | int | Persistence counter for INSIDE state. |
| `out_frames` | int | Persistence counter for OUT/EXITING state. |
| `clip_used` | int | 1 if CLIP was applied, 0 otherwise. |
| `clip_score` | float | CLIP similarity score (if applied). |
| `count_channelization` | int | Number of channelization objects detected. |
| `count_workers` | int | Number of workers detected. |
| `count_vehicles` | int | Number of work vehicles detected. |
| `ocr_text` | str | Extracted OCR text (if any). |
| `text_confidence` | float | OCR confidence * text classification confidence. |
| `text_category` | str | Text category: WORKZONE / SPEED / LANE / CAUTION / DIRECTION / UNCLEAR / NONE. |
| `scene_context` | str | Scene type: highway / urban / suburban (Phase 1.4). |
| **Phase 2.1 columns** (if enabled): |
| `cue_conf_channelization` | float | Per-cue CLIP confidence for channelization. |
| `cue_conf_workers` | float | Per-cue CLIP confidence for workers. |
| `cue_conf_vehicles` | float | Per-cue CLIP confidence for vehicles. |
| `cue_conf_signs` | float | Per-cue CLIP confidence for signs. |
| `cue_conf_equipment` | float | Per-cue CLIP confidence for equipment. |
| `motion_plausibility` | float | Motion plausibility score (0-1). |

---

## 📚 Parameter Reference

### Quick Reference Table

| Category | Parameter | Default | Range | Impact |
|----------|-----------|---------|-------|--------|
| **Inference** | Confidence | 0.25 | 0.05-0.90 | Lower = more detections, higher = fewer FPs |
| **Inference** | IoU | 0.70 | 0.10-0.90 | NMS overlap threshold |
| **Inference** | Stride | 2 | 1-30 | Process every N frames |
| **YOLO Weights** | Bias | -0.35 | -1.0-0.5 | Baseline offset |
| **YOLO Weights** | Channelization | 0.9 | 0.0-2.0 | Cones, drums, barriers weight |
| **YOLO Weights** | Workers | 0.8 | 0.0-2.0 | Construction workers weight |
| **YOLO Weights** | Vehicles | 0.5 | 0.0-2.0 | Work vehicles weight |
| **YOLO Weights** | TTC Signs | 0.7 | 0.0-2.0 | Traffic signs weight |
| **YOLO Weights** | Message Board | 0.6 | 0.0-2.0 | Message boards weight |
| **State Machine** | Enter threshold | 0.70 | 0.50-0.95 | APPROACHING → INSIDE |
| **State Machine** | Exit threshold | 0.45 | 0.05-0.70 | INSIDE → EXITING |
| **State Machine** | Approach threshold | 0.55 | 0.10-0.90 | OUT → APPROACHING |
| **State Machine** | Min INSIDE frames | 25 | 1-100 | Anti-flicker (inside) |
| **State Machine** | Min OUT frames | 15 | 1-50 | Anti-flicker (outside) |
| **EMA** | Alpha | 0.25 | 0.05-0.60 | Smoothing factor |
| **CLIP** | Weight | 0.35 | 0.0-0.8 | CLIP fusion weight |
| **CLIP** | Trigger threshold | 0.45 | 0.0-1.0 | Apply CLIP when YOLO ≥ this |
| **Orange Boost** | Weight | 0.25 | 0.0-0.6 | Orange pixel boost weight |
| **Orange Boost** | Trigger below | 0.55 | 0.0-1.0 | Apply when YOLO < this |

---

## 🖥️ CLI Batch Processing

For headless/scripted workflows, use the command-line interface.

### Basic Usage

```bash
python scripts/process_video_fusion.py \
  <input_video> \
  --output-dir <output_folder> \
  [options]
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device: `cuda` or `cpu` | `cuda` |
| `--conf` | YOLO confidence threshold | `0.25` |
| `--iou` | YOLO IoU threshold | `0.70` |
| `--stride` | Frame stride | `2` |
| `--ema-alpha` | EMA smoothing | `0.25` |
| `--no-clip` | Disable CLIP | `False` (CLIP enabled) |
| `--clip-weight` | CLIP fusion weight | `0.35` |
| `--clip-trigger-th` | CLIP trigger threshold | `0.45` |
| `--enter-th` | State machine enter threshold | `0.70` |
| `--exit-th` | State machine exit threshold | `0.45` |
| `--approach-th` | State machine approach threshold | `0.55` |
| `--min-inside-frames` | Min frames inside | `25` |
| `--min-out-frames` | Min frames outside | `15` |
| `--orange-weight` | Orange boost weight | `0.25` |
| `--enable-phase1-1` | Enable multi-cue temporal logic | `False` |
| `--enable-phase1-4` | Enable scene context | `False` |
| `--enable-phase2-1` | Enable per-cue + motion tracking | `False` |
| `--enable-ocr` | Enable OCR | `False` |
| `--no-video` | Skip video output | `False` |
| `--no-csv` | Skip CSV output | `False` |
| `--quiet` | Suppress progress prints | `False` |

### Example Commands

#### 1. Basic Run (Phase 1.0)

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/basic
```

#### 2. With CLIP Tuning

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/clip_tuned \
  --clip-weight 0.40 \
  --clip-trigger-th 0.40
```

#### 3. Phase 1.4 (Scene Context + OCR)

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/phase1_4 \
  --enable-phase1-4 \
  --enable-ocr
```

#### 4. Phase 2.1 (Full Pipeline)

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/phase2_1 \
  --enable-phase1-1 \
  --enable-phase2-1 \
  --enable-ocr \
  --no-motion \
  --stride 2
```

#### 5. High-Throughput Batch (No Video Output)

```bash
python scripts/process_video_fusion.py \
  data/dataset/*.mp4 \
  --output-dir outputs/batch \
  --stride 5 \
  --no-video \
  --quiet
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. **App won't launch / crashes immediately**

**Symptoms**:
- `streamlit: command not found`
- `ModuleNotFoundError`

**Solutions**:
```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Check streamlit installation
streamlit --version
```

#### 2. **CUDA out of memory**

**Symptoms**:
- `RuntimeError: CUDA out of memory`
- GPU crashes during processing

**Solutions**:
```bash
# Increase frame stride (process fewer frames)
--stride 5

# Reduce batch size (if training)
--batch-size 1

# Use CPU instead
--device cpu

# Close other GPU applications
```

#### 3. **Slow inference (< 5 FPS on GPU)**

**Symptoms**:
- Processing takes hours
- Live preview is choppy

**Solutions**:
- **Increase stride**: `--stride 5` or `--stride 10` for calibration.
- **Disable CLIP**: `--no-clip` (saves ~15ms/frame).
- **Disable OCR**: Uncheck "Enable OCR" in UI.
- **Disable Phase 2.1**: Uncheck Phase 2.1 (saves trajectory tracking overhead).
- **Lower resolution**: Use 640px YOLO model (if available).

#### 4. **OCR fails or returns garbage**

**Symptoms**:
- OCR text is blank or nonsensical
- `PaddleOCR failed`

**Solutions**:
```bash
# EasyOCR is the primary OCR engine (more robust)
# Check EasyOCR installation
pip install easyocr

# Test EasyOCR
python -c "import easyocr; reader = easyocr.Reader(['en']); print('OK')"

# If PaddleOCR is problematic, it will auto-fallback to EasyOCR
```

#### 5. **Models not found**

**Symptoms**:
- `FileNotFoundError: weights/yolo12s_hardneg_1280.pt`
- `Model not found`

**Solutions**:
```bash
# Download models
bash scripts/download_models.sh

# Or manually download from Hugging Face/Google Drive
# Place in weights/ folder
```

#### 6. **Phase 2.1 errors**

**Symptoms**:
- `'CueClassifier' object has no attribute 'classify'`
- Phase 2.1 warnings

**Solutions**:
- **Update code**: Ensure you have the latest version with `classify()` method.
- **Disable Phase 2.1**: Uncheck the checkbox to bypass errors.
- **Check imports**: Ensure `workzone.detection.cue_classifier`, `workzone.models.per_cue_verification`, `workzone.models.trajectory_tracking` are importable.

---

## ✅ Best Practices

### 1. **Calibration Workflow**

1. **Start with defaults**: Use default parameters first to establish a baseline.
2. **Use live preview**: Adjust parameters in real-time while watching the video.
3. **Increase stride**: Use stride=5-10 during calibration for faster iteration.
4. **Save configurations**: Take notes or export parameter sets that work well.
5. **Batch process**: Once calibrated, run batch mode with stride=1-2 for final results.

### 2. **Parameter Tuning Order**

1. **YOLO semantic weights**: Adjust weights based on which cues are most reliable in your dataset.
2. **State machine thresholds**: Tune enter/exit thresholds to balance precision/recall.
3. **CLIP fusion**: Fine-tune CLIP weight and trigger threshold.
4. **EMA alpha**: Adjust smoothing if scores are too noisy or too laggy.
5. **Orange boost**: Enable if YOLO misses orange-heavy work zones.
6. **Phase 1.4 / 2.1**: Enable advanced features after basic pipeline is working.

### 3. **Avoiding False Positives**

- **Increase enter_th**: 0.75-0.80 for stricter entry.
- **Increase min_inside_frames**: 30-50 to require sustained detections.
- **Negative YOLO bias**: -0.4 to -0.5 to reduce baseline score.
- **Enable Phase 2.1 motion tracking**: Filter moving objects.
- **CLIP verification**: Increase CLIP weight to 0.4-0.5 to rely more on semantic verification.

### 4. **Avoiding False Negatives (Missing Work Zones)**

- **Lower enter_th**: 0.60-0.65 for easier entry.
- **Lower confidence**: 0.15-0.20 to detect more objects.
- **Increase YOLO weights**: Boost weights for reliable cues (channelization, workers).
- **Enable orange boost**: Helps detect orange-heavy scenes.
- **Enable OCR**: Text detection can catch work zones with visible message boards.

### 5. **Dataset-Specific Tuning**

| Dataset Type | Recommended Settings |
|--------------|---------------------|
| **Highway** | enter_th=0.75, channelization=1.0, ttc_signs=0.8, enable Phase 1.4 |
| **Urban** | enter_th=0.65, workers=0.9, vehicles=0.6, enable OCR |
| **Suburban** | enter_th=0.70 (default), balanced weights |
| **Night** | message_board=0.8, enable orange boost, enable OCR |
| **Daytime Clear** | Use defaults, enable CLIP for semantic verification |
| **Mixed Environment** | Enable Phase 1.4 scene context for adaptive thresholds |

### 6. **Performance Optimization**

- **GPU**: Always use GPU (`--device cuda`) for 10-20x speedup.
- **Stride**: Use stride=2-3 for balanced speed/accuracy.
- **Disable unused features**: If not using OCR/Phase2.1, disable them to save time.
- **Batch processing**: Process multiple videos in parallel (separate terminals).
- **TensorRT**: Export YOLO to TensorRT for edge deployment (Jetson Orin).

---

## 📊 Interpreting Results

### 1. **Score Plots**

- **YOLO EMA** (blue): Smoothed YOLO confidence. Should be high inside work zones.
- **Fused EMA** (orange): Final fused score after CLIP/OCR/orange boost. Used for state transitions.
- **Thresholds** (dashed lines): Green = enter_th, Red = exit_th.

**Good calibration**:
- Fused score clearly above enter_th inside work zones.
- Fused score clearly below exit_th outside work zones.
- Minimal oscillation around thresholds.

**Poor calibration**:
- Score hovers around thresholds (frequent state changes).
- Score doesn't reach enter_th in obvious work zones (false negatives).
- Score exceeds enter_th in non-work zones (false positives).

### 2. **State Transition Histogram**

Shows time spent in each state. Ideal:
- **INSIDE**: Majority of time inside work zone scenes.
- **OUT**: Majority of time outside work zones.
- **APPROACHING / EXITING**: Minimal time (fast transitions).

If you see:
- **Long APPROACHING**: Increase approach_th or lower enter_th.
- **Long EXITING**: Increase min_out_frames or lower exit_th.
- **Frequent transitions**: Increase min_inside_frames and min_out_frames.

### 3. **Explainability Dashboard**

#### OCR + CLIP Cues:
- **OCR confidence** (blue line): High when message board text is detected.
- **CLIP score** (orange dots): CLIP similarity scores.
- **CLIP trigger** (red X): Frames where CLIP was applied.

#### Object Counts:
- Shows channelization, workers, vehicles counts over time.
- Should correlate with work zone presence.

#### Persistence Counters:
- **Inside counter** (green): Increases while INSIDE.
- **Outside counter** (red): Increases while OUT/EXITING.
- Used for anti-flicker logic.

### 4. **Phase 2.1 Plots** (if enabled)

#### Per-Cue Confidences:
- 5 lines (one per cue type).
- High confidence = CLIP strongly agrees that cue is present.
- Helps identify which cues are firing at each moment.

#### Motion Plausibility:
- High (0.8-1.0): Objects are stationary/slow (consistent with work zone).
- Low (0.0-0.5): Objects are moving fast (likely not a work zone).
- Threshold at 0.5 (dashed red line).

---

## 🎓 Tutorials

### Tutorial 1: Calibrate for Highway Dataset

**Goal**: Tune parameters for highway work zones with long stretches of cones and signs.

**Steps**:
1. Launch app: `streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py`
2. Select a highway video from Dataset.
3. Set **Live preview** mode.
4. Adjust:
   - `channelization weight = 1.0` (cones are common)
   - `ttc_signs weight = 0.8` (signs are visible)
   - `enter_th = 0.75` (stricter entry for long highways)
   - `exit_th = 0.50` (require clear absence of cones to exit)
5. Watch preview and adjust until state transitions look correct.
6. Switch to **Batch mode**, run full video, download CSV.

### Tutorial 2: Reduce False Positives in Urban Scenes

**Goal**: Avoid false alarms from construction vehicles in urban areas.

**Steps**:
1. Select an urban video with false positives.
2. Set **Live preview**.
3. Adjust:
   - `vehicles weight = 0.3` (reduce vehicle weight)
   - `workers weight = 0.9` (increase worker weight - more reliable in urban)
   - `enter_th = 0.75` (stricter entry)
   - `min_inside_frames = 40` (require sustained detections)
4. Enable **CLIP** and increase `clip_weight = 0.45` (rely more on semantic verification).
5. Enable **Phase 2.1** to filter moving vehicles (motion plausibility).
6. Test and batch process.

### Tutorial 3: Night Video with Message Boards

**Goal**: Detect work zones in night videos using illuminated message boards.

**Steps**:
1. Select a night video.
2. Enable **OCR**.
3. Adjust:
   - `message_board weight = 0.9` (message boards are very visible at night)
   - Enable **Orange boost** (orange lights on boards).
   - `enter_th = 0.65` (lower threshold for sparse detections at night)
4. Check OCR text in overlay - should detect "WORK AHEAD", etc.
5. Batch process and verify CSV has `ocr_text` entries.

---

**Happy Calibrating! 🚧**
