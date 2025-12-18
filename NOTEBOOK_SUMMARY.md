# Workzone Project - Jupyter Notebooks Summary

A comprehensive analysis of all notebooks in the workzone project, designed to facilitate conversion to professional Python modules.

---

## 1. **01_workzone_yolo_setup.ipynb**

### Purpose
Convert ROADWork COCO annotations to YOLO format for training object detectors.

### Key Imports
- `pathlib.Path`
- `json`
- `dataclasses`
- `collections.defaultdict`
- `shutil`, `os`
- `typing` (Dict, List)
- `yaml`

### Main Functionality

#### Data Input
- ROADWork annotations in COCO format:
  - Train: `instances_train_gps_split_with_signs.json`
  - Val: `instances_val_gps_split_with_signs.json`
- Image directory: `data/images/`

#### Key Classes/Functions
1. **`coco_to_yolo_bbox(bbox, img_width, img_height)`**
   - Converts COCO format `[x, y, w, h]` to YOLO `(x_center_norm, y_center_norm, w_norm, h_norm)`
   - Normalizes coordinates to [0, 1] range

2. **`convert_split_coco_to_yolo(images, anns_by_image, id2img, img_src_dir, img_out_dir, lbl_out_dir, split_name)`**
   - Processes all images and annotations for a split
   - Filters categories based on `roadwork_to_yolo` mapping
   - Writes `.txt` label files and copies images

#### Work Zone Classes (50 classes)
- **Core:** Police Officer, Police Vehicle, Cone, Fence, Drum, Barricade, Barrier, Work Vehicle, Vertical Panel, Tubular Marker, Arrow Board, Bike Lane, Work Equipment, Worker
- **Traffic Control Message Board, Traffic Control Signs (various arrow/chevron types)**
- **Special Signs:** No turn signs, Pedestrian directions, Stop signs, Worker/Bicycle warnings, etc.

#### Data Outputs
```
workzone_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── workzone_yolo.yaml
```

#### YAML Structure
```yaml
path: <path_to_yolo_root>
train: images/train
val: images/val
names:
  0: "Police Officer"
  1: "Police Vehicle"
  ... (50 classes total)
```

#### Dependencies
- None (standard library + optional yaml for output)

---

## 2. **02_workzone_yolo_train_eval.ipynb**

### Purpose
Train YOLO detector on workzone dataset, evaluate performance, and measure inference speed.

### Key Imports
- `pathlib.Path`
- `time`
- `torch`
- `ultralytics.YOLO`
- `cv2` (OpenCV)
- `numpy as np`
- `IPython.display` (Image, display)

### Main Functionality

#### Model Selection
- Base models: `yolo12s.pt`, `yolov8n.pt` (nano), `yolov8m.pt` (medium)
- Recommended: `yolo12s.pt` for balance of speed/accuracy

#### Training Configuration
```python
Training Parameters:
- Image size (imgsz): 640
- Batch size: 4 (adjustable for memory)
- Epochs: 100
- Confidence threshold: 0.4
- IoU threshold: 0.5
```

#### Key Functions
1. **Model Loading**
   ```python
   model = YOLO("yolo12s.pt")
   results = model.train(
       data=str(DATA_YAML),
       imgsz=640,
       epochs=100,
       batch=4,
       project="runs_workzone",
       name="yolo12s_workzone",
   )
   ```

2. **Validation**
   ```python
   metrics = model.val(
       data=str(DATA_YAML),
       imgsz=640,
       split="val",
       save_json=True
   )
   ```

3. **FPS Benchmarking**
   - Measures inference speed on 50 validation images
   - Warms up GPU with 5 images first
   - Returns approximate FPS for single image inference

#### Outputs
- Trained weights: `runs_workzone/yolov8s_workzone/weights/best.pt`
- Training curves: `results.png`
- Confusion matrix: `confusion_matrix.png`
- Validation metrics (mAP@0.5, mAP@0.5:0.95, per-class metrics)

#### Dependencies
- `ultralytics` (YOLO)
- `torch` (PyTorch)
- `cv2` (OpenCV)
- `numpy`

---

## 3. **03_workzone_yolo_video_demo.ipynb**

### Purpose
Apply trained YOLO model to video, compute work-zone score, and annotate output video.

### Key Imports
- `pathlib.Path`
- `cv2` (video I/O, drawing)
- `numpy as np`
- `ultralytics.YOLO`
- `tqdm.auto`

### Main Functionality

#### Configuration
```python
conf_thres = 0.4
iou_thres = 0.5
Input video: data/demo/boston_workzone_short.mp4
Output video: runs_workzone/video_demos/boston_workzone_annotated.mp4
```

#### Work Zone Score Computation
```python
workzone_weights = {
    "Cone": 1.0,
    "Drum": 1.2,
    "Barricade": 1.5,
    "Barrier": 1.0,
    "Vertical Panel": 1.0,
    "Work Vehicle": 2.0,
    "Worker": 2.5,
    "Arrow Board": 1.5,
    "Temporary Traffic Control Message Board": 1.5,
    "Temporary Traffic Control Sign": 1.2,
}

score = Σ(weight × confidence) / max_possible  [0, 1]
```

#### Key Functions
1. **`compute_work_zone_score(result)`**
   - Iterates over detected boxes
   - Weights by class importance
   - Returns normalized score [0, 1]

2. **Video Processing Loop**
   - Read frame → YOLO inference → compute score → overlay → write

#### Banner Output
- **Red (score > 0.6):** "WORK ZONE - score X.XX"
- **Orange (0.3 < score < 0.6):** "POSSIBLE WORK ZONE - score X.XX"
- **Green (score < 0.3):** "NO WORK ZONE - score X.XX"

#### Dependencies
- `ultralytics` (YOLO)
- `cv2` (OpenCV)
- `numpy`
- `tqdm`

---

## 4. **04_workzone_video_state_machine.ipynb**

### Purpose
Implement stateful work-zone detection with hysteresis, EMA smoothing, and flicker reduction via class grouping and proximity heuristics.

### Key Imports
- `pathlib.Path`, `json`, `math`, `time`
- `cv2`, `numpy`, `pandas`
- `torch`, `ultralytics.YOLO`
- `tqdm.auto`
- `dataclasses`

### Main Functionality

#### Class Grouping (6 Groups)
```python
"channelization": cone, drum, barricade, barrier, vertical panel, tubular marker, fence
"workers": worker, police officer, flagger
"vehicles": work vehicle, police vehicle
"ttc_signs": temporary traffic control sign (all variants)
"message_board": message board, arrow board
"other_roadwork": remaining classes
```

#### WorkzoneConfig (Tunable Parameters)
```python
@dataclass
class WorkzoneConfig:
    w_channel: float = 1.0      # weight for channelization
    w_workers: float = 1.1      # weight for workers
    w_vehicles: float = 0.6     # weight for vehicles
    w_ttc: float = 0.9          # weight for TTC signs
    w_msg: float = 0.9          # weight for message boards
    w_near: float = 1.2         # weight for proximity (bbox area)
    
    bias: float = -4.0          # baseline (higher = harder to trigger)
    ema_alpha: float = 0.20     # EMA smoothing [0, 1]
    
    enter_th: float = 0.62      # threshold to enter WORKZONE state
    exit_th: float = 0.45       # threshold to exit WORKZONE state
    
    min_enter_sec: float = 1.0  # min time above enter_th
    min_exit_sec: float = 1.0   # min time below exit_th
    cooldown_after_enter_sec: float = 1.0  # ignore exit for N sec after entering
```

#### Key Functions

1. **`compute_near_proxy(boxes_xyxy, img_w, img_h) -> float`**
   - Approximates proximity via bbox area / image area
   - Returns fraction of "near-ish" boxes (area > 2% of image)

2. **`compute_frame_score(groups_count, near_proxy, cfg) -> (score, raw)`**
   - Weighted sum of group detections
   - Logistic sigmoid maps to [0, 1]
   - `raw = bias + Σ(weight × count) + w_near × near_proxy`

3. **`run_state_machine(timeline_df, cfg) -> DataFrame`**
   - Implements hysteresis + cooldown logic
   - Outputs: `score_ema`, `is_workzone`, `state`, `toggle_count_so_far`
   - Prevents flickering via EMA + min_enter_sec/min_exit_sec

4. **`process_one_video(video_path, cfg, stride, imgsz, conf, iou) -> (df, meta)`**
   - Processes every N frames (stride=3 for speed)
   - Returns per-frame timeline with:
     - `frame, time_sec, score_raw, raw`
     - `channel, workers, vehicles, ttc, msg, near_proxy` (feature counts)
     - `score_ema, is_workzone, state, toggle_count_so_far`

#### Outputs
- CSV per video: `outputs/notebook4/timelines_csv/<video>_timeline.csv`
- Summary: `outputs/notebook4/summary_all_videos.csv`
  - Metrics: duration, toggles, first_enter_time, time_in_workzone, mean_score_ema
  - Ranked by flicker rate for debugging
- Annotated videos: `outputs/notebook4/annotated_worst/` (top 10 worst)

#### Dependencies
- `ultralytics` (YOLO)
- `torch` (PyTorch)
- `cv2`, `numpy`, `pandas`
- `dataclasses`, `math`

---

## 5. **05_workzone_video_timeline_calibration.ipynb**

### Purpose
Multi-GPU batch processing with feature extraction, improved scoring with proximity heuristics (bottom-half + bbox area), and hysteresis tuning.

### Key Imports
- `pathlib.Path`, `dataclasses`
- `cv2`, `numpy`, `pandas`
- `torch`, `ultralytics.YOLO`
- `concurrent.futures` (ProcessPoolExecutor)
- `tqdm.auto`

### Main Functionality

#### Enhanced ScoreConfig
```python
@dataclass
class ScoreConfig:
    stride: int = 3
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.7
    
    ema_alpha: float = 0.20
    enter_th: float = 0.55
    exit_th: float = 0.45
    k_enter: int = 6          # consecutive frames above enter_th
    k_exit: int = 10          # consecutive frames below exit_th
    
    w_channel: float = 0.35
    w_workers: float = 0.35
    w_vehicles: float = 0.15
    w_ttc: float = 0.40
    w_msg: float = 0.40
    
    w_bottom: float = 0.45    # NEW: weight for bottom-half proximity
    w_near: float = 0.35      # NEW: weight for bbox area
    
    bias: float = -0.65
```

#### Enhanced Feature Computation
```python
compute_frame_features() returns:
- Group counts: channel, workers, vehicles, ttc, msg
- bottom_half: count of hazard objects in bottom 50% of frame
- near_proxy: sum of normalized bbox areas (clipped to [0, 1])
- total: total detections
```

#### Key Functions

1. **`bbox_area_norm(xyxy, w, h) -> float`**
   - Normalizes bbox area: `(area) / (image_area)`

2. **`compute_frame_features(result, w, h, groups) -> Dict`**
   - Counts per group + proximity proxies
   - bottom_half: objects with center in bottom 50%
   - near_proxy: accumulated bbox area for hazard classes

3. **`score_from_feats(feats, cfg) -> float`**
   - `raw = bias + Σ(w_i × feat_i) + w_near × (10 × near_proxy)`
   - `score = sigmoid(raw)`

4. **`run_hysteresis(scores, cfg) -> (ema, is_wz)`**
   - Consecutive frame logic: need k_enter frames above enter_th
   - New feature: consecutive frame accumulation instead of time-based

5. **`worker_process(video_paths, weights, device, out_dir, cfg)`**
   - Multi-GPU compatible process function
   - Loads separate YOLO instance per GPU
   - Writes partial summary CSVs

#### Multi-GPU Processing
```python
# Split videos across GPUs
if 1 GPU:
    process_all_on_DEVICE0
else:
    split_videos_50/50 between DEVICE0 and DEVICE1
    use ProcessPoolExecutor(max_workers=2)
```

#### Outputs
- Per-video timeline: `outputs/notebook5/timelines/<video>_timeline.csv`
- Summary: `outputs/notebook5/video_summary.csv`
- Hard cases:
  - `hardcases_early_enter.csv` (enters in first 2 sec)
  - `hardcases_never_enter.csv` (never triggers)
  - `hardcases_flicker.csv` (high flicker rate)

#### Key Metrics
- `flicker_per_min = toggles / (duration_sec / 60)`
- `early_enter_flag`: enters early AND stays mostly off
- `never_enter_flag`: never triggers workzone

#### Dependencies
- `ultralytics`, `torch`, `cv2`, `numpy`, `pandas`
- `concurrent.futures`, `dataclasses`, `math`

---

## 6. **06_triggered_vlm_semantic_verification.ipynb**

### Purpose
Use CLIP vision-language model to verify and refine work-zone decisions, reducing flicker by fusing YOLO + semantic scores.

### Key Imports
- `pathlib.Path`, `json`, `math`, `time`
- `cv2`, `numpy`, `pandas`
- `torch`, `ultralytics.YOLO`
- `open_clip`
- `PIL.Image`
- `tqdm.auto`

### Main Functionality

#### CLIP Model
```python
model_name = "ViT-B-32"
pretrained = "openai"
# Device: cuda or cpu
```

#### Semantic Prompts (7 prompts)
```python
PROMPTS = [
    "a road work zone under construction",
    "traffic cones and construction barriers on the road",
    "road work ahead sign",
    "lane closed ahead",
    "lane shift ahead",
    "a normal road with no construction",
    "a normal street scene with no road work",
]

workzone_semantic = sum(first 5) - sum(last 2)
```

#### Trigger Frame Selection
Selects frames for CLIP evaluation:
1. **Baseline:** Every N frames
2. **Uncertainty:** frames where score_ema ∈ [0.25, 0.75]
3. **Transitions:** peaks in score_ema derivative
4. **Downsampling:** enforce min_gap between selected frames

#### Key Functions

1. **`clip_score_image(frame_bgr) -> (scores_dict, workzone_sem)`**
   - CLIP vision encoder on frame
   - CLIP text encoder on prompts
   - Returns per-prompt similarity scores
   - Computes semantic workzone score

2. **`pick_trigger_frames(df_timeline, every_n_frames, ema_low, ema_high, min_gap_frames)`**
   - Selects subset of frames for VLM evaluation
   - Reduces computational cost while targeting uncertain frames

3. **`run_semantics_on_one_video(timeline_csv, imgsz, yolo_conf, yolo_iou, every_n_frames)`**
   - Runs CLIP on trigger frames
   - Detects TTC/messageboard/arrowboard boxes
   - Crops ROI and scores separately
   - Merges into timeline via forward-fill

4. **Fusion Scoring**
   ```python
   sem01 = sigmoid(3.0 * sem)  # Normalize semantics to [0,1]
   fused = 0.75 * ema + 0.25 * sem01  # 75% temporal, 25% semantic
   # Apply hysteresis on fused score
   ```

#### Outputs
- Semantics CSV: `outputs/notebook6/<video>_semantics.csv`
- Fused timeline: `outputs/notebook6/<video>_timeline_fused.csv`
  - Columns: frame, score_ema, score_fused, is_workzone, is_workzone_fused
- Index: `outputs/notebook6/notebook6_outputs_index.csv`

#### Benefits
- **Reduces flicker:** EMA + semantic fusion
- **Semantic verification:** CLIP confirms work-zone context
- **Selective evaluation:** Only runs CLIP on uncertain frames
- **Flexible tuning:** Fused score weights adjustable

#### Dependencies
- `ultralytics`, `torch`, `cv2`, `numpy`, `pandas`
- `open_clip` (CLIP)
- `PIL.Image`

---

## 7. **Workingzone.ipynb**

### Purpose
Exploratory analysis of ROADWork dataset, focus on Temporary Traffic Control (TTC) signs and speed limit extraction via CLIP.

### Key Imports
- `pathlib.Path`, `json`, `collections`
- `matplotlib.pyplot`, `PIL.Image`
- `torch`, `open_clip`
- `regex` (re)
- `dataclasses`, `tqdm`

### Main Functionality

#### Dataset Loading
```python
Train annotations: instances_train_gps_split_with_signs.json
Val annotations: instances_val_gps_split_with_signs.json
Images: data/images/
```

#### TTC Category Definition
```python
TTC_CAT_IDS = [16, 17, 19-50]  # 35+ categories
Examples:
- Category 16: Temporary Traffic Control Message Board
- Category 17: Temporary Traffic Control Sign
- Categories 19-50: Specific arrow types, chevrons, pedestrian signs, etc.
```

#### Data Structures

1. **`@dataclass TTCSample`**
   ```python
   image_id: int
   file_name: str
   category_id: int
   category_name: str
   bbox: list [x, y, w, h]
   sign_text: str          # OCR'd text or annotation
   text_occluded: bool
   ```

2. **`crop_ttc_sign(sample, pad=4) -> PIL.Image`**
   - Crops ROI around TTC sign with padding

#### Key Analyses

1. **TTC Sign Text Statistics**
   - Count unique sign text values
   - Most common: ~20 variants
   - Many with empty text

2. **CLIP Evaluation on TTC Signs**
   - 50 most frequent text classes
   - CLIP-ViT-B-16 trained on OpenAI CLIP weights
   - Accuracy on 300 samples: ~X% (notebook dependent)

3. **Speed Limit Extraction**
   - Regex: `r"SPEED LIMIT\s+(\d+)"`
   - Extracts speed values (25, 35, 45 mph, etc.)
   - Creates prompts: "temp work zone speed limit sign showing X mph"
   - CLIP evaluates on extracted speeds

#### Outputs
- Visualizations of TTC signs (12-sample grid)
- CLIP accuracy metrics on text classification
- Speed limit distribution analysis
- Accuracy on speed limit classification

#### Dependencies
- `torch`, `open_clip`
- `matplotlib`, `PIL`
- `re` (regex)
- `dataclasses`

---

## 8. **nvidia_vla_alpamayo_smoke_test.ipynb**

### Purpose
Real-time video processing with Alpamayo-R1 (NVIDIA Vision Language Action model), outputting chain-of-thought reasoning overlaid on video using threaded I/O.

### Key Imports
- `os`, `sys`, `time`, `textwrap`, `queue`, `threading`
- `pathlib.Path`
- `cv2`, `numpy`
- `torch`
- `ipywidgets`
- `IPython.display`
- Custom: `alpamayo_r1` modules

### Main Functionality

#### Model
```python
MODEL_ID = "nvidia/Alpamayo-R1-10B"
DEVICE = "cuda" (default)
DTYPE = torch.bfloat16 (if CUDA), torch.float16 (CPU)
```

#### Threaded Architecture

1. **`ThreadedVideoReader(path, queue_obj, target_size, stride=1)`**
   - Reads frames in background thread
   - Resizes to target size
   - Converts BGR → RGB
   - Puts frames into queue
   - Supports frame striding

2. **`ThreadedVideoWriter(path, queue_obj, fps, width, height)`**
   - Writes frames in background thread
   - Receives from queue
   - Handles full-resolution output

#### Key Functions

1. **`clean_and_wrap_text(raw_text, width=60) -> (lines, text)`**
   - Cleans model output (removes tokens, decodes)
   - Wraps to specified width
   - Returns as list of lines + clean text

2. **`run_live_notebook(video_path, model, processor, tmpl, out_path, stride=3)`**
   - Orchestrates reading, inference, display, writing
   - Uses ipywidgets for live display in Jupyter
   - Displays FPS label and image widget

3. **`run_live_notebook_smooth(...)`** (improved version)
   - Throttles display updates (max 10 FPS to notebook)
   - Always saves full resolution to disk (full FPS)
   - Prevents network congestion in Jupyter

#### Inference Pipeline
```python
1. Buffer frames until BATCH_SIZE reached
2. Stack frames into tensor [1, T1*T2, 3, H, W]
3. Create message with image + instruction
4. Apply chat template + tokenize
5. Model forward: sample_trajectories_from_data_with_vlm_rollout()
6. Extract chain-of-thought reasoning (raw_cot)
7. Overlay on frames
8. Write to disk + display in notebook
```

#### Template Data Structure
```python
tmpl = {
    "image_frames": shape (T1, T2, C, H, W),
    "ego_history_xyz": shape (...),
    "ego_history_rot": shape (...),
}
```

#### Visual Output
```python
Banner (black rect):
- Title: "ALPAYMAO REASONING:" (yellow)
- FPS (red, top-right)
- Wrapped text lines (white, multi-line)

Overlay frame with:
- Drawing of reasoning
- Both to disk (full resolution)
- Resized to 640px width for notebook display
```

#### Outputs
- Video file: `jacksonville_live_notebook.mp4` (full resolution)
- Live Jupyter widget with streaming preview

#### Dependencies
- `torch`, `cv2`, `numpy`
- `alpamayo_r1` (custom model module)
- `ipywidgets`, `IPython`
- `queue`, `threading`

---

## Summary Table

| Notebook | Purpose | Key Model/Library | Input | Output | GPU |
|----------|---------|-------------------|-------|--------|-----|
| 01 | COCO → YOLO conversion | - | ROADWork JSON | YOLO dataset | No |
| 02 | Training & evaluation | YOLO (Ultralytics) | YOLO dataset | best.pt, metrics | Yes |
| 03 | Simple video demo | YOLO | Video, best.pt | Annotated video | Yes |
| 04 | State machine pipeline | YOLO + heuristics | Video, best.pt | Timeline CSV | Yes |
| 05 | Multi-GPU batch + tuning | YOLO + hysteresis | Videos, best.pt | Summary CSV | Yes (1-2) |
| 06 | CLIP semantic fusion | YOLO + CLIP | Video, timelines | Fused timeline | Yes |
| 07 | TTC analysis | CLIP (ViT-B-16) | ROADWork images | Stats, accuracy | Yes |
| 08 | Live inference + display | Alpamayo-R1 VLA | Video, model | Annotated video | Yes |

---

## Architecture Recommendations for Modularization

### Core Modules to Create

1. **`yolo_pipeline.py`**
   - YOLO model loading
   - Frame-by-frame inference
   - Class grouping utilities
   - Feature extraction

2. **`workzone_scorer.py`**
   - Frame feature → raw score
   - Logistic scoring with weights
   - Configuration management

3. **`state_machine.py`**
   - EMA smoothing
   - Hysteresis logic
   - State transitions

4. **`clip_verifier.py`**
   - CLIP model management
   - Prompt engineering
   - Frame selection logic
   - Score fusion

5. **`video_processor.py`**
   - Video I/O (read/write)
   - Frame batching
   - Progress tracking

6. **`dataset_converter.py`**
   - COCO → YOLO conversion
   - Annotation filtering
   - YAML generation

7. **`dataset_analysis.py`**
   - TTC sign exploration
   - CLIP evaluation on TTC
   - Speed limit extraction

8. **`vla_inference.py`**
   - Alpamayo model management
   - Threaded video processing
   - Visualization overlay

---

## Data Formats

### COCO Annotation Format
```json
{
  "images": [{"id": 1, "file_name": "...", "width": 1920, "height": 1080}],
  "annotations": [{"image_id": 1, "category_id": 3, "bbox": [x, y, w, h], "area": 1000}],
  "categories": [{"id": 3, "name": "Cone"}]
}
```

### YOLO Label Format
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
# e.g., 2 0.5 0.5 0.3 0.2
```

### Timeline CSV Format
```csv
frame,time_sec,score_raw,score_ema,is_workzone,channel,workers,vehicles,ttc,msg,near_proxy,total
0,0.0,0.1,0.1,0,...
```

### Fused Timeline CSV Format
```csv
frame,time_sec,score_ema,score_fused,is_workzone,is_workzone_fused,workzone_sem_full,workzone_sem_roi
0,0.0,0.1,0.12,0,0,0.3,NaN
```

---

## Model Checkpoints & Weights

| Model | Source | Size | Use Case |
|-------|--------|------|----------|
| yolo12s.pt | Ultralytics | ~25MB | Main detector (recommended) |
| yolov8n.pt | Ultralytics | ~6MB | Fast inference |
| yolov8s.pt | Ultralytics | ~22MB | Small model |
| ViT-B-32 | OpenAI CLIP | ~349MB | Semantic verification |
| ViT-B-16 | OpenAI CLIP | ~560MB | Sign text analysis |
| Alpamayo-R1-10B | NVIDIA | ~10B params | VLA chain-of-thought |

---

## Key Configuration Parameters (Tuning Guide)

### YOLO Training
```python
imgsz = 640          # image size (adjust for GPU memory)
batch = 4            # batch size
epochs = 100         # training epochs
conf = 0.25-0.4      # confidence threshold
iou = 0.5-0.7        # NMS IoU threshold
```

### Work Zone Scoring
```python
# Notebook 04
w_channel, w_workers, w_vehicles, w_ttc, w_msg: [0.5, 2.5]
bias: [-5.0, 0.0]
ema_alpha: [0.05, 0.5]
enter_th, exit_th: [0.4, 0.8]
min_enter_sec, min_exit_sec: [0.5, 3.0]
cooldown_after_enter_sec: [0.5, 3.0]

# Notebook 05 (Enhanced)
w_bottom, w_near: [0.2, 1.0]
k_enter, k_exit: [3, 15]  # consecutive frames
```

### CLIP Fusion (Notebook 06)
```python
fused_weight_ema = 0.75
fused_weight_semantic = 0.25
enter_th = 0.55
exit_th = 0.45
```

### VLM (Notebook 08)
```python
stride = 3           # process every N frames
imgsz = 960          # input image size
temperature = 0.6
top_p = 0.8
max_generation_length = 256
```

---

## Performance Notes

- **Notebook 02:** Training time varies; use mixed precision (autocast)
- **Notebook 03:** ~30-60 FPS inference on single GPU (depends on imgsz)
- **Notebook 04:** Stride=3 reduces processing time 3x; adjust for real-time
- **Notebook 05:** Multi-GPU recommended for batch processing (50+ videos)
- **Notebook 06:** CLIP inference slower (~5-10 FPS per frame); selective triggering essential
- **Notebook 08:** VLA inference slow (~2-5 FPS); threaded I/O critical for responsiveness

