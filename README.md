# 🚧 WorkZone Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv12](https://img.shields.io/badge/YOLO-v12-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Real-time construction work zone detection and monitoring system using state-of-the-art computer vision.**

Built for ESV (Enhanced Safety of Vehicles) competition. Features multi-modal verification (YOLO + CLIP + OCR), temporal attention, scene context classification, and edge deployment optimization for NVIDIA Jetson Orin.

---

## ⚡ Key Features

| Feature | Description |
|---------|-------------|
| **🎯 YOLO12s Detection** | 50-class object detection with 84.6% false positive reduction |
| **🧠 Multi-Modal Fusion** | CLIP semantic verification + OCR text extraction |
| **📊 Temporal Attention** | Phase 2.1: Per-cue confidence tracking + motion plausibility |
| **🌍 Scene Context** | Highway/Urban/Suburban classification (92.8% accuracy) |
| **🔄 Adaptive State Machine** | Context-aware thresholds: OUT → APPROACHING → INSIDE → EXITING |
| **⚡ Edge Optimized** | Runs 30 FPS @ 1280px on Jetson Orin |
| **🎬 Interactive UI** | Streamlit calibration app with real-time visualization |

---

## 🎯 Performance Highlights

| Component | Metric | Value |
|-----------|--------|-------|
| **YOLO Detection** | False Positive Reduction | **84.6%** vs baseline |
| **YOLO Detection** | Inference Speed (A100) | **~85 FPS** @ 1280px |
| **YOLO Detection** | Inference Speed (Jetson) | **~30 FPS** @ 1280px |
| **Scene Context** | Classification Accuracy | **92.8%** |
| **OCR Classification** | Test Accuracy | **97.7%** (43/44) |
| **System** | GPU Memory (batch=1) | **2.4 GB** |

---

## 📦 Installation

### Prerequisites

- **Python**: 3.10 or 3.11 (3.12 not tested)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended) or CPU
- **RAM**: 16GB minimum, 32GB recommended
- **Disk**: ~10GB for models + data

### Step 1: Clone Repository

```bash
git clone https://github.com/WMaia9/workzone.git
cd workzone
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python3.11 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install workzone package
pip install -e .
```

**Note**: For **CPU-only** installation, install PyTorch CPU version first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

### Step 4: Download Pre-trained Models

```bash
# Download all required models (~3GB)
bash scripts/download_models.sh
```

This downloads:
- ✅ `yolo12s_hardneg_1280.pt` - Hard-negative trained YOLO (recommended)
- ✅ `yolo12s_fusion_baseline.pt` - Baseline YOLO model
- ✅ `scene_context_classifier.pt` - Phase 1.4 scene context model
- ✅ CLIP ViT-B/32 (auto-downloaded on first run)

### Step 5: Verify Installation

```bash
# Quick test - process demo video
python scripts/process_video_fusion.py \
  data/demo/boston_workzone_short.mp4 \
  --output-dir outputs/test \
  --stride 5

# Expected: Annotated video + CSV timeline in outputs/test/
```

---

## 🚀 Quick Start

### Option 1: Interactive Calibration App (Recommended)

Launch the **Streamlit calibration UI** for interactive parameter tuning:

```bash
streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py
```

**Features**:
- 📹 Real-time video preview with live parameter adjustment
- 📊 Batch processing with explainability dashboards
- 💾 Export annotated videos + detailed CSV timelines
- 🎚️ Calibrate YOLO weights, CLIP fusion, OCR boost, state machine
- 🔬 Phase 2.1: Per-cue confidences + motion plausibility visualizations

👉 **See [APP_TESTING_GUIDE.md](APP_TESTING_GUIDE.md)** for detailed usage instructions.

---

### Option 2: Command-Line Batch Processing

Process videos from the command line for high-throughput workflows:

#### Basic Usage

```bash
python scripts/process_video_fusion.py \
  path/to/video.mp4 \
  --output-dir outputs/my_run
```

#### Phase 1.1: Multi-Cue Temporal Persistence

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/phase1_1 \
  --enable-phase1-1 \
  --no-motion
```

#### Phase 1.4: Scene Context Classification

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/phase1_4 \
  --enable-phase1-4 \
  --enable-ocr
```

#### Phase 2.1: Per-Cue Verification + Motion Tracking

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/phase2_1 \
  --enable-phase2-1 \
  --enable-phase1-1 \
  --enable-ocr \
  --no-motion \
  --stride 2
```

#### Full Pipeline (All Features)

```bash
python scripts/process_video_fusion.py \
  data/demo/video.mp4 \
  --output-dir outputs/full \
  --enable-phase1-1 \
  --enable-phase1-4 \
  --enable-phase2-1 \
  --enable-ocr \
  --device cuda \
  --stride 2 \
  --clip-weight 0.35 \
  --clip-trigger-th 0.45 \
  --enter-th 0.70 \
  --exit-th 0.45
```

#### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device: `cuda` or `cpu` | `cuda` |
| `--conf` | YOLO confidence threshold | `0.25` |
| `--stride` | Frame stride (1 = every frame) | `2` |
| `--enable-ocr` | Enable OCR text extraction | `False` |
| `--enable-phase1-1` | Multi-cue temporal logic | `False` |
| `--enable-phase1-4` | Scene context classification | `False` |
| `--enable-phase2-1` | Per-cue CLIP + motion tracking | `False` |
| `--clip-weight` | CLIP fusion weight | `0.35` |
| `--clip-trigger-th` | CLIP trigger threshold | `0.45` |
| `--enter-th` | WORKZONE entry threshold | `0.70` |
| `--exit-th` | WORKZONE exit threshold | `0.45` |
| `--no-video` | Skip video output (faster) | `False` |
| `--no-csv` | Skip CSV output | `False` |

---

## 📊 System Architecture

### Detection Pipeline

```
┌─────────────────┐
│  Input Video    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  YOLO12s Object Detection                       │
│  • 50 work zone classes                         │
│  • Hard-negative trained (84.6% FP reduction)   │
│  • 1280px @ 30 FPS (Jetson Orin)               │
└────────┬────────────────────────────────────────┘
         │
         ├──────────────────┬──────────────────┬─────────────────┐
         ▼                  ▼                  ▼                 ▼
┌────────────────┐  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐
│ CLIP Semantic  │  │ OCR Text        │  │ Scene Context│  │ Per-Cue CLIP    │
│ Verification   │  │ Extraction      │  │ Classifier   │  │ (Phase 2.1)     │
│ (Global)       │  │ (Message Boards)│  │ (Phase 1.4)  │  │ • Channelization│
└────────┬───────┘  └────────┬────────┘  └──────┬───────┘  │ • Workers       │
         │                   │                   │          │ • Vehicles      │
         │                   │                   │          │ • Signs         │
         │                   │                   │          │ • Equipment     │
         │                   │                   │          └────────┬────────┘
         │                   │                   │                   │
         └───────────────────┴───────────────────┴───────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  Multi-Modal Fusion   │
                          │  • Weighted EMA       │
                          │  • Context Boost      │
                          │  • OCR Boost          │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  State Machine        │
                          │  OUT → APPROACHING    │
                          │      → INSIDE         │
                          │      → EXITING → OUT  │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  Output               │
                          │  • Annotated Video    │
                          │  • Timeline CSV       │
                          │  • State Transitions  │
                          └───────────────────────┘
```

### Phase Progression

| Phase | Feature | Description |
|-------|---------|-------------|
| **1.0** | Base System | YOLO + CLIP + EMA + State Machine |
| **1.1** | Multi-Cue Logic | Temporal persistence tracking (5 cue types) |
| **1.2** | Hard-Negative Mining | 84.6% FP reduction through iterative training |
| **1.3** | Motion Validation | Trajectory-based false positive filtering |
| **1.4** | Scene Context | Highway/Urban/Suburban classification (92.8%) |
| **2.1** | Per-Cue Verification | CLIP confidence per cue + motion plausibility |

---

## 📂 Repository Structure

```
workzone/
├── README.md                          # This file
├── APP_TESTING_GUIDE.md              # Comprehensive calibration guide
├── requirements.txt                   # Python dependencies
├── pyproject.toml                    # Package configuration
├── setup.py                          # Installation script
│
├── configs/                          # Configuration files
│   ├── config.yaml                   # Main config
│   ├── multi_cue_config.yaml         # Phase 1.1 multi-cue settings
│   └── motion_cue_config.yaml        # Phase 1.3 motion settings
│
├── data/                             # Data directory (gitignored)
│   ├── 01_raw/                       # Raw videos
│   ├── 02_processed/                 # Processed annotations
│   ├── 03_demo/                      # Demo videos
│   ├── 04_derivatives/               # Hard-negative mining outputs
│   └── 05_workzone_yolo/             # YOLO training data
│
├── weights/                          # Pre-trained models (download via script)
│   ├── yolo12s_hardneg_1280.pt      # Recommended model
│   ├── yolo12s_fusion_baseline.pt   # Baseline model
│   ├── scene_context_classifier.pt  # Phase 1.4 model
│   └── ...
│
├── scripts/                          # CLI tools
│   ├── process_video_fusion.py       # Main video processing CLI
│   ├── download_models.sh            # Model download script
│   ├── mine_hard_negatives.py        # Hard-negative mining
│   ├── train_scene_context.py        # Scene context training
│   └── evaluate_phase1_4.py          # Phase 1.4 evaluation
│
├── src/workzone/                     # Core package
│   ├── detection/                    # Detection components
│   │   ├── yolo_detector.py          # YOLO wrapper
│   │   └── cue_classifier.py         # Multi-cue classification
│   ├── fusion/                       # Multi-modal fusion
│   │   ├── clip_verifier.py          # CLIP semantic verification
│   │   └── multi_cue_gate.py         # Phase 1.1 AND gate
│   ├── ocr/                          # OCR text extraction
│   │   ├── text_detector.py          # EasyOCR/Paddle wrapper
│   │   └── text_classifier.py        # Text category classification
│   ├── models/                       # Advanced models
│   │   ├── scene_context.py          # Phase 1.4 scene classifier
│   │   ├── per_cue_verification.py   # Phase 2.1 per-cue CLIP
│   │   └── trajectory_tracking.py    # Phase 2.1 motion plausibility
│   ├── temporal/                     # Temporal logic
│   │   └── persistence_tracker.py    # Phase 1.1 persistence
│   ├── state/                        # State machine
│   │   └── workzone_states.py        # State transitions
│   └── apps/                         # Applications
│       └── streamlit/                # Streamlit UI
│           └── app_phase2_1_evaluation.py  # Calibration app
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_workzone_yolo_setup.ipynb
│   ├── 02_workzone_yolo_train_eval.ipynb
│   ├── 03_workzone_yolo_video_demo.ipynb
│   ├── 04_workzone_video_state_machine.ipynb
│   ├── 05_workzone_video_timeline_calibration.ipynb
│   ├── 06_triggered_vlm_semantic_verification.ipynb
│   └── 07_phase1_4_scene_context.ipynb
│
├── tests/                            # Unit tests
│   ├── test_config.py
│   ├── test_models.py
│   └── test_pipelines.py
│
├── docs/                             # Documentation
│   ├── MODEL_REGISTRY.md             # Model performance metrics
│   ├── PHASE1_3.md                   # Phase 1.3 motion validation
│   └── guides/                       # User guides
│
└── outputs/                          # Processing outputs (gitignored)
    ├── phase1_1_demo/
    ├── phase1_4_demo/
    ├── phase2_1_demo/
    └── ...
```

---

## 🔬 Advanced Usage

### Training Custom Models

#### YOLO Fine-tuning

```bash
cd workzone-yolo-v12/
yolo train \
  data=workzone.yaml \
  model=yolo12s.pt \
  epochs=50 \
  imgsz=1280 \
  batch=8 \
  device=0
```

#### Scene Context Training

```bash
python scripts/train_scene_context.py \
  --data-root data/05_workzone_yolo \
  --output-dir runs/scene_context \
  --epochs 30 \
  --batch-size 32 \
  --backbone resnet18
```

#### Phase 2.1 Temporal Attention Training

```bash
python scripts/train_phase2_1_attention.py \
  --data-path data/phase2_1_trajectories \
  --output-dir runs/phase2_1_attention \
  --epochs 50 \
  --batch-size 16 \
  --device cuda
```

### Hard-Negative Mining

See [docs/reports/PHASE1_2_MINING_REPORT.md](docs/reports/PHASE1_2_MINING_REPORT.md) for details.

```bash
# 1. Mine candidates from video dataset
bash scripts/HARDNEG_QUICKSTART.sh

# 2. Review and categorize
python scripts/review_hard_negatives.py

# 3. Consolidate annotations
python scripts/consolidate_candidates.py

# 4. Retrain YOLO
cd workzone-yolo-v12/
yolo train data=workzone_hardneg.yaml model=yolo12s.pt ...
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=workzone --cov-report=html

# Run specific test
pytest tests/test_models.py -v
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [APP_TESTING_GUIDE.md](APP_TESTING_GUIDE.md) | **Comprehensive calibration guide** with all parameters explained |
| [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md) | Model performance benchmarks |
| [docs/PHASE1_3.md](docs/PHASE1_3.md) | Motion validation details |
| [docs/reports/PHASE1_2_MINING_REPORT.md](docs/reports/PHASE1_2_MINING_REPORT.md) | Hard-negative mining methodology |

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](alpamayo/CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](alpamayo/LICENSE) for details.

---

## 🙏 Acknowledgments

- ESV Competition organizers
- Ultralytics for YOLOv12
- OpenAI for CLIP
- PaddleOCR and EasyOCR teams
- W&B for experiment tracking

---

## 📧 Contact

For questions or feedback:
- **GitHub Issues**: [github.com/WMaia9/workzone/issues](https://github.com/WMaia9/workzone/issues)
- **Email**: [your-email@domain.com]

---

**Built with ❤️ for safer roads**
