# Repository Structure

This document provides a professional overview of the WorkZone repository organization.

## 📁 Root Directory Structure

```
workzone/
├── src/workzone/           # Production source code
├── tests/                  # Test suites
├── scripts/                # Application entry points
├── notebooks/              # Jupyter notebooks for analysis
├── docs/                   # Documentation
├── configs/                # Configuration files
├── data/                   # Datasets (gitignored)
├── outputs/                # Processing results (gitignored)
├── weights/                # Model weights (gitignored)
└── README.md               # Main documentation
```

## 🔧 Production Code (`src/workzone/`)

**Purpose**: Production-ready Python package for construction zone detection.

```
src/workzone/
├── __init__.py
├── detection/              # YOLO detection and multi-modal fusion
│   ├── yolo_detector.py    # Core YOLOv12 detection
│   ├── clip_verifier.py    # CLIP semantic verification
│   └── fusion.py           # Score fusion logic
├── ocr/                    # OCR text extraction
│   ├── text_detector.py    # PaddleOCR detection
│   └── text_classifier.py  # Text semantic classification (97.7% accuracy)
├── state_machine/          # Temporal state tracking
│   └── workzone_tracker.py # State machine logic
├── models/                 # ML models
│   ├── scene_context.py    # Scene classification (Highway/Urban/Suburban)
│   ├── per_cue_verification.py  # Per-cue CLIP verification
│   └── trajectory_tracking.py   # Motion plausibility tracking
├── apps/                   # Applications
│   ├── streamlit/          # Web interfaces
│   └── cli/                # Command-line tools
└── utils/                  # Shared utilities
```

**Key Features**:
- YOLO12s object detection (50 classes)
- CLIP semantic verification (global + per-cue)
- PaddleOCR text extraction with 97.7% classification accuracy
- Scene context classification (92.8% accuracy)
- Per-cue confidence tracking (5 cue types)
- Motion plausibility from trajectory tracking
- Adaptive state machine with temporal persistence

## 🧪 Testing (`tests/`)

```
tests/
├── test_config.py          # Configuration tests
├── test_models.py          # Model loading tests
├── test_pipelines.py       # End-to-end pipeline tests
├── conftest.py             # PyTest configuration
└── exploratory/            # Development test scripts
    ├── README.md           # Exploratory tests documentation
    ├── test_ocr*.py        # OCR development tests
    ├── test_classifier_improved.py  # Classifier validation (97.7%)
    └── analyze_*.py        # Analysis scripts
```

**Testing Standards**:
- Production tests: `tests/test_*.py` (pytest suite)
- Exploratory tests: `tests/exploratory/` (development only)
- Coverage target: >80% for production code

## 📜 Scripts (`scripts/`)

**Purpose**: Application entry points for running the main applications.

```
scripts/
├── jetson_app.py                   # Main Jetson application
├── jetson_cli_app.py               # CLI application for Jetson
├── jetson_launcher.py              # GUI launcher for Jetson
├── jetson_launcher_sota.py         # Experimental GUI launcher
└── launch_streamlit.sh             # Script to launch the Streamlit app
```

## 🛠️ Tools (`tools/`)

**Purpose**: Automation, utility, and analysis scripts.

```
tools/
├── download_models.sh              # Download pre-trained weights
├── process_video_fusion.py         # Batch video processing
├── optimize_for_jetson.py          # Optimize models for Jetson
├── mine_hard_negatives.py          # Hard-negative mining
├── review_hard_negatives.py        # Human-in-the-loop review
└── analysis/                       # Analysis scripts
    └── ...
```

## 📊 Notebooks (`notebooks/`)

**Purpose**: Interactive analysis and experimentation.

```
notebooks/
├── 01_workzone_yolo_setup.ipynb        # YOLO setup and training
├── 02_workzone_yolo_train_eval.ipynb   # Training evaluation
├── 03_workzone_yolo_video_demo.ipynb   # Video inference demo
├── 04_workzone_video_state_machine.ipynb  # State machine testing
├── 05_workzone_video_timeline_calibration.ipynb  # Threshold tuning
├── 06_triggered_vlm_semantic_verification.ipynb  # CLIP integration
└── 07_phase1_4_scene_context.ipynb     # Scene context analysis
```

## 📚 Documentation (`docs/`)

```
docs/
├── README.md                       # Documentation index
├── REPOSITORY_STRUCTURE.md         # This guide
├── MODEL_REGISTRY.md               # Model registry
├── guides/
│   └── QUICKSTART.md               # Quick start
├── technical/                      # Technical docs
│   ├── OCR_IMPROVEMENTS.md         # OCR (97.7%)
│   └── OCR_REALTIME_STRATEGY.md    # Jetson deployment
├── phase1_4/                       # Phase 1.4
│   ├── PHASE1_4.md
│   └── PHASE1_4_SCENE_CONTEXT.md
├── reports/                        # Training reports
└── archive/                        # Historical docs
```

## ⚙️ Configuration (`configs/`)

```
configs/
├── config.yaml                         # Main system configuration
├── motion_cue_config.yaml              # Motion detection config
└── multi_cue_config.yaml               # Multi-cue fusion config
```

## 📁 Data (`data/`) - Gitignored

**Purpose**: Dataset storage (not tracked in git).

```
data/
├── 00_README.md                        # Dataset documentation
├── 00_DATASET_METADATA.json            # Metadata
├── DATA_ORGANIZATION_PLAN.md           # Organization guide
├── QUICKSTART.md                       # Data quickstart
├── 01_raw/                             # Raw annotations
├── 02_processed/                       # Processed datasets
├── 03_demo/                            # Demo videos
├── 04_derivatives/                     # Derived datasets
└── 05_workzone_yolo/                   # YOLO training data
```

## 📤 Outputs (`outputs/`) - Gitignored

**Purpose**: Processing results and artifacts.

```
outputs/
├── ocr_intensive_test_results.csv      # OCR test results (1,195 samples)
├── ocr_reprocessed_improved.csv            # Improved reprocessing results
├── phase1_1_integrated.csv             # Phase 1.1 results
├── phase1_3_demo/                      # Phase 1.3 demo outputs
├── phase1_4_complete_demo/             # Phase 1.4 demo outputs
├── phase1_4_evaluation/                # Phase 1.4 evaluation
├── hardneg_mining/                     # Hard-negative mining results
└── hardneg_preview/                    # Hard-negative preview images
```

## 🎯 Model Weights (`weights/`) - Gitignored

**Purpose**: Pre-trained model checkpoints.

```
weights/
├── bestv12.pt                          # YOLO12s baseline
├── yolo12s_fusion_baseline.pt          # Fusion baseline
├── scene_context_classifier.pt         # Scene context model
└── .gitkeep                            # Placeholder
```

## 🔧 Configuration Files

```
workzone/
├── pyproject.toml                      # Poetry dependencies
├── requirements.txt                    # Pip dependencies
├── setup.sh                            # Environment setup script
├── Makefile                            # Build automation
├── .gitignore                          # Git ignore rules
└── APP_TESTING_GUIDE.md                # Application testing guide
```

## 📋 Development Workflow

### 1. Adding New Features

```bash
# 1. Create feature in src/workzone/
# 2. Add tests in tests/
# 3. Document in docs/
# 4. Update README.md
```

### 2. Running Tests

```bash
# Production tests
pytest tests/

# Exploratory tests (optional)
python tests/exploratory/test_classifier_improved.py
```

### 3. Processing Videos

```bash
# CLI batch processing
python tools/process_video_fusion.py video.mp4 --output-dir outputs/

# Web interface
streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py
```

## 🚀 Deployment

### Jetson Orin Preparation

```bash
# Optimize the models for Jetson
python tools/optimize_for_jetson.py

# Convert to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

**Performance Targets**:
- YOLO: 15-20ms per frame
- OCR: 50-80ms per frame (1 Hz sampling)
- Scene Context: 5-10ms per frame
- **Total**: ~30 FPS real-time

## 📊 Key Metrics

| Component | Accuracy/Performance |
|-----------|---------------------|
| YOLO Detection | 84.6% FP reduction |
| Scene Context | 92.8% accuracy |
| OCR Classification | 97.7% test set accuracy |
| OCR Useful Rate | 39% (up from 26%) |
| System Throughput | 85 FPS (A100), 30 FPS (Jetson) |

## 📖 Additional Resources

- [Main README](../README.md) - Project overview
- [APP_TESTING_GUIDE.md](../APP_TESTING_GUIDE.md) - Testing guide
- [docs/README.md](README.md) - Documentation index
- [MODEL_REGISTRY.md](MODEL_REGISTRY.md) - Model performance
- [technical/OCR_IMPROVEMENTS.md](technical/OCR_IMPROVEMENTS.md) - OCR improvements report

## 🏆 Competition Ready

This repository is organized for professional presentation in the ESV competition:

✅ Clean code structure  
✅ Comprehensive documentation  
✅ Production-ready tests  
✅ Performance benchmarks  
✅ Deployment guides  
✅ Performance results documented  

**Status**: Ready for evaluation and deployment.
