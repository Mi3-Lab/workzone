# WorkZone Project Index

Complete navigation guide for the WorkZone AI construction zone detection system.

## 📚 Documentation

### Getting Started
- **[README.md](../README.md)** - Main project overview and features
- **[Quick Start Guide](guides/QUICKSTART.md)** - Get running in 5 minutes
- **[Model Registry](MODEL_REGISTRY.md)** - Performance metrics and model selection

### Technical Reports
- **[Phase 1.2 Mining Report](reports/PHASE1_2_MINING_REPORT.md)** - Hard-negative mining process and results
- **[Phase 1.2 Completion Report](reports/PHASE1_2_COMPLETION_REPORT.txt)** - Full completion summary
- **[Model Update Summary](reports/MODEL_UPDATE_SUMMARY.md)** - Integration of hard-neg trained model
- **[Hard-Negatives Summary](reports/HARD_NEGATIVES_SUMMARY.md)** - Hard-negative categorization details
- **[Results Index](reports/RESULTS_INDEX.md)** - Training results and metrics

## 🛠️ Key Scripts

### Video Processing
- **[process_video_fusion.py](../scripts/process_video_fusion.py)** - Main video processing with YOLO + CLIP + Phase 1.1
  - CLI tool for batch video processing
  - Outputs: annotated video + CSV timeline
  - Features: state machine, adaptive EMA, context boost

### Hard-Negative Mining
- **[mine_hard_negatives.py](../scripts/mine_hard_negatives.py)** - Extract false positive candidates from videos
- **[review_hard_negatives.py](../scripts/review_hard_negatives.py)** - Interactive review and categorization tool
- **[consolidate_candidates.py](../scripts/consolidate_candidates.py)** - Merge candidates from multiple mining runs
- **[sample_candidates.py](../scripts/sample_candidates.py)** - Analyze candidate distributions

### Utilities
- **[batch_mine_hard_negatives.py](../scripts/batch_mine_hard_negatives.py)** - Parallel mining across GPUs
- **[HARDNEG_QUICKSTART.sh](../scripts/HARDNEG_QUICKSTART.sh)** - Quickstart script for hard-negative workflow

## 🎨 Applications

### Streamlit Web Interfaces
- **[app_phase1_1_fusion.py](../src/workzone/apps/streamlit/app_phase1_1_fusion.py)** ⭐ - Complete pipeline (recommended)
  - Model selection (Hard-Neg, Baseline, Custom)
  - Real-time playback and batch processing
  - Phase 1.1 visualization
  
- **[app_basic_detection.py](../src/workzone/apps/streamlit/app_basic_detection.py)** - Basic YOLO detection
- **[app_advanced_scoring.py](../src/workzone/apps/streamlit/app_advanced_scoring.py)** - Advanced scoring with EMA

### Command-Line Interfaces
- **[src/workzone/cli/](../src/workzone/cli/)** - CLI entry points for training and inference

## 🧩 Source Code

### Core Modules
- **[src/workzone/detection/](../src/workzone/detection/)** - Cue classification and detection logic
  - `CueClassifier` - Maps YOLO classes to cue groups
  
- **[src/workzone/temporal/](../src/workzone/temporal/)** - Temporal persistence tracking
  - `PersistenceTracker` - Sliding window cue persistence
  
- **[src/workzone/fusion/](../src/workzone/fusion/)** - Multi-cue fusion logic
  - `MultiCueGate` - AND logic for sustained cues
  
- **[src/workzone/models/](../src/workzone/models/)** - Model wrappers (YOLO, CLIP)
- **[src/workzone/pipelines/](../src/workzone/pipelines/)** - Training and inference pipelines
- **[src/workzone/utils/](../src/workzone/utils/)** - Utilities (logging, paths, config)

### Applications
- **[src/workzone/apps/streamlit/](../src/workzone/apps/streamlit/)** - Streamlit web applications
- **[src/workzone/apps/streamlit_utils.py](../src/workzone/apps/streamlit_utils.py)** - Shared utilities for Streamlit apps

## ⚙️ Configuration

### YAML Configs
- **[configs/config.yaml](../configs/config.yaml)** - Main project configuration
  - YOLO parameters (model, imgsz, batch, device)
  - Processing parameters (fps, threading)
  - Data paths
  - W&B integration
  
- **[configs/multi_cue_config.yaml](../configs/multi_cue_config.yaml)** - Phase 1.1 configuration
  - Cue group definitions (50 classes → 5 groups)
  - Detection thresholds per group
  - Temporal persistence parameters
  - Multi-cue gate logic
  - State machine transitions

### Setup Files
- **[requirements.txt](../requirements.txt)** - Python dependencies
- **[pyproject.toml](../pyproject.toml)** - Project metadata and build config
- **[setup.sh](../setup.sh)** - Environment setup script
- **[Makefile](../Makefile)** - Common development tasks

## 📊 Data

### Dataset Structure
- **[data/01_raw/](../data/01_raw/)** - Raw annotations, images, trajectories
- **[data/02_processed/](../data/02_processed/)** - Processed data + hard negatives
  - `hard_negatives/` - Approved hard negatives with manifest.csv
  
- **[data/03_demo/](../data/03_demo/)** - Demo videos for quick testing
- **[data/04_derivatives/](../data/04_derivatives/)** - Generated data
- **[data/05_workzone_yolo/](../data/05_workzone_yolo/)** - YOLO format dataset
  - `workzone_yolo.yaml` - Dataset config
  - `images/train/` - Training images
  - `images/val/` - Validation images
  - `labels/train/` - Training labels
  - `labels/val/` - Validation labels

### Outputs
- **[outputs/](../outputs/)** - Generated outputs
  - `demo_hardneg_test/` - Demo processing results
  - `hardneg_mining/` - Hard-negative mining candidates
  - `notebook*/` - Notebook outputs
  - Phase 1.1 test results

### Training Results
- **[runs/train/](../runs/train/)** - YOLO training runs
  - Each run folder contains: weights/, results.csv, plots
- **[weights/](../weights/)** - Model checkpoints
  - `yolo12s_hardneg_1280.pt` ⭐ - Production model
  - `yolo12s_fusion_baseline.pt` - Baseline model
  - Other models (yolo11n, yolo8s, etc.)

## 🧪 Testing

- **[tests/](../tests/)** - Unit tests
  - `test_config.py` - Configuration tests
  - `test_models.py` - Model wrapper tests
  - `test_pipelines.py` - Pipeline tests
  - `conftest.py` - Pytest fixtures

## 📓 Notebooks

- **[notebooks/](../notebooks/)** - Jupyter notebooks for experiments
  - `01_workzone_yolo_setup.ipynb` - Dataset setup
  - `02_workzone_yolo_train_eval.ipynb` - Training and evaluation
  - `03_workzone_yolo_video_demo.ipynb` - Video inference demo
  - `04_workzone_video_state_machine.ipynb` - State machine development
  - `05_workzone_video_timeline_calibration.ipynb` - Timeline calibration
  - `06_triggered_vlm_semantic_verification.ipynb` - VLM experiments
  - `nvidia_vla_alpamayo_smoke_test.ipynb` - Alpamayo integration

## 🔗 Quick Links

### Common Tasks
- **Train model**: `yolo train data=data/05_workzone_yolo/workzone_yolo.yaml model=yolo12s.pt epochs=100`
- **Process video**: `python scripts/process_video_fusion.py VIDEO_PATH --enable-phase1-1`
- **Launch web UI**: `streamlit run src/workzone/apps/streamlit/app_phase1_1_fusion.py`
- **Mine hard negatives**: `python scripts/mine_hard_negatives.py --video-dir data/videos_compressed`
- **Run tests**: `pytest tests/`

### Key Metrics
- **False Positive Reduction**: 84.6% (Phase 1.2)
- **Production Model**: yolo12s_hardneg_1280.pt (19 MB)
- **Inference Speed**: ~85 FPS (A100), ~30 FPS (Jetson Orin est.)
- **Dataset**: 50 classes, 134 hard negatives

### External Resources
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **OpenCLIP**: https://github.com/mlfoundations/open_clip
- **Streamlit**: https://streamlit.io/
- **W&B**: https://wandb.ai/

---

**Last Updated**: January 3, 2026
