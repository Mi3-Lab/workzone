# WorkZone: AI-Powered Construction Zone Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**WorkZone** is a professional AI system for detecting, tracking, and analyzing construction zones using advanced computer vision. It combines YOLO object detection with Vision-Language Models (VLMs) for semantic understanding, specifically developed for the ESV (Embodied Scene Understanding for Vehicles) competition.

## 🎯 Project Overview

### ESV Competition Context

This project is part of the **ESV (Embodied Scene Understanding for Vehicles)** competition, which focuses on enabling autonomous vehicles to understand and navigate complex construction zones safely. WorkZone provides:

- **Real-time construction zone detection** using state-of-the-art YOLO models
- **Semantic verification** using CLIP for reducing false positives
- **Vision-Language reasoning** with Nvidia's Alpamayo-R1 for contextual understanding
- **Multi-GPU support** for efficient processing of large video datasets
- **Experiment tracking** with Weights & Biases integration

### Key Features

✨ **Core Capabilities:**
- Frame-by-frame YOLO inference with high throughput
- Multi-class construction zone object detection (50 classes)
- Real-time video processing with optional output video generation
- Stateful detection with exponential moving average smoothing
- CLIP-based semantic verification for false positive reduction
- Vision-Language Model integration for scene reasoning
- Full W&B experiment tracking for reproducibility

🚀 **Professional Infrastructure:**
- PEP 8 compliant codebase with type hints
- Comprehensive logging and error handling
- CLI entry points for easy execution
- YAML-based configuration management
- pytest test framework
- pyproject.toml for modern Python packaging

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU inference, optional for CPU mode)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/WMaia9/workzone.git
cd workzone
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n workzone python=3.11
conda activate workzone
```

### Step 3: Install Dependencies

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Or for production use
pip install -e .
```

### Step 4: Download Model Weights (Optional)

```bash
mkdir -p weights
cd weights

# Download YOLO model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo12s.pt

cd ..
```

### Step 5: Verify Installation

```bash
# Test imports
python -c "from src.workzone.models.yolo_detector import YOLODetector; print('✅ Installation successful')"

# Check CLI commands
python -m src.workzone.cli.train_yolo --help
python -m src.workzone.cli.infer_video --help
```

## 🚀 Quick Start

### Training YOLO Model

```bash
# Train with default settings
python -m src.workzone.cli.train_yolo --device 0

# Train with custom parameters
python -m src.workzone.cli.train_yolo \
  --model yolo12s.pt \
  --epochs 300 \
  --batch 32 \
  --imgsz 960 \
  --device 0
```

### Running Video Inference

```bash
# Simple inference
python -m src.workzone.cli.infer_video \
  --video data/Construction_Data/sample_video.mp4 \
  --model weights/best.pt

# With output video and custom confidence
python -m src.workzone.cli.infer_video \
  --video data/Construction_Data/sample_video.mp4 \
  --model weights/best.pt \
  --output results/annotated_video.mp4 \
  --conf 0.5 \
  --skip-frames 1
```

### Python API Usage

```python
from pathlib import Path
from src.workzone.models.yolo_detector import YOLODetector
from src.workzone.pipelines.video_inference import VideoInferencePipeline

# Initialize detector
detector = YOLODetector(
    model_path="weights/best.pt",
    confidence_threshold=0.5,
    device="cuda"
)

# Create inference pipeline
pipeline = VideoInferencePipeline(
    detector=detector,
    video_path=Path("video.mp4"),
    output_path=Path("output.mp4")
)

# Process video
results = pipeline.process(draw_detections=True)
print(f"Processed {results['processed_frames']} frames")
```

## 📁 Project Structure

```
workzone/
├── src/workzone/                    # Main package
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   │
│   ├── models/                     # AI models
│   │   ├── yolo_detector.py       # YOLO detection wrapper
│   │   └── vlm.py                 # VLM integration (Alpamayo, CLIP)
│   │
│   ├── pipelines/                 # Processing pipelines
│   │   ├── yolo_training.py       # Training pipeline
│   │   └── video_inference.py     # Inference pipeline
│   │
│   ├── cli/                       # Command-line interfaces
│   │   ├── train_yolo.py         # Training CLI
│   │   └── infer_video.py        # Inference CLI
│   │
│   └── utils/                     # Utilities
│       ├── logging_config.py      # Logging setup
│       └── path_utils.py          # Path utilities
│
├── configs/                        # Configuration files
│   └── config.yaml                # Main configuration
│
├── data/                          # Dataset directory
│   └── Construction_Data/         # Video data
│
├── weights/                       # Model weights
│   ├── best.pt
│   └── yolo12s.pt
│
├── tests/                         # Unit and integration tests
│   ├── test_models.py
│   └── test_pipelines.py
│
├── notebooks/                     # Legacy Jupyter notebooks
│   ├── 01_workzone_yolo_setup.ipynb
│   ├── 02_workzone_yolo_train_eval.ipynb
│   └── ...
│
├── README.md                      # This file
├── pyproject.toml                # Project metadata and dependencies
├── requirements.txt               # pip requirements (legacy)
└── .gitignore
```

## 💻 Usage Guide

### Training

#### Using CLI

```bash
# Basic training
python -m src.workzone.cli.train_yolo \
  --model yolo12s.pt \
  --data data/workzone_yolo/workzone_yolo.yaml \
  --epochs 300 \
  --batch 35 \
  --device cuda:0
```

#### Using Python API

```python
from src.workzone.config import YOLOConfig
from src.workzone.pipelines.yolo_training import YOLOTrainingPipeline

config = YOLOConfig(
    model_name="yolo12s",
    imgsz=960,
    epochs=300,
    batch_size=35,
    device="cuda:0"
)

pipeline = YOLOTrainingPipeline(
    config=config,
    wandb_enabled=True,
    wandb_project="workzone-yolo"
)

results = pipeline.train(run_name="baseline_v1")
```

### Inference

#### Using CLI

```bash
# Process single video
python -m src.workzone.cli.infer_video \
  --video sample.mp4 \
  --model weights/best.pt \
  --output result.mp4 \
  --conf 0.5
```

#### Using Python API

```python
from pathlib import Path
from src.workzone.models.yolo_detector import YOLODetector
from src.workzone.pipelines.video_inference import VideoInferencePipeline

detector = YOLODetector("weights/best.pt")
pipeline = VideoInferencePipeline(
    detector=detector,
    video_path=Path("video.mp4"),
    output_path=Path("output.mp4")
)

results = pipeline.process()
print(f"Detections in {results['processed_frames']} frames")
```

### Web Applications

WorkZone includes production-ready Streamlit applications for interactive analysis:

#### Streamlit Apps

```bash
# Basic Detection (YOLO only)
streamlit run src/workzone/apps/streamlit/app_basic_detection.py

# Advanced Semantic Scoring (with statistical normalization)
streamlit run src/workzone/apps/streamlit/app_advanced_scoring.py

# YOLO + CLIP Fusion (with state machine)
streamlit run src/workzone/apps/streamlit/app_semantic_fusion.py
```

**Features:**
- 🎥 Live video preview with real-time detection
- 📊 Semantic scoring and statistical normalization
- 🔄 EMA smoothing for temporal consistency
- 🤖 CLIP-based semantic verification
- 📈 Timeline analysis with CSV export
- 🎨 State machine visualization (OUT/APPROACHING/INSIDE/EXITING)

#### Alpamayo VLA Apps

```bash
# 10Hz VLA Inspector
python src/workzone/apps/alpamayo/alpamayo_10hz_inspector.py \
  --video data/demo/sample.mp4 \
  --output output.mp4

# Zero-Lag Threaded Player
python src/workzone/apps/alpamayo/alpamayo_threaded.py \
  --video data/demo/sample.mp4
```

**Features:**
- 🧠 Vision-Language Model reasoning (Alpamayo-R1-10B)
- ⚡ Real-time asynchronous inference
- 🎬 Video annotation with reasoning overlay
- 🔄 Thread-based zero-lag processing

For detailed app documentation, see [APPS_GUIDE.md](APPS_GUIDE.md).

### Configuration

Configure WorkZone via:

1. **YAML file** (recommended for projects)
2. **Environment variables** (for CI/CD)
3. **Python API** (for programmatic use)

#### YAML Configuration

Edit `configs/config.yaml`:

```yaml
device:
  type: "cuda"
  id: 0

yolo:
  model_name: "yolo12s"
  imgsz: 960
  epochs: 300
  batch_size: 35
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

#### Environment Variables

```bash
export DEVICE="cuda"
export YOLO_MODEL="yolo12s"
export YOLO_IMGSZ="960"
export YOLO_CONF="0.5"
export YOLO_EPOCHS="300"
```

#### Python API

```python
from src.workzone.config import ProjectConfig, YOLOConfig

config = ProjectConfig(
    device="cuda",
    yolo=YOLOConfig(
        model_name="yolo12s",
        imgsz=960,
        epochs=300,
        confidence_threshold=0.5
    )
)
```

## 📚 API Documentation

### YOLODetector

```python
from src.workzone.models.yolo_detector import YOLODetector

detector = YOLODetector(
    model_path: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    device: str = "cuda"
)

# Detect objects in single frame
detections = detector.detect(frame: np.ndarray)
# Returns: {
#   "boxes": np.ndarray,       # (N, 4) bounding boxes
#   "confidences": np.ndarray, # (N,) confidence scores
#   "class_ids": np.ndarray,   # (N,) class IDs
#   "class_names": list        # (N,) class names
# }

# Batch detection
detections_list = detector.detect_batch(frames: list[np.ndarray])
```

### YOLOTrainingPipeline

```python
from src.workzone.pipelines.yolo_training import YOLOTrainingPipeline
from src.workzone.config import YOLOConfig

pipeline = YOLOTrainingPipeline(
    config: YOLOConfig,
    wandb_enabled: bool = True,
    wandb_project: str = "workzone-yolo"
)

# Load model
pipeline.load_model()

# Train
results = pipeline.train(run_name: str = "baseline", epochs: int = 300)

# Validate
val_results = pipeline.validate()

# Save
output_path = pipeline.save_model(Path("weights/model.pt"))
```

### VideoInferencePipeline

```python
from src.workzone.pipelines.video_inference import VideoInferencePipeline
from src.workzone.models.yolo_detector import YOLODetector
from pathlib import Path

pipeline = VideoInferencePipeline(
    detector: YOLODetector,
    video_path: Path,
    output_path: Path = None,
    skip_frames: int = 1
)

# Process video
results = pipeline.process(
    confidence_threshold: float = 0.5,
    draw_detections: bool = True
)
```

## 🧪 Development

### Setting Up Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/workzone tests/

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Building Documentation

```bash
# Generate API documentation
sphinx-build -b html docs/ docs/_build/
```

## 📊 Performance Benchmarks

### Hardware
- **GPU**: NVIDIA A100 (40GB)
- **CPU**: Intel Xeon (96 cores)
- **RAM**: 500GB

### Throughput

| Model | Input Size | Batch Size | FPS | GPU Memory |
|-------|-----------|-----------|-----|-----------|
| YOLOv12s | 960×960 | 1 | ~85 | 2.4GB |
| YOLOv12s | 960×960 | 8 | ~150 | 8.5GB |
| YOLOv8s | 640×640 | 32 | ~240 | 12GB |

### Latency (single frame, GPU)
- **Inference**: ~12ms
- **Preprocessing**: ~2ms
- **Postprocessing**: ~1ms
- **Total**: ~15ms per frame

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Code Guidelines

- Follow PEP 8 and use `black` for formatting
- Add type hints to all functions
- Include docstrings for classes and public methods
- Write tests for new functionality
- Update README for user-facing changes

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8/v11/v12 implementation
- **NVIDIA** for Alpamayo-R1 Vision-Language Model
- **OpenAI** for CLIP model
- **Weights & Biases** for experiment tracking
- **ESV Competition** organizers

## 📞 Contact & Support

For issues, questions, or suggestions:

- **GitHub Issues**: [Report a bug](https://github.com/WMaia9/workzone/issues)
- **Email**: contact@workzone.dev

---

**Made with ❤️ for the ESV Competition**
