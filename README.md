# WorkZone: AI-Powered Construction Zone Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Real-time construction zone detection using YOLO with semantic verification via CLIP. Built for the ESV (Enhanced Safety of Vehicles) competition and optimized for deployment on edge devices (Jetson Orin).

## ⚡ Quick Features

- 🚗 **Real-time YOLO detection** - 50-class construction zone object detection
- 🎨 **Lightweight fusion & smoothing** - Adaptive EMA + orange-cue context boost (no heavy models)
- 📊 **State machine tracking** - Anti-flicker work zone states (OUT → APPROACHING → INSIDE → EXITING)
- 🎬 **Interactive apps** - Streamlit web UIs with live preview and batch processing
- 💻 **Edge-ready** - Optimized for Jetson Orin; FP16 inference, configurable stride
- 📈 **Experiment tracking** - Weights & Biases integration
- 🔧 **Professional codebase** - PEP 8, type hints, comprehensive logging

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/WMaia9/workzone.git
cd workzone

# System dependencies (Linux)
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 ffmpeg

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install (development mode)
pip install -e ".[dev]"
```

## 📖 Usage

### Training YOLO

```bash
python -m src.workzone.cli.train_yolo \
  --model yolo12s.pt \
  --epochs 300 \
  --batch 32 \
  --device 0
```

### Video Inference

```bash
python -m src.workzone.cli.infer_video \
  --video data/demo/sample.mp4 \
  --model weights/best.pt \
  --output result.mp4
```

### Python API

```python
from src.workzone.models.yolo_detector import YOLODetector
from src.workzone.pipelines.video_inference import VideoInferencePipeline
from pathlib import Path

detector = YOLODetector("weights/best.pt", device="cuda")
pipeline = VideoInferencePipeline(detector, Path("video.mp4"), Path("output.mp4"))
results = pipeline.process()
```

## 🎨 Web Applications

Three production-ready Streamlit apps for interactive analysis:

```bash
# Basic YOLO detection with simple scoring
streamlit run src/workzone/apps/streamlit/app_basic_detection.py

# Advanced semantic scoring (z-scores, EMA smoothing)
streamlit run src/workzone/apps/streamlit/app_advanced_scoring.py

# YOLO + CLIP fusion with adaptive EMA and context boost (RECOMMENDED)
streamlit run src/workzone/apps/streamlit/app_semantic_fusion.py
```

**Features**:
- **Live preview** (real-time playback with instant feedback)
- **Batch processing** (save outputs: video + CSV timeline)
- **Score visualization** (YOLO and fused score curves)
- **Device selection** (Auto-detect GPU, manual CPU/CUDA)
- **Fusion boosts**:
  - ✅ Adaptive EMA smoothing (scales from 0.4× to 1.2× base alpha based on evidence)
  - ✅ Orange-cue context boost (HSV-based work zone color detection when YOLO is uncertain)
  - ✅ Tunable HSV thresholds for traffic cone/barrier colors
- **CSV export** (frame-by-frame scores, states, clip usage)
- **Model upload** (test with custom YOLO weights)

### Streamlit Cloud Deployment

- Place `packages.txt` (apt dependencies) alongside your app. This repo stores it at `src/workzone/apps/streamlit/packages.txt`.
- Example apt packages: `libgl1`, `libglib2.0-0`, `ffmpeg`.

### Vision-Language Apps (Optional)

```bash
# 10Hz VLA reasoning with overlay
python src/workzone/apps/alpamayo/alpamayo_10hz_inspector.py \
  --video data/demo/sample.mp4 --output output.mp4

# Zero-lag threaded VLA player
python src/workzone/apps/alpamayo/alpamayo_threaded.py \
  --video data/demo/sample.mp4
```

## 📁 Project Structure

```
src/workzone/
├── models/              # YOLO, CLIP, Alpamayo wrappers
├── pipelines/          # Training and inference pipelines
├── cli/                # Command-line interfaces
├── apps/              # Streamlit and Alpamayo apps
│   ├── streamlit/     # Web UIs
│   └── alpamayo/      # VLA apps
├── config.py          # Configuration management
└── utils/             # Logging, path utilities
```

## ⚙️ Configuration

YAML-based configuration:

```bash
# Edit configs/config.yaml
yolo:
  model_name: yolo12s
  imgsz: 960
  batch_size: 32
  device: cuda:0
```

Or use environment variables:

```bash
export WORKZONE_YOLO_MODEL=yolo12s
export WORKZONE_YOLO_DEVICE=cuda:0
```

Or Python API:

```python
from src.workzone.config import ProjectConfig, YOLOConfig

config = ProjectConfig(
    device="cuda",
    yolo=YOLOConfig(model_name="yolo12s", imgsz=960, epochs=300)
)
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=src/workzone

# Format code
black src/ tests/
isort src/ tests/

# Lint & type check
flake8 src/ tests/
mypy src/
```

## 📊 Performance

| Model | Size | FPS (batch=1) | GPU Memory | Notes |
|-------|------|---------------|-----------|-------|
| YOLOv12s | 960×960 | ~85 | 2.4GB | RTX 4090, FP16 |
| YOLOv12s | 960×960 (stride=2) | ~150 | 2.4GB | Real-time on Orin estimate |
| YOLOv8s | 640×640 (batch=32) | ~240 | 12GB | Batch mode, A100 |

**Latency (single frame, GPU)**:
- YOLO inference: ~12ms (960×960, FP16)
- Orange cue: <1ms
- CLIP (optional, triggered): ~30ms
- Total: 12–42ms depending on trigger

**Fusion model**: Adaptive EMA + lightweight context boost (no additional latency vs. base YOLO)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

**Code guidelines**:
- PEP 8 compliance with black formatter
- Type hints on all functions
- Docstrings for classes/methods
- Tests for new functionality

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8/v11/v12
- **OpenAI** - CLIP model for semantic verification
- **Weights & Biases** - Experiment tracking
- **ESV Competition** - Competition organizers
- **Streamlit** - Interactive web applications

---

**Built for the ESV Competition** 🏆 | **Edge-ready for Jetson Orin** 🚀
