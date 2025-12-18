# WorkZone: AI-Powered Construction Zone Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Real-time construction zone detection using YOLO with semantic verification via CLIP and Vision-Language Models. Built for the ESV (Embodied Scene Understanding for Vehicles) competition.

## ⚡ Quick Features

- 🚗 **Real-time YOLO detection** - 50-class construction zone object detection
- 🧠 **Semantic verification** - CLIP + Alpamayo-R1 for contextual understanding
- 📊 **State machine tracking** - Anti-flicker work zone states (OUT → APPROACHING → INSIDE → EXITING)
- 🎬 **Interactive apps** - Streamlit web UIs for analysis and visualization
- 📈 **Experiment tracking** - Weights & Biases integration
- 🔧 **Professional codebase** - PEP 8, type hints, comprehensive logging

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/WMaia9/workzone.git
cd workzone

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

# YOLO + CLIP fusion with state machine
streamlit run src/workzone/apps/streamlit/app_semantic_fusion.py
```

**Features**: Live preview, batch processing, score visualization, CSV export, model upload

### Vision-Language Apps

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

| Model | Size | FPS (batch=1) | GPU Memory |
|-------|------|---------------|-----------|
| YOLOv12s | 960×960 | ~85 | 2.4GB |
| YOLOv12s | 960×960 (batch=8) | ~150 | 8.5GB |
| YOLOv8s | 640×640 (batch=32) | ~240 | 12GB |

**Latency (single frame, GPU)**: ~15ms total (12ms inference + 2ms preprocessing + 1ms postprocessing)

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
- **NVIDIA** - Alpamayo-R1 VLM
- **OpenAI** - CLIP model
- **Weights & Biases** - Experiment tracking
- **ESV Competition** - Competition organizers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/WMaia9/workzone/issues)
- **Email**: contact@workzone.dev

---

**Built for the ESV Competition** 🏆
