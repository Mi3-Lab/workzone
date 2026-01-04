# WorkZone: AI-Powered Construction Zone Detection 🚧

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![YOLOv12](https://img.shields.io/badge/YOLO-v12-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

Real-time construction zone detection and monitoring system using state-of-the-art computer vision. Features YOLO object detection with multi-modal semantic verification (CLIP), temporal persistence tracking, and adaptive state machine logic. Built for the ESV (Enhanced Safety of Vehicles) competition and optimized for edge deployment on Jetson Orin.

**🎯 Key Achievement**: 84.6% false positive reduction through hard-negative mining and retraining.

## ⚡ Key Features

### Core Detection
- 🎯 **YOLO12s Object Detection** - 50-class construction zone detection with 84.6% FP reduction
- 🧠 **Multi-Modal Fusion** - CLIP semantic verification + orange-cue context boost
- 📊 **Phase 1.1 Multi-Cue Logic** - Temporal persistence tracking with AND logic
- 🔄 **Adaptive State Machine** - Anti-flicker states: OUT → APPROACHING → INSIDE → EXITING

### Hard-Negative Mining Pipeline
- 🔍 **Automated Mining** - Extract false positives from 400+ videos
- 👁️ **Interactive Review** - Human-in-the-loop categorization tools
- 📈 **Continuous Improvement** - Iterative retraining reduces FPs by 84.6%

### Production Ready
- 🎬 **Streamlit Web UI** - Real-time inference with visual analytics
- ⚡ **CLI Batch Processing** - High-throughput video processing
- 💻 **Edge Optimized** - Jetson Orin ready (FP16, TensorRT export)
- 📈 **W&B Integration** - Experiment tracking and model versioning

## 📊 Performance Highlights

| Metric | Value |
|--------|-------|
| **False Positive Reduction** | 84.6% (vs baseline) |
| **Model** | YOLO12s @ 1280px |
| **Inference Speed** | ~85 FPS (A100), ~30 FPS (Jetson Orin) |
| **GPU Memory** | 2.4 GB (batch=1) |
| **Hard Negatives Trained** | 134 manually-reviewed images |

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/WMaia9/workzone.git
cd workzone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Process demo video
python scripts/process_video_fusion.py \
  data/03_demo/videos/boston_workzone_short.mp4 \
  --output-dir outputs/demo \
  --enable-phase1-1

# 3. Launch web interface
streamlit run src/workzone/apps/streamlit/app_phase1_1_fusion.py
```

**📖 Full guide**: See [docs/guides/QUICKSTART.md](docs/guides/QUICKSTART.md)

## 🎯 Use Cases

- ✅ **Autonomous Vehicles** - Real-time work zone detection for safe navigation
- ✅ **Fleet Management** - Automated logging of construction zone encounters
- ✅ **Traffic Analysis** - Large-scale video dataset processing
- ✅ **Safety Systems** - Driver warning systems and adaptive cruise control
- ✅ **Research** - Benchmarking and dataset creation

## 📖 Documentation

- **[Quick Start Guide](docs/guides/QUICKSTART.md)** - Get running in 5 minutes
- **[Model Registry](docs/MODEL_REGISTRY.md)** - Performance metrics and model selection
- **[Phase 1.2 Report](docs/reports/PHASE1_2_MINING_REPORT.md)** - Hard-negative mining details
- **[Results Index](docs/reports/RESULTS_INDEX.md)** - Training results and metrics
- **[API Documentation](docs/API.md)** - Python API reference

## 🛠️ CLI Tools

### Video Processing

Process videos with YOLO + CLIP fusion and Phase 1.1 multi-cue logic:

```bash
python scripts/process_video_fusion.py VIDEO_PATH \
  --output-dir outputs/results \
  --enable-phase1-1 \
  --conf 0.50 \
  --device cuda
```

**Outputs**: Annotated video + CSV timeline with frame-level detection data

### Hard-Negative Mining

Mine false positives for model improvement:

```bash
# 1. Mine candidates from videos
python scripts/mine_hard_negatives.py \
  --video-dir data/videos_compressed \
  --output-dir outputs/hardneg_mining \
  --weights weights/yolo12s_fusion_baseline.pt

# 2. Review interactively
python scripts/review_hard_negatives.py interactive \
  --candidates outputs/hardneg_mining/candidates_master.csv

# 3. Generate manifest
python scripts/review_hard_negatives.py manifest \
  --candidates outputs/hardneg_mining/candidates_master.csv
```

### Model Training

Train YOLO on WorkZone dataset:

```bash
yolo train \
  data=data/05_workzone_yolo/workzone_yolo.yaml \
  model=yolo12s.pt \
  epochs=100 \
  imgsz=1280 \
  batch=16 \
  device=0,1  # Multi-GPU
```

**See full training guide**: [docs/guides/TRAINING.md](docs/guides/TRAINING.md)

## 🎨 Web Applications

### Phase 1.1 Fusion App (Recommended)

Complete pipeline with YOLO + CLIP + Phase 1.1 multi-cue logic:

```bash
streamlit run src/workzone/apps/streamlit/app_phase1_1_fusion.py --server.port 8501
```

**Features**:
- Model selection (Hard-Neg Trained, Fusion Baseline, Custom Upload)
- Real-time video playback with detection overlay
- Phase 1.1 multi-cue visualization
- State timeline graph
- CSV export with frame-level data
- Batch processing mode

**Access**: http://localhost:8501

### Alternative Apps

**Basic Detection** - Simple YOLO + scoring:
```bash
streamlit run src/workzone/apps/streamlit/app_basic_detection.py
```

**Advanced Scoring** - Z-scores + EMA smoothing:
```bash
streamlit run src/workzone/apps/streamlit/app_advanced_scoring.py
```

## 📁 Project Structure

```
workzone/
├── configs/                    # YAML configuration files
│   ├── config.yaml            # Main project config
│   └── multi_cue_config.yaml  # Phase 1.1 parameters
├── data/                      # Dataset and videos
│   ├── 01_raw/               # Raw annotations and images
│   ├── 02_processed/         # Processed data + hard negatives
│   ├── 03_demo/              # Demo videos
│   ├── 04_derivatives/       # Generated data
│   └── 05_workzone_yolo/     # YOLO format dataset
├── docs/                      # Documentation
│   ├── guides/               # User guides
│   │   └── QUICKSTART.md
│   ├── reports/              # Technical reports
│   └── MODEL_REGISTRY.md     # Model performance metrics
├── notebooks/                 # Jupyter notebooks for experiments
├── outputs/                   # Generated outputs (videos, CSVs)
├── scripts/                   # CLI processing scripts
│   ├── process_video_fusion.py        # Main video processor
│   ├── mine_hard_negatives.py         # Hard-neg mining
│   ├── review_hard_negatives.py       # Interactive review
│   └── consolidate_candidates.py      # Candidate consolidation
├── src/workzone/              # Main package
│   ├── apps/                 # Web applications
│   │   └── streamlit/        # Streamlit UIs
│   ├── detection/            # Cue classification
│   ├── fusion/               # Multi-cue fusion logic
│   ├── models/               # Model wrappers
│   ├── pipelines/            # Training/inference pipelines
│   ├── temporal/             # Persistence tracking
│   └── utils/                # Utilities
├── tests/                     # Unit tests
├── weights/                   # Model checkpoints
│   ├── yolo12s_hardneg_1280.pt       # ⭐ Production model
│   ├── yolo12s_fusion_baseline.pt    # Baseline model
│   └── *.pt                          # Other models
├── pyproject.toml            # Project metadata
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔬 Technical Details

### Detection Pipeline

1. **YOLO Object Detection** - 50-class construction zone detection
2. **Semantic Scoring** - Weighted scoring based on detected objects
3. **CLIP Verification** (optional) - Semantic similarity check when YOLO is uncertain
4. **Context Boost** - Orange pixel detection for additional evidence
5. **Adaptive EMA** - Temporal smoothing with evidence-based alpha
6. **Phase 1.1 Multi-Cue** - Temporal persistence + AND logic across cue types
7. **State Machine** - Anti-flicker state transitions with hysteresis

### State Machine

```
OUT → APPROACHING → INSIDE → EXITING → OUT
 ↑         ↓           ↓        ↓        ↑
 └─────────┴───────────┴────────┴────────┘
```

**States**:
- **OUT** (Green): No work zone detected
- **APPROACHING** (Orange): Work zone ahead, gathering evidence
- **INSIDE** (Red): Confirmed work zone presence
- **EXITING** (Magenta): Leaving work zone

**Key Feature**: APPROACHING never transitions backward to OUT (prevents flickering)
### Phase 1.1 Multi-Cue Logic

**Cue Groups** (50 YOLO classes mapped to 5 groups):
- **CHANNELIZATION**: Cones, drums, barriers, barricades
- **SIGNAGE**: Temporary traffic control signs, arrow boards
- **PERSONNEL**: Workers, police officers
- **EQUIPMENT**: Work vehicles, construction equipment
- **INFRASTRUCTURE**: Modified road features

**Logic**:
1. Detect objects per frame → Classify into cue groups
2. Track persistence over sliding window (90 frames = 3 seconds)
3. Cue is "sustained" if present in ≥40% of window
4. Work zone confirmed if ≥1 sustained cue type (configurable)

**Configuration**: `configs/multi_cue_config.yaml`

### Hard-Negative Mining

Iterative process to reduce false positives:

1. **Mine**: Process videos with current model, save high-confidence detections
2. **Filter**: Extract frames where Phase 1.1 fails (likely false positives)
3. **Review**: Manual categorization (cones, signs, trucks, other)
4. **Retrain**: Add approved negatives to dataset, retrain model
5. **Evaluate**: Measure FP reduction on held-out test set

**Phase 1.2 Results**: 134 hard negatives → 84.6% FP reduction

### Models

See [docs/MODEL_REGISTRY.md](docs/MODEL_REGISTRY.md) for full details.

**Production Model** (`yolo12s_hardneg_1280.pt`):
- YOLO12s @ 1280px
- Trained with 134 hard negatives
- 84.6% FP reduction vs baseline
- 19 MB, ~85 FPS on A100

**Baseline Model** (`yolo12s_fusion_baseline.pt`):
- YOLO12s @ 960px
- Base training on WorkZone dataset
- Used for hard-negative mining
- 36 MB

## ⚙️ Configuration

### YAML Config

Edit `configs/config.yaml` for global settings:

```yaml
yolo:
  model_name: yolo12s
  imgsz: 1280
  batch_size: 16
  confidence_threshold: 0.50
  device: cuda

processing:
  video_fps: 30
  frame_queue_size: 5
  enable_threading: true
```

Edit `configs/multi_cue_config.yaml` for Phase 1.1 parameters:

```yaml
temporal:
  window_size: 90              # 3 seconds at 30fps
  persistence_threshold: 0.4   # 40% presence required

multi_cue_gate:
  min_sustained_cues: 1        # Require 1 sustained cue type
  confidence_per_cue: 0.20     # Boost per additional cue

detection_thresholds:
  CHANNELIZATION: 0.25
  SIGNAGE: 0.35
  PERSONNEL: 0.50
```

### Python API

```python
from ultralytics import YOLO
from workzone.detection import CueClassifier
from workzone.temporal import PersistenceTracker
from workzone.fusion import MultiCueGate

# Load model
model = YOLO("weights/yolo12s_hardneg_1280.pt")

# Initialize Phase 1.1 components
classifier = CueClassifier()
persistence = PersistenceTracker()
multi_cue = MultiCueGate()

# Process frame
results = model(frame)
cues = classifier.classify_detections(results)
states = persistence.update(cues)
decision = multi_cue.evaluate(cues, states)

print(f"Work Zone Detected: {decision.passed}")
print(f"Confidence: {decision.confidence:.2f}")
```

## 🧪 Testing & Development

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/workzone --cov-report=html

# Specific module
pytest tests/test_detection.py
```

### Code Formatting

```bash
# Format with black
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint
flake8 src/ tests/
```

## 📊 Benchmarks

### Latency (Single Frame, A100)

| Component | Time |
|-----------|------|
| YOLO Inference (1280px) | ~12ms |
| Phase 1.1 Cue Classification | <1ms |
| CLIP (when triggered) | ~30ms |
| Orange Context Boost | <1ms |
| **Total (with CLIP)** | ~43ms (23 FPS) |
| **Total (YOLO only)** | ~13ms (77 FPS) |

### Throughput (Batch Processing)

| Configuration | FPS | Notes |
|--------------|-----|-------|
| A100, batch=1, stride=1 | 77 | Real-time capable |
| A100, batch=1, stride=2 | 150 | 2× speedup |
| 2×A100, batch=16 | 240 | Training mode |
| Jetson Orin (estimated) | 30 | FP16, stride=2 |

### Memory Usage

| Configuration | GPU Memory |
|--------------|-----------|
| YOLO12s @ 1280px, batch=1 | 2.4 GB |
| YOLO12s @ 1280px, batch=16 | 12 GB |
| YOLO12s @ 960px, batch=1 | 1.8 GB |
| + CLIP (ViT-B-32) | +1.2 GB |

## 🚀 Deployment

### Edge Deployment (Jetson Orin)

```bash
# Export to TensorRT
yolo export \
  model=weights/yolo12s_hardneg_1280.pt \
  format=engine \
  imgsz=1280 \
  half=true

# Result: yolo12s_hardneg_1280.engine (~30 FPS expected)
```

### Docker Deployment

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8501
CMD ["streamlit", "run", "src/workzone/apps/streamlit/app_phase1_1_fusion.py"]
```

Build and run:
```bash
docker build -t workzone:latest .
docker run --gpus all -p 8501:8501 workzone:latest
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow code style (black, isort, type hints)
4. Add tests for new functionality
5. Update documentation
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open a Pull Request

**Code Standards**:
- PEP 8 compliance (enforced by black)
- Type hints on all functions
- Docstrings (Google style)
- Tests with >80% coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** - YOLOv8/v11/v12 framework
- **[OpenAI CLIP](https://github.com/openai/CLIP)** - Multi-modal semantic verification
- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** - Open-source CLIP implementation
- **[Streamlit](https://streamlit.io/)** - Interactive web applications
- **[Weights & Biases](https://wandb.ai/)** - Experiment tracking
- **ESV Competition** - Challenge organizers and dataset providers

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/WMaia9/workzone/issues)
- **Discussions**: [GitHub Discussions](https://github.com/WMaia9/workzone/discussions)
- **Email**: wesley.maia@example.com

## 🗺️ Roadmap

### Phase 1.3 (Q1 2026)
- [ ] Expand hard-negative set to 500-1000 images
- [ ] Add OCR for sign text reading
- [ ] Multi-resolution training (640, 960, 1280)
- [ ] Distillation to YOLO11n for ultra-fast inference

### Phase 2.0 (Q2 2026)
- [ ] Temporal transformer models for video understanding
- [ ] End-to-end multi-modal fusion (YOLO + CLIP jointly trained)
- [ ] Real-time deployment on Jetson Orin
- [ ] Mobile app integration

### Research
- [ ] Self-supervised learning on unlabeled dashcam footage
- [ ] Active learning for efficient annotation
- [ ] Explainability and failure analysis tools

---

**Built for ESV Competition** 🏆 | **Edge-Ready for Jetson Orin** 🚀 | **84.6% FP Reduction** 📉

*Last updated: January 3, 2026*
