# 🚀 Work Zone Detection System - Production Deployment Guide

## System Overview

Complete multi-stage work zone detection pipeline with:
- **YOLO**: Object detection (cones, workers, vehicles, signs)
- **CLIP**: Semantic verification for low-confidence detections
- **Phase 1.1**: Multi-cue temporal persistence (AND gate)
- **Phase 1.4**: Scene context pre-filter (highway/urban/suburban)
- **State Machine**: Temporal state tracking (OUT → APPROACHING → INSIDE → EXITING)

**Performance**: 92.8% scene context accuracy, <1ms overhead per frame

---

## Quick Start

### 1. Installation
```bash
# Clone and setup environment
git clone <repo>
cd workzone
bash setup.sh

# Download models
bash scripts/download_models.sh
```

### 2. Train Scene Context Model (One-Time)
```bash
# Option A: Quickstart (automated)
bash scripts/PHASE1_4_QUICKSTART.sh

# Option B: Manual training
python -c "from workzone.models.scene_context import create_training_dataset; \
  create_training_dataset('data/01_raw/annotations/instances_train_gps_split.json')"

srun --gpus=1 --partition gpu -t 180 bash -lc '
  source .venv/bin/activate
  python scripts/train_scene_context.py \
    --dataset-dir data/04_derivatives/scene_context_dataset_v4 \
    --backbone resnet18 --epochs 10 --batch-size 64 --learning-rate 1e-3 \
    --device cuda --auto-class-weights --num-workers 8 --freeze-backbone
  python scripts/train_scene_context.py \
    --dataset-dir data/04_derivatives/scene_context_dataset_v4 \
    --backbone resnet18 --epochs 10 --batch-size 64 --learning-rate 1e-4 \
    --device cuda --auto-class-weights --num-workers 8
'
```

### 3. Process Videos

**Full System (Recommended)**
```bash
python scripts/process_video_fusion.py video.mp4 \
  --enable-phase1-4 \
  --enable-phase1-1 --no-motion \
  --output-dir outputs/demo
```

**Baseline (No Context)**
```bash
python scripts/process_video_fusion.py video.mp4 \
  --enable-phase1-1 --no-motion \
  --output-dir outputs/baseline
```

**Evaluation Mode**
```bash
python scripts/evaluate_phase1_4.py --limit 5 --stride 6
```

---

## Architecture Details

### Pipeline Flow
```
Frame → YOLO → [Score < 0.55?] → CLIP → Phase 1.4 (Scene Context)
                                              ↓
                                    Context-Aware Thresholds
                                              ↓
                           Phase 1.1 (Multi-Cue Gate) → State Machine
                                              ↓
                                      OUT/APPROACHING/INSIDE/EXITING
```

### Phase 1.4: Scene Context Classifier

**Model**: ResNet18 (pretrained ImageNet) → 3-class classifier
- **Classes**: highway, urban, suburban
- **Input**: 224×224 RGB frame
- **Output**: Softmax probabilities
- **Speed**: <1ms GPU, ~5ms CPU
- **Accuracy**: 92.8% (val)

**Per-Class Performance**:
- Highway: 90.9% recall, 98.0% precision
- Urban: 94.3% recall, 94.3% precision
- Suburban: 91.8% recall, 78.9% precision

**Context-Specific Thresholds**:
```python
THRESHOLDS = {
    "highway": {
        "approach_th": 0.60,  # Stricter (avoid shoulder markings)
        "enter_th": 0.75,
        "exit_th": 0.45,
    },
    "urban": {
        "approach_th": 0.50,  # Looser (crowded scenes)
        "enter_th": 0.65,
        "exit_th": 0.40,
    },
    "suburban": {
        "approach_th": 0.55,  # Balanced
        "enter_th": 0.70,
        "exit_th": 0.45,
    },
}
```

---

## Configuration

### Command-Line Options

**Phase 1.4 (Scene Context)**
- `--enable-phase1-4`: Enable scene context pre-filter
- `--scene-context-weights PATH`: Custom weights path

**Phase 1.1 (Multi-Cue)**
- `--enable-phase1-1`: Enable multi-cue temporal gate
- `--p1-window SIZE`: Temporal window (default: 90 frames)
- `--p1-thresh THRESHOLD`: Persistence threshold (default: 0.4)
- `--p1-min-cues N`: Minimum sustained cues (default: 1)
- `--no-motion`: Disable motion validation

**Performance**
- `--stride N`: Process every N-th frame (default: 1)
- `--batch-size N`: YOLO batch size
- `--device cuda|cpu`: Device selection
- `--quiet`: Minimal output

**Output**
- `--output-dir DIR`: Output directory
- `--no-video`: Skip video output (faster)
- `--no-csv`: Skip CSV output (faster)

---

## Output Format

### CSV Timeline
```
frame,time_sec,yolo_score,yolo_score_ema,fused_score_ema,state,
clip_used,clip_score,count_channelization,count_workers,
p1_multi_cue_pass,p1_num_sustained,p1_confidence,scene_context
```

**Key Columns**:
- `state`: OUT | APPROACHING | INSIDE | EXITING
- `scene_context`: highway | urban | suburban
- `fused_score_ema`: Final confidence score
- `clip_used`: 1 if CLIP was triggered
- `p1_multi_cue_pass`: 1 if Phase 1.1 gate passed

### Video Output
Annotated video with:
- Color-coded banner (green=OUT, orange=APPROACHING, red=INSIDE, magenta=EXITING)
- Score display
- CLIP indicator (cyan)
- Phase 1.1 status (green/red)

---

## Performance Benchmarks

### Timing (per frame, A100 GPU)
- YOLO: 35ms
- CLIP (when triggered): 22ms
- Phase 1.1: <1ms
- Phase 1.4: <1ms
- **Total**: ~37ms/frame (27 FPS)

### Accuracy Metrics
| Component | Metric | Value |
|-----------|--------|-------|
| YOLO | mAP@0.5 | 0.847 |
| Phase 1.4 | Scene accuracy | 92.8% |
| Phase 1.1 | FP reduction | 15-25% |

---

## Dataset Requirements

### Scene Context Training
**Minimum**: 500 images per class  
**Recommended**: 1000+ images per class

**Current Dataset (v4)**:
- Highway: 542 images
- Urban: 800 images
- Suburban: 245 images

**Data Sources**:
- COCO: `scene_environment` tags
- Manual: Regex filters for specific contexts

### Adding Custom Data
```python
from workzone.models.scene_context import create_training_dataset

# Add your images to:
# data/04_derivatives/scene_context_dataset_custom/
#   highway/
#   urban/
#   suburban/

# Then retrain
python scripts/train_scene_context.py \
  --dataset-dir data/04_derivatives/scene_context_dataset_custom \
  --backbone resnet18 --epochs 20 --batch-size 64
```

---

## Troubleshooting

### Model Not Loading
```bash
# Check if weights exist
ls -lh weights/scene_context_classifier.pt

# Retrain if missing
bash scripts/PHASE1_4_QUICKSTART.sh
```

### Architecture Mismatch
**Error**: `Missing key(s) in state_dict`

**Solution**: Ensure backbone matches training
```python
# Weights trained with ResNet18
predictor = SceneContextPredictor(
    model_path="weights/scene_context_classifier.pt",
    backbone="resnet18"  # Must match training
)
```

### Poor Scene Classification
**Symptoms**: All videos classified as suburban

**Solutions**:
1. Expand training dataset with more highway/urban examples
2. Adjust class weights during training
3. Use data augmentation
4. Consider ensemble models

### Slow Inference
```bash
# Use stride for faster processing
--stride 4  # Process every 4th frame

# Disable expensive components
--no-video --no-csv  # Skip outputs

# Use CPU for small batches
--device cpu
```

---

## API Usage

### Python Integration
```python
from workzone.models.scene_context import SceneContextPredictor, SceneContextConfig
import cv2

# Initialize predictor
predictor = SceneContextPredictor(
    model_path="weights/scene_context_classifier.pt",
    device="cuda",
    backbone="resnet18"
)

# Predict on frame
frame = cv2.imread("frame.jpg")
context, confidences = predictor.predict(frame)

print(f"Scene: {context}")
print(f"Confidence: {confidences[context]:.2%}")

# Get context-specific thresholds
thresholds = SceneContextConfig.THRESHOLDS[context]
print(f"Approach threshold: {thresholds['approach_th']}")
```

### Batch Processing
```python
from pathlib import Path
import subprocess

videos = Path("data/videos_compressed").glob("*.mp4")

for video in videos:
    subprocess.run([
        "python", "scripts/process_video_fusion.py",
        str(video),
        "--enable-phase1-4",
        "--enable-phase1-1", "--no-motion",
        "--output-dir", "outputs/batch",
        "--stride", "4"
    ])
```

---

## Model Weights

| Model | Size | Description |
|-------|------|-------------|
| `yolo12s_hardneg_1280.pt` | 24 MB | YOLO hard-negative trained |
| `scene_context_classifier.pt` | 44 MB | Phase 1.4 ResNet18 |
| CLIP (cached) | ~350 MB | OpenAI ViT-B/32 |

**Download**: Models are automatically downloaded on first run or via:
```bash
bash scripts/download_models.sh
```

---

## Evaluation

### Compare Baseline vs Phase 1.4
```bash
python scripts/evaluate_phase1_4.py \
  --videos data/videos_compressed/*.mp4 \
  --limit 10 \
  --stride 6
```

**Output**: JSON report with:
- State counts (APPROACHING, INSIDE, OUT)
- Transition counts
- CLIP trigger frequency
- Scene context distribution
- Per-video comparisons

### Metrics
```bash
# CSV analysis
python -c "
import pandas as pd
df = pd.read_csv('outputs/demo/video_timeline_fusion.csv')
print(df['state'].value_counts())
print(df['scene_context'].value_counts())
print(f'Average confidence: {df[\"fused_score_ema\"].mean():.3f}')
"
```

---

## Production Checklist

- [ ] Models trained and validated (>90% accuracy)
- [ ] Weights backed up to secure storage
- [ ] GPU acceleration enabled
- [ ] Logging configured (see `workzone/utils/logging_config.py`)
- [ ] Error handling tested
- [ ] Performance profiled (target: >20 FPS)
- [ ] Documentation updated
- [ ] User training completed

---

## Support & Maintenance

### Version Compatibility
- Python: 3.11+
- PyTorch: 2.0+
- CUDA: 11.8+ (for GPU)
- Ultralytics: 8.0+

### Updates
```bash
# Update dependencies
pip install -U ultralytics torch torchvision open-clip-torch

# Retrain models with new data
python scripts/train_scene_context.py --dataset-dir data/new_dataset
```

### Monitoring
```bash
# Check GPU usage
nvidia-smi

# Profile inference
python scripts/process_video_fusion.py video.mp4 --quiet --no-video --no-csv
```

---

## References

- **Phase 1.1 Guide**: `docs/guides/PHASE1_1_GUIDE.md`
- **Phase 1.4 Guide**: `docs/guides/PHASE1_4_SCENE_CONTEXT.md`
- **Quick Reference**: `PHASE1_4_QUICK_REFERENCE.md`
- **Implementation Summary**: `PHASE1_4_SUMMARY.md`
- **Project Index**: `PROJECT_INDEX.md`

---

## Contact

For issues, questions, or contributions:
- GitHub Issues: <repo_url>/issues
- Documentation: `docs/`
- Examples: `notebooks/`

**Last Updated**: January 4, 2026  
**System Version**: Phase 1.4 Complete  
**Model Version**: ResNet18 3-class (v4)
