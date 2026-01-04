# Model Registry

Performance metrics and specifications for all trained models in the WorkZone project.

## Production Models

### yolo12s_hardneg_1280.pt ⭐ (Recommended)

**Description**: YOLO12s trained with 134 manually-reviewed hard negatives at maximum resolution

**Training Configuration**:
- Base model: YOLO12s (19M parameters)
- Resolution: 1280×1280
- Batch size: 16
- Epochs: 71 (early stopped, patience=20)
- Hardware: 2×A100 40GB
- Training time: ~2.5 hours
- Dataset: WorkZone + 134 hard negatives

**Performance**:
| Metric | Value |
|--------|-------|
| False Positive Reduction | **84.6%** (vs baseline) |
| FP Count (37 test images) | 91 (baseline: 590) |
| Model Size | 19 MB |
| Inference Speed (A100) | ~85 FPS @ 1280px |
| Inference Speed (Jetson Orin) | ~30 FPS @ 1280px (estimated) |

**Hard-Negative Composition**:
- Random traffic cones: 52 images
- Roadside signs: 51 images
- Orange trucks/equipment: 24 images
- Other roadwork lookalikes: 7 images

**Use Cases**:
- Production deployment ✅
- High-precision requirements ✅
- Edge devices (Jetson Orin) ✅
- Real-time video processing ✅

**Location**: `weights/yolo12s_hardneg_1280.pt`

---

### yolo12s_fusion_baseline.pt

**Description**: YOLO12s trained on base WorkZone dataset (pre hard-negatives)

**Training Configuration**:
- Base model: YOLO12s (19M parameters)
- Resolution: 960×960
- Batch size: 35
- Epochs: 300
- Hardware: A100 40GB
- Dataset: WorkZone (50 classes)

**Performance**:
| Metric | Value |
|--------|-------|
| mAP@50 | 0.72 |
| Precision | 0.68 |
| Recall | 0.71 |
| Model Size | 36 MB |
| FP Count (37 test images) | 590 |

**Use Cases**:
- Baseline comparison
- Hard-negative mining
- Transfer learning base

**Location**: `weights/yolo12s_fusion_baseline.pt`

---

## Development Models

### yolo12s.pt

**Description**: Pretrained YOLO12s from Ultralytics (generic COCO weights)

**Specifications**:
- Parameters: 19M
- Size: 19 MB
- Pretrained on: COCO dataset

**Use Cases**:
- Transfer learning starting point
- Quick prototyping

**Location**: `weights/yolo12s.pt`

---

### yolo11n.pt

**Description**: YOLO11 nano - lightweight variant

**Specifications**:
- Parameters: 5.4M
- Size: 5.4 MB
- Speed: Very fast

**Use Cases**:
- Ultra-fast inference
- Resource-constrained devices
- Lower accuracy acceptable

**Location**: `weights/yolo11n.pt`

---

### yolov8s.pt

**Description**: YOLOv8 small - previous generation

**Specifications**:
- Parameters: 11M
- Size: 22 MB

**Use Cases**:
- Legacy compatibility
- Comparison benchmarks

**Location**: `weights/yolov8s.pt`

---

## Training History

### Phase 1.2: Hard-Negative Training (Jan 2026)

**Objective**: Reduce false positives by training on challenging negative examples

**Process**:
1. Mined 17,957 candidate frames from 406 videos
2. Manual review and categorization
3. Selected 134 hard negatives across 4 categories
4. Retrained YOLO12s at 1280px with hard negatives
5. Achieved 84.6% false positive reduction

**Results**: `yolo12s_hardneg_1280.pt` (production model)

**Documentation**: See [docs/reports/PHASE1_2_MINING_REPORT.md](../reports/PHASE1_2_MINING_REPORT.md)

---

### Phase 1.1: Baseline Training (Dec 2025)

**Objective**: Train initial YOLO model on WorkZone 50-class dataset

**Process**:
1. Annotated dataset with 50 construction zone classes
2. Trained YOLO12s at 960px
3. Integrated CLIP semantic verification
4. Implemented Phase 1.1 multi-cue logic

**Results**: `yolo12s_fusion_baseline.pt`

---

## Model Selection Guide

### For Production Deployment

**Use**: `yolo12s_hardneg_1280.pt`
- Highest precision
- Lowest false positives
- Edge-ready

### For Training/Fine-tuning

**Use**: `yolo12s_fusion_baseline.pt` or `yolo12s.pt`
- Good starting checkpoints
- Transfer learning ready

### For Hard-Negative Mining

**Use**: `yolo12s_fusion_baseline.pt`
- Generates false positives to learn from
- Well-calibrated confidence scores

### For Maximum Speed (Low Accuracy OK)

**Use**: `yolo11n.pt`
- Smallest model
- Fastest inference
- Lower accuracy

---

## Benchmark Comparisons

### False Positive Rate (37 Test Images)

| Model | FP Count | Reduction vs Baseline |
|-------|----------|----------------------|
| yolo12s_hardneg_1280 | 91 | **84.6%** ↓ |
| yolo12s_fusion_baseline | 590 | baseline |

### Inference Speed (Single Frame, FP16)

| Model | Resolution | A100 FPS | Orin FPS (est) |
|-------|-----------|----------|----------------|
| yolo12s_hardneg_1280 | 1280×1280 | ~85 | ~30 |
| yolo12s_fusion_baseline | 960×960 | ~120 | ~45 |
| yolo11n | 640×640 | ~240 | ~90 |

### GPU Memory Usage

| Model | Resolution | Batch=1 | Batch=16 |
|-------|-----------|---------|----------|
| yolo12s_hardneg_1280 | 1280×1280 | 2.4 GB | 12 GB |
| yolo12s_fusion_baseline | 960×960 | 1.8 GB | 8 GB |
| yolo11n | 640×640 | 1.2 GB | 4 GB |

---

## Exporting for Edge Deployment

### TensorRT (Jetson Orin)

```bash
# Export to TensorRT
yolo export \
  model=weights/yolo12s_hardneg_1280.pt \
  format=engine \
  imgsz=1280 \
  half=true \
  device=0

# Result: yolo12s_hardneg_1280.engine
```

### ONNX (General)

```bash
# Export to ONNX
yolo export \
  model=weights/yolo12s_hardneg_1280.pt \
  format=onnx \
  imgsz=1280 \
  simplify=true

# Result: yolo12s_hardneg_1280.onnx
```

---

## Model Versioning

Models follow semantic versioning in filenames:

**Format**: `{architecture}_{variant}_{resolution}.pt`

**Examples**:
- `yolo12s_hardneg_1280.pt` - YOLO12s, hard-neg trained, 1280px
- `yolo12s_fusion_baseline.pt` - YOLO12s, fusion baseline
- `yolo11n.pt` - YOLO11 nano

---

## Future Models

### Planned Training (Phase 1.3)

- [ ] Expand hard-negative set to 500-1000 images
- [ ] Train at multiple resolutions (640, 960, 1280)
- [ ] Distill to YOLO11n for ultra-fast inference
- [ ] Add text-reading capabilities (OCR integration)

### Research Directions

- [ ] Multi-modal fusion (CLIP + YOLO end-to-end)
- [ ] Temporal models (LSTM/Transformer for video)
- [ ] Self-supervised learning on unlabeled videos
- [ ] Active learning for efficient annotation

---

**Last Updated**: January 3, 2026
