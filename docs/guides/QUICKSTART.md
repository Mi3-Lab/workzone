# WorkZone Quick Start Guide

Get up and running with WorkZone construction zone detection in 5 minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for training, 4GB+ for inference

## Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/WMaia9/workzone.git
cd workzone

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Download Pretrained Models

```bash
# Download YOLO weights to weights/ folder
mkdir -p weights
cd weights

# Hard-negative trained model (recommended)
wget https://your-model-host/yolo12s_hardneg_1280.pt

# Or use base YOLO12s
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo12s.pt
```

## Quick Demo

### Run on Demo Video

```bash
# Process single video with default settings
python tools/process_video_fusion.py \
  data/03_demo/videos/boston_workzone_short.mp4 \
  --output-dir outputs/demo \
  --enable-phase1-1

# View results
ls outputs/demo/
# boston_workzone_short_annotated_fusion.mp4  # Annotated video
# boston_workzone_short_timeline_fusion.csv   # Frame-by-frame data
```

### Launch Web Interface

```bash
# Start Streamlit app (recommended)
make app
# or manually:
streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py --server.port 8502

# Open browser to http://localhost:8502
# Upload video and see real-time detection!
```

**OCR Features:**
- **Customizable OCR threshold** (0.1-0.9): Adjust text detection sensitivity
- **Speed sign priority**: Speed limit signs automatically prioritized in detection
- **Visual alerts**: Large red warning box when speed signs detected
- **Score boost controls**: Enable/disable OCR confidence boost for workzone keywords
- **Full-Frame & YOLO-Guided modes**: Choose between scanning entire frame or only detected sign regions

**Batch Processing Features:**
- **Integrated video player**: Play/pause/seek through processed video with native controls
- **H.264 encoding**: Browser-compatible video format with timeline support
- **Real-time graphs**: Latency profiles, throughput metrics, per-cue confidences
- **Export options**: Download annotated video, CSV timeline, and config JSON

## CLI Script Reference

### Video Processing

```bash
# Basic usage
python tools/process_video_fusion.py VIDEO_PATH

# Full options
python tools/process_video_fusion.py VIDEO_PATH \
  --output-dir outputs/results \
  --weights weights/yolo12s_hardneg_1280.pt \
  --device cuda \
  --conf 0.50 \
  --stride 2 \
  --enable-phase1-1
```

**Options:**
- `--output-dir`: Where to save results (default: `./outputs`)
- `--weights`: Model weights path (default: hard-neg trained model)
- `--weights-baseline`: Use fusion baseline model instead
- `--device`: `cuda` or `cpu` (default: cuda)
- `--conf`: YOLO confidence threshold (default: 0.25)
- `--stride`: Frame stride - process every Nth frame (default: 2)
- `--no-clip`: Disable CLIP semantic verification
- `--enable-phase1-1`: Enable Phase 1.1 multi-cue logic

### Hard-Negative Mining

```bash
# Mine hard negatives from videos
python tools/mine_hard_negatives.py \
  --video-dir data/videos_compressed \
  --output-dir outputs/hardneg_mining_new \
  --weights weights/yolo12s_fusion_baseline.pt \
  --gpu-id 0

# Review candidates interactively
python -m src.workzone.utils.review_hard_negatives interactive \
  --candidates outputs/hardneg_mining_new/candidates_master.csv \
  --images-dir outputs/hardneg_mining_new/images

# Export approved hard negatives
python -m src.workzone.utils.review_hard_negatives manifest \
  --candidates outputs/hardneg_mining_new/candidates_master.csv
```

## Training YOLO

### Basic Training (YOLO12)

```bash
# Train on workzone dataset using CLI
python -m workzone.cli.train_yolo \
  --model yolo12s.pt \
  --data data/05_workzone_yolo/workzone_yolo.yaml \
  --imgsz 1280 \
  --batch 32 \
  --epochs 100 \
  --device 0 \
  --run-name yolo12_basic_training \
  --project workzone_yolo_training
```

### Multi-GPU Training (YOLO12)

```bash
# Use 2 GPUs with DDP
python -m workzone.cli.train_yolo \
  --model yolo12s.pt \
  --data data/05_workzone_yolo/workzone_yolo.yaml \
  --imgsz 1280 \
  --batch 64 \
  --epochs 100 \
  --device 0,1 \
  --run-name yolo12_multi_gpu_training \
  --project workzone_yolo_training
```

### YOLO11 Training (Latest Architecture)

```bash
# Train with YOLO11 at 1080px resolution
python -m workzone.cli.train_yolo \
  --model yolo11.yaml \
  --data data/05_workzone_yolo/workzone_yolo.yaml \
  --imgsz 1080 \
  --batch 96 \
  --epochs 100 \
  --device 0,1 \
  --run-name yolo11_1080px_training \
  --project workzone_yolo11_training
```

### With Hard Negatives

```bash
# After mining and reviewing hard negatives
python -m workzone.cli.train_yolo \
  --model weights/yolo12s_fusion_baseline.pt \
  --data data/05_workzone_yolo/workzone_yolo.yaml \
  --imgsz 1280 \
  --batch 64 \
  --epochs 100 \
  --device 0,1 \
  --run-name yolo12_hardneg_training \
  --project workzone_yolo_training \
  --workers 8
```

## Understanding Outputs

### Video Processing Outputs

**Annotated Video** (`*_annotated_fusion.mp4`):
- Visual overlay with bounding boxes
- State banner: OUT (green), APPROACHING (orange), INSIDE (red), EXITING (magenta)
- CLIP indicator when active
- Phase 1.1 status (if enabled)

**Timeline CSV** (`*_timeline_fusion.csv`):

| Column | Description |
|--------|-------------|
| `frame` | Frame number |
| `time_sec` | Timestamp in seconds |
| `yolo_score` | Raw YOLO semantic score |
| `yolo_score_ema` | Smoothed YOLO score (EMA) |
| `fused_score_ema` | Final fused score (YOLO + CLIP + context) |
| `state` | Work zone state |
| `clip_used` | 1 if CLIP was triggered |
| `clip_score` | CLIP similarity score |
| `count_channelization` | Number of cones/barriers detected |
| `count_workers` | Number of workers detected |
| `p1_multi_cue_pass` | Phase 1.1 pass/fail |
| `p1_num_sustained` | Number of sustained cue types |
| `p1_confidence` | Phase 1.1 confidence score |

### Training Outputs

Saved to `runs/train/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.csv` - Training metrics per epoch
- `confusion_matrix.png` - Class confusion matrix
- `results.png` - Training curves (loss, mAP, precision, recall)

## Next Steps

- **Fine-tune parameters**: Edit `configs/config.yaml` and `configs/multi_cue_config.yaml`
- **Train custom model**: Use your own dataset in YOLO format
- **Deploy to edge**: Export to TensorRT for Jetson Orin
- **Add hard negatives**: Improve model by mining false positives

## Troubleshooting

**CUDA out of memory:**
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 960`
- Use stride: `--stride 4`

**CLIP loading slow:**
- First run downloads ~300MB model to `~/.cache/open_clip/`
- Subsequent runs load from cache (2-3 seconds)

**False positives:**
- Mine hard negatives and retrain
- Increase confidence threshold: `--conf 0.60`
- Enable Phase 1.1: `--enable-phase1-1`

**Need help?**
- Check [docs/reports/](../reports/) for detailed technical reports
- See example configs in `configs/`
- Review trained model metrics in [docs/MODEL_REGISTRY.md](MODEL_REGISTRY.md)
