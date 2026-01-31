# Phase 1.1 Scripts

## Test Scripts

- **`test_phase1_1_integrated.py`** - Main test script integrating Phase 1.1 with your existing Streamlit pipeline
- **`visualize_phase1_1_video.py`** - Create annotated video with Phase 1.1 results
- **`analyze_phase1_1.py`** - Generate charts and statistics from results
- **`debug_detections.py`** - Debug script to see raw YOLO detections

## Usage

All scripts should be run from the workspace root:

```bash
cd /home/wesleyferreiramaia/data/workzone
source .venv/bin/activate

# Run integrated test
python scripts/phase1_1/test_phase1_1_integrated.py --video data/demo/boston_workzone_short.mp4 --model weights/best.pt --output outputs/phase1_1_integrated.csv

# Create annotated video
python scripts/phase1_1/visualize_phase1_1_video.py --video data/demo/boston_workzone_short.mp4 --results outputs/phase1_1_integrated.csv --output outputs/phase1_1_annotated.mp4

# Analyze results
python scripts/phase1_1/analyze_phase1_1.py
```
