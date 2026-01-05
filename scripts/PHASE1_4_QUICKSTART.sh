#!/bin/bash

# Quick-start: Phase 1.4 Scene Context Pre-Filter
# Usage: bash scripts/PHASE1_4_QUICKSTART.sh

set -e

WORKDIR="/home/wesleyferreiramaia/data/workzone"
cd "$WORKDIR"
source .venv/bin/activate

echo "=========================================="
echo "Phase 1.4 Scene Context Quick-Start"
echo "=========================================="

# Step 1: Create training dataset
echo ""
echo "[1/4] Creating scene context training dataset..."
python3 << 'EOF'
from workzone.models.scene_context import create_training_dataset
from pathlib import Path

dataset_dir = Path("data/04_derivatives/scene_context_dataset")
if dataset_dir.exists():
    print(f"  ✓ Dataset already exists at {dataset_dir}")
else:
    print(f"  Creating dataset from COCO annotations...")
    create_training_dataset(
        coco_json_path="data/01_raw/annotations/instances_train_gps_split.json",
        output_dir=str(dataset_dir)
    )
EOF

# Step 2: Train scene context classifier
echo ""
echo "[2/4] Training scene context classifier..."
echo "  (Takes ~10-15 minutes on A100, ~1 hour on CPU)"
python scripts/train_scene_context.py \
    --dataset-dir data/04_derivatives/scene_context_dataset \
    --output weights/scene_context_classifier.pt \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 1e-3

# Step 3: Test Phase 1.4 on demo video
echo ""
echo "[3/4] Testing Phase 1.4 on Boston demo video..."
python scripts/process_video_fusion.py \
    data/videos_compressed/boston_2bdb5a72602342a5991b402beb8b7ab4_000001_23370_snippet.mp4 \
    --output-dir outputs/phase1_4_demo \
    --enable-phase1-4 \
    --scene-context-weights weights/scene_context_classifier.pt \
    --enable-phase1-1 \
    --no-motion \
    --stride 2

# Step 4: Analyze results
echo ""
echo "[4/4] Analyzing Phase 1.4 results..."
python3 << 'EOF'
import pandas as pd
from pathlib import Path

csv_file = Path("outputs/phase1_4_demo").glob("*timeline_fusion.csv")
csv_file = list(csv_file)[0] if list(csv_file) else None

if not csv_file:
    print("  ❌ No CSV output found")
    exit(1)

df = pd.read_csv(csv_file)

print(f"  ✓ Processed {len(df)} frames")
print(f"\n  Scene Context Distribution:")
if 'scene_context' in df.columns:
    context_counts = df['scene_context'].value_counts()
    for ctx, count in context_counts.items():
        pct = 100 * count / len(df)
        print(f"    {ctx:10s}: {count:4d} frames ({pct:5.1f}%)")
else:
    print("    (scene_context column not found - Phase 1.4 may not be enabled)")

print(f"\n  State Transitions:")
state_counts = df['state'].value_counts()
for state, count in state_counts.items():
    pct = 100 * count / len(df)
    print(f"    {state:12s}: {count:4d} frames ({pct:5.1f}%)")

print(f"\n  CSV saved to: {csv_file}")
print(f"  Video saved to: outputs/phase1_4_demo/*_annotated_fusion.mp4")
EOF

echo ""
echo "=========================================="
echo "✅ Phase 1.4 Quick-Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review Phase 1.4 documentation:"
echo "     docs/guides/PHASE1_4_SCENE_CONTEXT.md"
echo "  2. Compare with/without Phase 1.4:"
echo "     diff outputs/phase1_3_demo/* outputs/phase1_4_demo/*"
echo "  3. Fine-tune thresholds in:"
echo "     src/workzone/models/scene_context.py (SceneContextConfig)"
echo ""
