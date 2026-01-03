#!/bin/bash
# Quick download script for Phase 1.1 results
# Run this on your local machine to download all results from HPC

echo "================================================"
echo "Phase 1.1 Results Download Script"
echo "================================================"
echo ""

# Configuration
HPC_USER="wesleyferreiramaia"
HPC_HOST="hpc.ucmerced.edu"
REMOTE_PATH="/home/wesleyferreiramaia/data/workzone/outputs"
LOCAL_DIR="./workzone_phase1_1_results"

# Create local directory
mkdir -p "$LOCAL_DIR"
echo "📁 Created directory: $LOCAL_DIR"
echo ""

# Files to download
FILES=(
    "phase1_1_annotated.mp4:Annotated video with detections and state visualization"
    "phase1_1_test.csv:Detailed frame-by-frame results"
    "phase1_1_visualization.png:Time-series charts of cues and state"
)

echo "🔄 Downloading Phase 1.1 results..."
echo ""

for FILE_SPEC in "${FILES[@]}"; do
    FILE="${FILE_SPEC%%:*}"
    DESC="${FILE_SPEC##*:}"
    
    echo "  📥 $FILE"
    echo "     └─ $DESC"
    
    scp "$HPC_USER@$HPC_HOST:$REMOTE_PATH/$FILE" "$LOCAL_DIR/" 2>/dev/null
    
    if [ -f "$LOCAL_DIR/$FILE" ]; then
        SIZE=$(du -h "$LOCAL_DIR/$FILE" | cut -f1)
        echo "     ✅ Downloaded ($SIZE)"
    else
        echo "     ⚠️  Failed to download"
    fi
    echo ""
done

echo "================================================"
echo "✅ Download Complete!"
echo ""
echo "📂 Files downloaded to: $LOCAL_DIR"
echo ""
echo "📋 Next steps:"
echo "  1. Open 'phase1_1_annotated.mp4' in any video player"
echo "  2. Analyze 'phase1_1_test.csv' in Excel or Python"
echo "  3. View 'phase1_1_visualization.png' for charts"
echo ""
echo "📖 For more info, see DOWNLOAD_RESULTS.md on HPC"
echo "================================================"
