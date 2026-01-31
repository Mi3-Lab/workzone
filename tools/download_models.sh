#!/bin/bash
# Download pretrained models for WorkZone detection

set -e

WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

echo "Downloading WorkZone models..."

# Hard-negative trained model (recommended)
if [ ! -f "$WEIGHTS_DIR/yolo12s_hardneg_1280.pt" ]; then
    echo "Downloading yolo12s_hardneg_1280.pt..."
    # Replace with your actual download URL (Google Drive, Dropbox, Hugging Face, etc.)
    # Example for Google Drive:
    # gdown --id YOUR_FILE_ID -O "$WEIGHTS_DIR/yolo12s_hardneg_1280.pt"
    # Example for direct URL:
    # wget -O "$WEIGHTS_DIR/yolo12s_hardneg_1280.pt" "YOUR_DOWNLOAD_URL"
    echo "TODO: Add download URL for yolo12s_hardneg_1280.pt"
else
    echo "✓ yolo12s_hardneg_1280.pt already exists"
fi

# Fusion baseline model
if [ ! -f "$WEIGHTS_DIR/yolo12s_fusion_baseline.pt" ]; then
    echo "Downloading yolo12s_fusion_baseline.pt..."
    # Replace with your actual download URL
    echo "TODO: Add download URL for yolo12s_fusion_baseline.pt"
else
    echo "✓ yolo12s_fusion_baseline.pt already exists"
fi

# Base YOLO12s (from Ultralytics)
if [ ! -f "$WEIGHTS_DIR/yolo12s.pt" ]; then
    echo "Downloading base yolo12s.pt from Ultralytics..."
    wget -O "$WEIGHTS_DIR/yolo12s.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt" || \
        echo "Warning: Could not download yolo12s.pt"
else
    echo "✓ yolo12s.pt already exists"
fi

echo ""
echo "Model download complete!"
echo "Available models:"
ls -lh "$WEIGHTS_DIR"/*.pt 2>/dev/null || echo "No models found"
