#!/bin/bash
set -e

echo "🚀 Setting up WorkZone for Jetson Orin Nano (JetPack 6.2)..."

# 1. Create/Activate Venv (System Site Packages is CRITICAL for Jetson)
# We need system packages because standard python-opencv from pip usually lacks GStreamer support on Jetson.
if [ ! -d "venv" ]; then
    echo "📦 Creating venv with --system-site-packages..."
    python3 -m venv venv --system-site-packages
fi
source venv/bin/activate

# 2. Uninstall pip-based OpenCV if present (causes conflicts)
echo "🧹 Cleaning up conflicting OpenCV..."
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>/dev/null || true

# 3. Install PyTorch from Local Wheels (Priority) or NVIDIA Index
echo "🔥 Installing PyTorch..."

# Look for local wheels first
TORCH_WHL=$(find temp_wheels -name "torch-*.whl" | head -n 1)
VISION_WHL=$(find temp_wheels -name "torchvision-*.whl" | head -n 1)

if [ -n "$TORCH_WHL" ]; then
    echo "   Found local PyTorch: $TORCH_WHL"
    pip install "$TORCH_WHL"
else
    echo "   Downloading PyTorch for JetPack 6 (Defaulting to 2.5.0 if not found)..."
    # Fallback to a known compatible version if local file missing
    wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.5.0a0+872d923nv24.08-cp310-cp310-linux_aarch64.whl -O /tmp/torch.whl
    pip install /tmp/torch.whl
fi

# 4. Install Torchvision
if [ -n "$VISION_WHL" ]; then
    echo "   Found local Torchvision: $VISION_WHL"
    pip install "$VISION_WHL"
else
    echo "   Building/Installing Torchvision..."
    # Usually safer to build from source or pick matching version. 
    # For Torch 2.5, we usually want Torchvision 0.20
    pip install torchvision==0.20.0
fi

# 5. Install Other Dependencies (Skipping torch/opencv/torchvision)
echo "📚 Installing dependencies..."
# Extract dependencies excluding conflicts
grep -vE "opencv|torch|torchvision" pyproject.toml > /tmp/reqs_clean.txt
# This is a hacky way to parse toml, better to just use pip install but force no-deps for torch
# Actually, just install . but exclude deps check for torch
pip install "numpy<2" # Fix for OpenCV compatibility
pip install -e . --no-deps

# Re-install essential libs that might have been skipped
pip install "ultralytics>=8.0.0" "transformers>=4.30.0" "open_clip_torch>=2.20.0" "pyserial" "pyyaml" "tqdm" "pillow"

# 6. Verify Installation
echo "🔍 Verifying Installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python3 -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo "✅ Setup Complete! Run 'make workzone' to start."
