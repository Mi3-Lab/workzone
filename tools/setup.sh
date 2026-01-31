#!/bin/bash

# WorkZone Quick Setup Script
# This script sets up the development environment

set -e

echo "======================================"
echo "WorkZone Setup Script"
echo "======================================"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip upgraded"

# Install in development mode
echo "Installing WorkZone..."
pip install -e ".[dev]" > /dev/null 2>&1
echo "✓ WorkZone installed"

# Create necessary directories
mkdir -p data/Construction_Data
mkdir -p weights
mkdir -p outputs
mkdir -p logs
echo "✓ Directories created"

# Create .gitkeep files
touch weights/.gitkeep
touch logs/.gitkeep
touch outputs/.gitkeep
echo "✓ .gitkeep files created"

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download model weights: cd weights && wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo12s.pt"
echo "  3. Run tests: pytest tests/ -v"
echo "  4. Train model: python -m src.workzone.cli.train_yolo --device 0"
echo "  5. Run inference: python -m src.workzone.cli.infer_video --video video.mp4 --model weights/best.pt"
echo ""
echo "See README.md for detailed documentation"
echo ""
