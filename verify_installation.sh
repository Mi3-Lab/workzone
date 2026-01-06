#!/bin/bash

# WorkZone Installation Verification Script
# This script verifies that all dependencies are correctly installed

set -e

echo "======================================"
echo "WorkZone Installation Verification"
echo "======================================"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Check critical imports
echo ""
echo "Checking critical imports..."

imports=(
    "torch:torch"
    "torchvision:torchvision"
    "ultralytics:ultralytics"
    "streamlit:streamlit"
    "cv2:opencv-python"
    "matplotlib:matplotlib"
    "numpy:numpy"
    "pandas:pandas"
    "transformers:transformers"
    "open_clip_torch:open_clip_torch"
    "easyocr:easyocr"
)

failed_imports=()

for import_check in "${imports[@]}"; do
    module=$(echo $import_check | cut -d: -f1)
    package=$(echo $import_check | cut -d: -f2)

    if python -c "import $module" 2>/dev/null; then
        echo "✅ $package ($module)"
    else
        echo "❌ $package ($module) - FAILED"
        failed_imports+=("$package")
    fi
done

# Check workzone package
echo ""
if python -c "import workzone" 2>/dev/null; then
    echo "✅ workzone package"
else
    echo "❌ workzone package - FAILED"
    failed_imports+=("workzone")
fi

# Check model files
echo ""
echo "Checking model files..."
models=(
    "weights/yolo12s_hardneg_1280.pt"
    "weights/yolo12s_fusion_baseline.pt"
    "weights/scene_context_classifier.pt"
)

for model in "${models[@]}"; do
    if [ -f "$model" ]; then
        size=$(du -h "$model" | cut -f1)
        echo "✅ $model ($size)"
    else
        echo "❌ $model - MISSING"
        failed_imports+=("$model")
    fi
done

# Summary
echo ""
echo "======================================"
if [ ${#failed_imports[@]} -eq 0 ]; then
    echo "🎉 ALL CHECKS PASSED!"
    echo "WorkZone is ready to use."
    echo ""
    echo "Next steps:"
    echo "  1. Launch app: streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py"
    echo "  2. Or run: ./launch_streamlit.sh"
else
    echo "❌ SOME CHECKS FAILED!"
    echo "Failed components: ${failed_imports[*]}"
    echo ""
    echo "Please check the installation steps in README.md"
    exit 1
fi
echo "======================================"