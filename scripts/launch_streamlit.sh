#!/bin/bash

# WorkZone Streamlit Launcher
# This script properly sets up the environment and launches the Streamlit app

set -e

echo "======================================"
echo "WorkZone Streamlit App Launcher"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Set PYTHONPATH to include src directory
echo "🔧 Setting PYTHONPATH..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${SCRIPT_DIR}/src:$PYTHONPATH"
echo "✅ PYTHONPATH set"

# Set LD_LIBRARY_PATH for Jetson (libcusparseLt)
if [ -d "${PROJECT_ROOT}/libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib" ]; then
    echo "🔧 Setting LD_LIBRARY_PATH for libcusparseLt..."
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib:$LD_LIBRARY_PATH"
    echo "✅ LD_LIBRARY_PATH set"
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
    echo "✅ Streamlit installed"
fi

# Launch the app
echo ""
echo "🚀 Launching WorkZone Streamlit App..."
echo "📱 App will be available at: http://localhost:8502"
echo ""
streamlit run src/workzone/apps/streamlit/app_phase2_1_evaluation.py --server.port 8502 --server.headless true