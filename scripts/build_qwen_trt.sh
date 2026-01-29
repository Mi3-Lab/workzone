#!/bin/bash
set -e

# Ensure script is executable
chmod +x scripts/internal_build_qwen.sh

# Container Tag - Adjust if you have a custom build
CONTAINER_TAG="dustynv/tensorrt_llm:r36.4.0"

echo "[HOST] Launching TensorRT-LLM container ($CONTAINER_TAG) to build Qwen2.5-VL..."
echo "[HOST] This may take a while..."

# Use jetson-containers run.sh to handle mounting and GPU access
./jetson-containers/run.sh \
    -v $(pwd):/data \
    $CONTAINER_TAG \
    /bin/bash /data/scripts/internal_build_qwen.sh

echo "[HOST] Build process finished."
