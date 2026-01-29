#!/bin/bash
set -e

# Ensure script is executable
chmod +x scripts/internal_build_vila.sh

# Container Tag - Adjust if you have a custom build
CONTAINER_TAG="dustynv/tensorrt_llm:0.12-r36.4.0"

echo "[HOST] Launching TensorRT-LLM container ($CONTAINER_TAG) to build VILA 1.5 3B..."
echo "[HOST] This may take a while..."

# Use jetson-containers run.sh to handle mounting and GPU access
./jetson-containers/run.sh \
    -v $(pwd):/workzone \
    $CONTAINER_TAG \
    /bin/bash /workzone/scripts/internal_build_vila.sh

echo "[HOST] Build process finished."

