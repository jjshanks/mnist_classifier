#!/bin/bash
# GPU-enabled training script for WSL2 using uv

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running in WSL2
if [[ $(uname -r) == *"microsoft"* ]]; then
    echo -e "${YELLOW}WSL2 detected - configuring GPU environment...${NC}"

    # Get the site-packages path
    SITE_PACKAGES=$(uv run python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

    # Build the library paths
    LD_PATHS="/usr/lib/wsl/lib"

    if [ -d "$SITE_PACKAGES/nvidia/cuda_runtime/lib" ]; then
        LD_PATHS="$LD_PATHS:$SITE_PACKAGES/nvidia/cuda_runtime/lib"
        echo -e "${GREEN}✓ Found CUDA runtime libraries${NC}"
    fi

    if [ -d "$SITE_PACKAGES/nvidia/cudnn/lib" ]; then
        LD_PATHS="$LD_PATHS:$SITE_PACKAGES/nvidia/cudnn/lib"
        echo -e "${GREEN}✓ Found cuDNN libraries${NC}"
    fi

    if [ -d "$SITE_PACKAGES/nvidia/cublas/lib" ]; then
        LD_PATHS="$LD_PATHS:$SITE_PACKAGES/nvidia/cublas/lib"
        echo -e "${GREEN}✓ Found cuBLAS libraries${NC}"
    fi

    echo -e "\n${GREEN}Running training with GPU support...${NC}\n"

    # Run with all environment variables set
    LD_LIBRARY_PATH="$LD_PATHS:$LD_LIBRARY_PATH" \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    exec uv run python src/mnist_classifier/training/train.py "$@"
else
    # Not WSL2, just run normally
    exec uv run python src/mnist_classifier/training/train.py "$@"
fi
