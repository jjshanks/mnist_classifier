#!/bin/bash
# Setup script for GPU environment in WSL2

# Set CUDA library path for WSL2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2

# Optional: Set CUDA cache directory to avoid recompilation
export CUDA_CACHE_PATH=$HOME/.cuda_cache
mkdir -p $CUDA_CACHE_PATH

echo "GPU environment configured for WSL2"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "TF_CPP_MIN_LOG_LEVEL: $TF_CPP_MIN_LOG_LEVEL"
echo ""
echo "To use this configuration:"
echo "  source setup_gpu_env.sh"
echo "  python train_model.py"
