#!/usr/bin/env python3
"""Test GPU configuration and availability."""

import logging
import os
import sys
from pathlib import Path

# Configure logging to see all messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run GPU setup BEFORE importing TensorFlow
print("Setting up GPU environment...")
from src.mnist_classifier.utils import setup_gpu_environment  # noqa: E402

setup_gpu_environment()

# Now import TensorFlow
print("\nImporting TensorFlow...")
import tensorflow as tf  # noqa: E402

print("\n" + "=" * 50)
print("GPU Configuration Test")
print("=" * 50)

# Check TensorFlow build info
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Check for GPUs
gpus = tf.config.list_physical_devices("GPU")
print(f"\nNumber of GPUs available: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

    # Test GPU computation
    print("\nTesting GPU computation...")
    with tf.device("/GPU:0"):
        # Create some tensors
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Perform matrix multiplication
        c = tf.matmul(a, b)

        print(f"Matrix multiplication result shape: {c.shape}")
        print(f"Result computed on: {c.device}")

    print("\n✅ GPU is working correctly!")
else:
    print("\n❌ No GPU found. Check your CUDA installation.")
    print("\nTroubleshooting steps:")
    print("1. Run 'nvidia-smi' to check if GPU is detected")
    print("2. Check LD_LIBRARY_PATH includes /usr/lib/wsl/lib")
    print("3. Verify CUDA libraries are installed")

# Show environment variables
print("\n" + "=" * 50)
print("Environment Variables")
print("=" * 50)

print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"CUDNN_PATH: {os.environ.get('CUDNN_PATH', 'Not set')}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
