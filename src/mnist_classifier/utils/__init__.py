"""Utility modules for the MNIST classifier."""

from .gpu_config import (
    configure_gpu_and_cuda,
    configure_tensorflow_gpu,
    setup_gpu_environment,
)

__all__ = [
    "setup_gpu_environment",
    "configure_gpu_and_cuda",
    "configure_tensorflow_gpu",
]
