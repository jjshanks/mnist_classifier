"""GPU and CUDA configuration utilities for TensorFlow."""

import logging
import os
import platform
import site
import sys
import traceback
from pathlib import Path

import tensorflow as tf

logger = logging.getLogger(__name__)


def configure_gpu_and_cuda() -> None:
    """Configure GPU settings and CUDA paths for TensorFlow XLA compilation.

    This function sets up the necessary environment variables to help TensorFlow's
    XLA compiler find the NVIDIA libdevice library, which is required for GPU
    kernel compilation.
    """
    # First, check if we're already configured
    if "xla_gpu_cuda_data_dir" in os.environ.get("XLA_FLAGS", ""):
        logger.debug("CUDA paths already configured")
        return

    # Set XLA flags for GPU fallback
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found" not in xla_flags:
        xla_flags = (
            "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found " + xla_flags
        )

    # Search for libdevice.10.bc in common locations
    libdevice_found = False
    search_paths = []

    # Add current virtual environment paths first (highest priority)
    if hasattr(sys, "prefix"):
        venv_site = (
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if venv_site.exists():
            # Prioritize nvidia-cuda-nvcc over triton
            search_paths.extend(
                [
                    venv_site / "nvidia" / "cuda_nvcc" / "nvvm" / "libdevice",
                    venv_site / "nvidia" / "cuda_runtime" / "nvvm" / "libdevice",
                    venv_site / "triton" / "backends" / "nvidia" / "lib",
                ]
            )

    # Add site-packages paths
    for site_pkg in site.getsitepackages():
        search_paths.extend(
            [
                Path(site_pkg) / "triton" / "backends" / "nvidia" / "lib",
                Path(site_pkg) / "nvidia" / "cuda_nvcc" / "nvvm" / "libdevice",
                Path(site_pkg) / "nvidia" / "cuda_runtime" / "nvvm" / "libdevice",
            ]
        )

    # Add user site-packages
    if site.USER_SITE:
        search_paths.extend(
            [
                Path(site.USER_SITE) / "triton" / "backends" / "nvidia" / "lib",
                Path(site.USER_SITE) / "nvidia" / "cuda_nvcc" / "nvvm" / "libdevice",
            ]
        )

    # Search for libdevice.10.bc
    for path in search_paths:
        libdevice_path = path / "libdevice.10.bc"
        if libdevice_path.exists():
            # XLA expects the parent directory containing nvvm/libdevice structure
            # For triton, we need special handling as it has a different structure
            if "triton" in str(path) and "nvidia/lib" in str(path):
                # Triton structure: backends/nvidia/lib/libdevice.10.bc
                # XLA expects: <cuda_data_dir>/nvvm/libdevice/libdevice.10.bc
                # So we need to create a symlink or use the lib directory directly
                cuda_data_dir = str(path)  # Use the lib directory directly
            elif "nvidia/cuda_nvcc/nvvm/libdevice" in str(path):
                # NVIDIA package structure:
                # nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
                # XLA expects: <cuda_data_dir>/nvvm/libdevice/libdevice.10.bc
                # So we need to go up to cuda_nvcc directory
                cuda_data_dir = str(path.parent.parent)  # Points to cuda_nvcc
            else:
                # For other structures
                cuda_data_dir = str(path.parent.parent)  # Generic fallback

            xla_flags += f" --xla_gpu_cuda_data_dir={cuda_data_dir}"
            libdevice_found = True
            logger.info(f"Found libdevice.10.bc at: {libdevice_path}")
            logger.info(f"Setting CUDA data dir to: {cuda_data_dir}")
            break

    if not libdevice_found:
        warning_msg = (
            "Could not find libdevice.10.bc in standard locations. "
            "GPU compilation may fail. Consider installing "
            "nvidia-cuda-nvcc-cu12 package."
        )
        logger.warning(warning_msg)

    # Update environment
    os.environ["XLA_FLAGS"] = xla_flags.strip()
    logger.debug(f"XLA_FLAGS set to: {os.environ['XLA_FLAGS']}")


def configure_tensorflow_gpu() -> None:
    """Configure TensorFlow GPU settings for optimal performance."""
    try:
        # Suppress TensorFlow warnings about duplicate library registrations
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # Get GPU devices
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for GPU: {gpu.name}")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth for {gpu.name}: {e}")

            # Log GPU details
            logger.info(f"Configured {len(gpus)} GPU(s) for TensorFlow:")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.name}")

            # Test GPU availability
            logger.info(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
            cuda_ver = tf.sysconfig.get_build_info().get("cuda_version", "Unknown")
            logger.info(f"CUDA version: {cuda_ver}")
            cudnn_ver = tf.sysconfig.get_build_info().get("cudnn_version", "Unknown")
            logger.info(f"cuDNN version: {cudnn_ver}")
        else:
            logger.warning("No GPUs found by TensorFlow, using CPU")
            logger.info("Check nvidia-smi output and CUDA installation")

    except Exception as e:
        logger.error(f"Error configuring TensorFlow GPU: {e}")
        logger.debug(traceback.format_exc())


def configure_wsl2_cuda_paths() -> None:
    """Configure CUDA library paths for WSL2 environment."""
    # Check if we're running in WSL2
    if platform.system() == "Linux" and "microsoft" in platform.release().lower():
        wsl_cuda_path = "/usr/lib/wsl/lib"
        if Path(wsl_cuda_path).exists():
            # Add WSL2 CUDA path to LD_LIBRARY_PATH
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if wsl_cuda_path not in current_ld_path:
                new_ld_path = (
                    f"{wsl_cuda_path}:{current_ld_path}"
                    if current_ld_path
                    else wsl_cuda_path
                )
                os.environ["LD_LIBRARY_PATH"] = new_ld_path
                logger.info(f"Added WSL2 CUDA path to LD_LIBRARY_PATH: {wsl_cuda_path}")

            # Also set CUDNN_PATH if cudnn libraries are found
            cudnn_lib = Path(wsl_cuda_path) / "libcudnn.so"
            if cudnn_lib.exists():
                os.environ["CUDNN_PATH"] = wsl_cuda_path
                logger.info(f"Set CUDNN_PATH to: {wsl_cuda_path}")
        else:
            logger.warning(
                f"WSL2 detected but CUDA libraries not found at {wsl_cuda_path}"
            )


def setup_gpu_environment() -> None:
    """Complete GPU environment setup."""
    # Configure WSL2 CUDA paths first (must be done before TensorFlow import)
    configure_wsl2_cuda_paths()

    # Force clear any existing XLA flags to avoid conflicts
    os.environ.pop("XLA_FLAGS", None)
    configure_gpu_and_cuda()
    configure_tensorflow_gpu()
