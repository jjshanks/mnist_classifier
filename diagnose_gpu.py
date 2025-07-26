#!/usr/bin/env python3
"""
Comprehensive GPU diagnostics for TensorFlow in WSL2.
This script helps identify and fix GPU configuration issues.
"""

import os
import platform
import site
import subprocess
from pathlib import Path

import tensorflow as tf


# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    END = "\033[0m"


def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def check_system():
    """Check system information."""
    print_header("System Information")

    print(f"Platform: {platform.system()}")
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")

    # Check if WSL2
    is_wsl2 = "microsoft" in platform.release().lower()
    if is_wsl2:
        print_success("Running in WSL2")
    else:
        print_warning("Not running in WSL2")

    return is_wsl2


def check_nvidia_smi():
    """Check if nvidia-smi works."""
    print_header("NVIDIA Driver Check")

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            print_success("nvidia-smi found")
            print(f"GPU: {result.stdout.strip()}")
            return True
        print_error("nvidia-smi failed")
        return False
    except FileNotFoundError:
        print_error("nvidia-smi not found in PATH")
        return False


def check_cuda_libraries():
    """Check for CUDA libraries in various locations."""
    print_header("CUDA Library Locations")

    libraries_found = {}

    # Check WSL2 location
    wsl_cuda_path = Path("/usr/lib/wsl/lib")
    if wsl_cuda_path.exists():
        cuda_files = list(wsl_cuda_path.glob("*cuda*.so*"))
        if cuda_files:
            print_success(f"WSL2 CUDA driver found at {wsl_cuda_path}")
            print(f"  Files: {', '.join(f.name for f in cuda_files[:5])}")
            libraries_found["wsl_driver"] = str(wsl_cuda_path)

    # Check Python site-packages
    try:
        site_packages = Path(site.getsitepackages()[0])

        # Check for nvidia packages
        nvidia_path = site_packages / "nvidia"
        if nvidia_path.exists():
            print_success(f"NVIDIA packages found in {nvidia_path}")

            # Check specific libraries
            checks = [
                ("cuda_runtime/lib/libcudart.so*", "CUDA Runtime"),
                ("cudnn/lib/libcudnn.so*", "cuDNN"),
                ("cublas/lib/libcublas.so*", "cuBLAS"),
            ]

            for pattern, name in checks:
                libs = list(nvidia_path.glob(pattern))
                if libs:
                    lib_dir = libs[0].parent
                    print_success(f"  {name}: {lib_dir}")
                    libraries_found[name.lower().replace(" ", "_")] = str(lib_dir)
                else:
                    print_error(f"  {name}: Not found")
    except Exception as e:
        print_error(f"Error checking site-packages: {e}")

    return libraries_found


def check_environment():
    """Check environment variables."""
    print_header("Environment Variables")

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        print("LD_LIBRARY_PATH:")
        for path in ld_library_path.split(":"):
            if path:
                print(f"  - {path}")
    else:
        print_warning("LD_LIBRARY_PATH not set")

    # Check other relevant variables
    for var in ["CUDA_PATH", "CUDNN_PATH", "TF_CPP_MIN_LOG_LEVEL"]:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")


def check_tensorflow():
    """Check TensorFlow GPU configuration."""
    print_header("TensorFlow GPU Check")

    # Set minimal environment to reduce warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    try:
        print_success(f"TensorFlow {tf.__version__} imported successfully")

        # Check build info
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

        # Check for GPUs
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print_success(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")

            # Test computation
            print("\nTesting GPU computation...")
            with tf.device("/GPU:0"):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
                print_success(f"GPU computation successful! Result shape: {c.shape}")

            return True
        print_error("No GPUs found by TensorFlow")
        return False

    except Exception as e:
        print_error(f"TensorFlow error: {e}")
        return False


def generate_fix_script(libraries_found):
    """Generate a script to fix the environment."""
    print_header("Recommended Fix")

    if not libraries_found:
        print_error(
            "No NVIDIA libraries found. Please install nvidia-cuda-runtime-cu12"
        )
        print("Run: uv pip install nvidia-cuda-runtime-cu12")
        return

    # Build LD_LIBRARY_PATH
    paths = ["/usr/lib/wsl/lib"]  # Always include WSL path
    for lib_name, lib_path in libraries_found.items():
        if lib_name != "wsl_driver" and lib_path not in paths:
            paths.append(lib_path)

    print("Add this to your ~/.bashrc or use before running Python:")
    ld_path = ":".join(paths)
    export_cmd = f"export LD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH"
    print(f"\n{Colors.YELLOW}{export_cmd}{Colors.END}")
    print(f"{Colors.YELLOW}export TF_CPP_MIN_LOG_LEVEL=2{Colors.END}\n")

    print("Or use the provided training script:")
    print(f"{Colors.GREEN}./train_gpu.sh --epochs 10{Colors.END}")


def main():
    """Run all diagnostics."""
    print_header("TensorFlow GPU Diagnostics")

    # Run checks
    check_system()
    has_nvidia = check_nvidia_smi()
    libraries_found = check_cuda_libraries()
    check_environment()

    # Only check TensorFlow if we have the basic requirements
    tf_works = False
    if has_nvidia and libraries_found:
        tf_works = check_tensorflow()

    # Summary and recommendations
    print_header("Summary")

    if tf_works:
        print_success("GPU is properly configured and working!")
    else:
        print_error("GPU configuration needs fixing")
        generate_fix_script(libraries_found)


if __name__ == "__main__":
    main()
