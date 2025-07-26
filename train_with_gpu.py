#!/usr/bin/env python3
"""
GPU-enabled training wrapper for WSL2.
This script ensures CUDA libraries are properly loaded before starting training.
"""

import os
import subprocess
import sys

import train_model


def main():
    """Check and restart with proper CUDA environment if needed."""
    # Check if we're in WSL2
    if "microsoft" not in os.uname().release.lower():
        # Not WSL2, just run normally
        train_model.main()
        return

    # Check if CUDA path is already set
    wsl_cuda_path = "/usr/lib/wsl/lib"
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if wsl_cuda_path in current_ld_path:
        # Already configured, run normally
        train_model.main()
    else:
        # Need to restart with proper environment
        print("Restarting with GPU support enabled...")

        # Set up new environment
        new_env = os.environ.copy()
        new_env["LD_LIBRARY_PATH"] = (
            f"{wsl_cuda_path}:{current_ld_path}" if current_ld_path else wsl_cuda_path
        )
        new_env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow verbosity

        # Restart Python with the same arguments but proper environment
        cmd = [sys.executable, "train_model.py", *sys.argv[1:]]

        # Run the command with the new environment
        result = subprocess.run(cmd, env=new_env, check=False)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
