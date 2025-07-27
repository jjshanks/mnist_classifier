#!/usr/bin/env python3
"""
Set up a symlink to the latest trained model for CLI tools.

This script finds the most recently trained model and creates
a symlink at models/mnist_model.h5 for easy CLI access.
"""

import argparse
import sys
from pathlib import Path


def find_latest_model(experiments_dir: Path) -> Path | None:
    """Find the latest model file in experiments directory.

    Args:
        experiments_dir: Path to experiments directory

    Returns:
        Path to latest model file or None if not found
    """
    model_files = []

    # Look for final_model.h5 files in experiment directories
    for exp_dir in experiments_dir.glob("mnist_*/"):
        final_model = exp_dir / "final_model.h5"
        if final_model.exists():
            model_files.append(final_model)

    if not model_files:
        return None

    # Sort by modification time and return the latest
    return max(model_files, key=lambda p: p.stat().st_mtime)


def create_model_link(model_path: Path, link_path: Path) -> None:
    """Create a symlink to the model file.

    Args:
        model_path: Path to the actual model file
        link_path: Path where the symlink should be created
    """
    # Remove existing link if present
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    # Create relative path for the symlink
    try:
        # Try to create a relative symlink
        relative_path = Path("experiments") / model_path.parent.name / model_path.name
        link_path.symlink_to(relative_path)
    except Exception:
        # Fall back to absolute path if relative doesn't work
        link_path.symlink_to(model_path.absolute())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up symlink to latest trained model for CLI tools"
    )
    parser.add_argument(
        "--model", type=str, help="Specific model path to link (default: find latest)"
    )
    parser.add_argument(
        "--link-name",
        type=str,
        default="mnist_model.h5",
        help="Name for the symlink (default: mnist_model.h5)",
    )

    args = parser.parse_args()

    models_dir = Path("models")
    experiments_dir = models_dir / "experiments"

    if args.model:
        # Use specific model path
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Find latest model
        model_path = find_latest_model(experiments_dir)
        if not model_path:
            print(
                "Error: No trained models found in models/experiments/", file=sys.stderr
            )
            print("Train a model first with: ./train_gpu.sh", file=sys.stderr)
            sys.exit(1)

    # Create symlink
    link_path = models_dir / args.link_name

    try:
        create_model_link(model_path, link_path)
        print(f"✅ Created symlink: {link_path} -> {model_path}")
        print("\nYou can now use the CLI tools:")
        print("  python predict_digit.py <image>")
        print("  python batch_predict.py <images>")
    except Exception as e:
        print(f"Error creating symlink: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
