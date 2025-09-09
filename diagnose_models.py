#!/usr/bin/env python3
"""
Diagnose model loading issues.
"""

import os
from pathlib import Path

print("🔍 Diagnosing Model Loading")
print("=" * 50)

# Check current directory
print(f"\nCurrent directory: {os.getcwd()}")

# Find all model files
print("\n📁 Available model files:")
model_paths = [
    Path("models/experiments"),
    Path("models"),
]

h5_files = []
for base_path in model_paths:
    if base_path.exists():
        # Look for final_model.h5 or best_model.h5 files
        h5_files.extend(base_path.glob("**/final_model.h5"))
        h5_files.extend(base_path.glob("**/best_model.h5"))
        # Also check in models directory
        h5_files.extend(base_path.glob("mnist_model.h5"))
        h5_files.extend(base_path.glob("mnist_cnn_model.h5"))

# Filter out weights files
h5_files = [f for f in h5_files if "weights" not in f.name.lower()]

if h5_files:
    print(f"\nFound {len(h5_files)} model files:")
    for i, f in enumerate(
        sorted(h5_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    ):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i+1}. {f} ({size_mb:.1f} MB)")

    latest_model = max(h5_files, key=lambda p: p.stat().st_mtime)
    print(f"\n✅ Latest model: {latest_model}")

    # Test loading
    print("\n🧪 Testing model loading...")
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU for testing

        from src.mnist_classifier.models.cnn_model import get_activation_model

        print(f"Loading: {latest_model}")
        model, viz_model = get_activation_model(str(latest_model))
        print("✅ Models loaded successfully!")
        print(f"   Original model: {model.name}")
        print(f"   Visualization model: {viz_model.name}")
        print(f"   Outputs: {len(viz_model.outputs)} layers")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
else:
    print("❌ No model files found!")
    print("\nPlease train a model first:")
    print("  python train_model.py")
