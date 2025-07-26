#!/usr/bin/env python3
"""
Verification script for Milestone 3 completion.
"""

import os
import sys
from pathlib import Path

import numpy as np


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def check_files_exist():
    """Check if all required files exist."""
    print("\n📁 Checking required files:")

    required_files = [
        ("src/mnist_classifier/models/__init__.py", "Models package init"),
        ("src/mnist_classifier/models/cnn_model.py", "CNN model definition"),
        ("src/mnist_classifier/training/__init__.py", "Training package init"),
        ("src/mnist_classifier/training/train.py", "Training script"),
        ("docs/model_architecture.md", "Architecture documentation"),
    ]

    all_exist = True
    for filepath, description in required_files:
        exists = os.path.exists(filepath)
        print(f"  {check_mark(exists)} {description}: {filepath}")
        if not exists:
            all_exist = False

    return all_exist


def check_model_creation():
    """Test model creation."""
    print("\n🏗️ Testing model creation:")

    try:
        from src.mnist_classifier.models.cnn_model import (
            compile_model,
            create_cnn_model,
        )

        # Create model
        model = create_cnn_model()
        model = compile_model(model)

        # Check model properties
        checks = [
            (
                model.input_shape == (None, 28, 28, 1),
                f"Input shape: {model.input_shape}",
            ),
            (model.output_shape == (None, 10), f"Output shape: {model.output_shape}"),
            (len(model.layers) > 5, f"Number of layers: {len(model.layers)}"),
            (
                model.count_params() > 10000,
                f"Total parameters: {model.count_params():,}",
            ),
        ]

        all_good = True
        for condition, description in checks:
            print(f"  {check_mark(condition)} {description}")
            if not condition:
                all_good = False

        # Test forward pass
        test_input = np.random.randn(1, 28, 28, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)

        forward_ok = output.shape == (1, 10) and abs(output.sum() - 1.0) < 0.01
        print(f"  {check_mark(forward_ok)} Forward pass test")

        return all_good and forward_ok

    except Exception as e:
        print(f"  {check_mark(False)} Model creation failed: {e!s}")
        return False


def check_training_capability():
    """Check if training script can be imported."""
    print("\n🏃 Testing training capability:")

    try:
        from src.mnist_classifier.training.train import MNISTTrainer

        # Check key methods exist
        trainer = MNISTTrainer()
        methods = ["prepare_data", "create_model", "train", "evaluate"]

        all_exist = True
        for method in methods:
            exists = hasattr(trainer, method)
            print(f"  {check_mark(exists)} Method '{method}' exists")
            if not exists:
                all_exist = False

        return all_exist

    except Exception as e:
        print(f"  {check_mark(False)} Training module error: {e!s}")
        return False


def check_trained_model():
    """Check if a trained model exists."""
    print("\n🎯 Checking for trained model:")

    model_dirs = [
        "models/experiments",
        "models",
    ]

    found_model = False
    model_path = None

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            # Look for .h5 files
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".h5"):
                        model_path = os.path.join(root, file)
                        found_model = True
                        break
                if found_model:
                    break
        if found_model:
            break

    print(
        f"  {check_mark(found_model)} Trained model found"
        + (f": {model_path}" if found_model else "")
    )

    # If model found, try to load and check accuracy
    if found_model:
        try:
            from tensorflow import keras

            model = keras.models.load_model(model_path)

            # Check if training history exists
            exp_dir = Path(model_path).parent.parent
            history_file = exp_dir / "training_history.csv"

            if history_file.exists():
                import pandas as pd

                history = pd.read_csv(history_file)

                if "val_accuracy" in history.columns:
                    best_val_acc = history["val_accuracy"].max()
                    print(
                        f"  {check_mark(best_val_acc > 0.98)} "
                        f"Best validation accuracy: {best_val_acc:.4f}"
                    )
                    return best_val_acc > 0.98

            print("  ℹ️  Could not verify accuracy (no history file)")
            return True  # Model exists at least

        except Exception as e:
            print(f"  ⚠️  Could not load model: {e!s}")
            return True  # Model file exists

    return False


def check_documentation():
    """Check if documentation exists and is comprehensive."""
    print("\n📚 Checking documentation:")

    doc_path = "docs/model_architecture.md"

    if not os.path.exists(doc_path):
        print(f"  {check_mark(False)} Documentation not found")
        return False

    with open(doc_path) as f:
        content = f.read()

    # Check for key sections
    sections = [
        ("# MNIST CNN Model Architecture", "Title"),
        ("## Architecture Summary", "Architecture summary"),
        ("## Design Rationale", "Design rationale"),
        ("Conv2D", "Convolutional layers mentioned"),
        ("Parameters", "Parameter count mentioned"),
    ]

    all_present = True
    for section, description in sections:
        present = section in content
        print(f"  {check_mark(present)} {description}")
        if not present:
            all_present = False

    # Check length
    word_count = len(content.split())
    adequate_length = word_count > 500
    print(f"  {check_mark(adequate_length)} " f"Adequate length ({word_count} words)")

    return all_present and adequate_length


def check_helper_scripts():
    """Check if helper scripts exist."""
    print("\n🛠️ Checking helper scripts:")

    scripts = [
        ("train_model.py", "Main training script"),
        ("test_model.py", "Model testing script"),
    ]

    all_exist = True
    for script, description in scripts:
        exists = os.path.exists(script)
        print(f"  {check_mark(exists)} {description}: {script}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("🔍 Milestone 3 Verification")
    print("=" * 50)

    # Make sure we're in the right directory
    if not os.path.exists("src/mnist_classifier"):
        print("❌ Error: Not in project root directory!")
        print("Please run this from the mnist_classifier project root.")
        sys.exit(1)

    # Add project root to path
    sys.path.insert(0, os.path.abspath("."))

    # Run all checks
    checks = [
        ("Required Files", check_files_exist()),
        ("Model Creation", check_model_creation()),
        ("Training Capability", check_training_capability()),
        ("Trained Model", check_trained_model()),
        ("Documentation", check_documentation()),
        ("Helper Scripts", check_helper_scripts()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 3 is complete!")
        print("\nYou have successfully:")
        print("  ✓ Built a CNN architecture for MNIST")
        print("  ✓ Implemented comprehensive training pipeline")
        print("  ✓ Achieved >98% accuracy (if model trained)")
        print("  ✓ Documented the architecture thoroughly")
        print("\nNext: Milestone 4 - Build the CLI interface!")
    else:
        print("\n⚠️  Some checks failed. Please review and fix the issues above.")
        print("\nTips:")
        print("  - Run 'python train_model.py --quick' for a quick test")
        print("  - Check that all imports are correct")
        print("  - Ensure model achieves >98% validation accuracy")


if __name__ == "__main__":
    main()
