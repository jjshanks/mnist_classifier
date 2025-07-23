#!/usr/bin/env python3
"""
Verification script for Milestone 2 completion.
Checks all requirements and provides detailed feedback.
"""

import importlib.util
import json
import sys
from pathlib import Path


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def check_files_exist():
    """Check if all required files exist."""
    print("\n📁 Checking required files:")

    required_files = [
        ("src/mnist_classifier/__init__.py", "Package init file"),
        ("src/mnist_classifier/load_data.py", "Data loading module"),
        ("src/mnist_classifier/preprocess.py", "Preprocessing module"),
        ("src/mnist_classifier/data_pipeline.py", "Integrated pipeline"),
        ("notebooks/01_data_exploration.ipynb", "Exploration notebook"),
    ]

    all_exist = True
    for filepath, description in required_files:
        exists = Path(filepath).exists()
        print(f"  {check_mark(exists)} {description}: {filepath}")
        if not exists:
            all_exist = False

    return all_exist


def check_imports():
    """Check if all modules can be imported."""
    print("\n📦 Checking module imports:")

    modules = [
        "src.mnist_classifier.load_data",
        "src.mnist_classifier.preprocess",
        "src.mnist_classifier.data_pipeline",
    ]

    all_importable = True
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  {check_mark(True)} {module_name}")
        except Exception as e:
            print(f"  {check_mark(False)} {module_name}: {e!s}")
            all_importable = False

    return all_importable


def check_data_loading():
    """Test data loading functionality."""
    print("\n💾 Testing data loading:")

    try:
        from src.mnist_classifier.load_data import load_mnist

        (x_train, y_train), (x_test, y_test) = load_mnist()

        checks = [
            (x_train.shape == (60000, 28, 28), f"Training data shape: {x_train.shape}"),
            (y_train.shape == (60000,), f"Training labels shape: {y_train.shape}"),
            (x_test.shape == (10000, 28, 28), f"Test data shape: {x_test.shape}"),
            (y_test.shape == (10000,), f"Test labels shape: {y_test.shape}"),
        ]

        all_good = True
        for condition, description in checks:
            print(f"  {check_mark(condition)} {description}")
            if not condition:
                all_good = False

        return all_good

    except Exception as e:
        print(f"  {check_mark(False)} Failed to load data: {e!s}")
        return False


def check_preprocessing():
    """Test preprocessing functionality."""
    print("\n🔧 Testing preprocessing:")

    try:
        import numpy as np

        from src.mnist_classifier.preprocess import (
            normalize_pixels,
            one_hot_encode_labels,
            reshape_images,
        )

        # Test with sample data
        test_images = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
        test_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Test normalization
        normalized = normalize_pixels(test_images)
        norm_ok = normalized.min() >= 0 and normalized.max() <= 1
        print(
            f"  {check_mark(norm_ok)} Normalization: "
            f"[{normalized.min():.3f}, {normalized.max():.3f}]"
        )

        # Test reshaping
        reshaped = reshape_images(normalized)
        reshape_ok = reshaped.shape == (10, 28, 28, 1)
        print(f"  {check_mark(reshape_ok)} Reshaping: {reshaped.shape}")

        # Test one-hot encoding
        one_hot = one_hot_encode_labels(test_labels)
        onehot_ok = one_hot.shape == (10, 10) and np.all(one_hot.sum(axis=1) == 1)
        print(f"  {check_mark(onehot_ok)} One-hot encoding: {one_hot.shape}")

        return norm_ok and reshape_ok and onehot_ok

    except Exception as e:
        print(f"  {check_mark(False)} Preprocessing failed: {e!s}")
        return False


def check_pipeline():
    """Test integrated pipeline."""
    print("\n🔄 Testing integrated pipeline:")

    try:
        from src.mnist_classifier.data_pipeline import MNISTDataPipeline

        pipeline = MNISTDataPipeline()
        data = pipeline.prepare_data(validation_split=0.1, force_reload=True)

        checks = [
            ("x_train" in data, "Training data present"),
            ("x_val" in data, "Validation data present"),
            ("x_test" in data, "Test data present"),
            (
                data["x_train"].shape[0] + data["x_val"].shape[0] == 60000,
                "Train/val split correct",
            ),
            (
                data["x_train"].min() >= 0 and data["x_train"].max() <= 1,
                "Data normalized correctly",
            ),
        ]

        all_good = True
        for condition, description in checks:
            print(f"  {check_mark(condition)} {description}")
            if not condition:
                all_good = False

        return all_good

    except Exception as e:
        print(f"  {check_mark(False)} Pipeline failed: {e!s}")
        return False


def check_notebook():
    """Check if Jupyter notebook exists and has content."""
    print("\n📓 Checking Jupyter notebook:")

    notebook_path = "notebooks/01_data_exploration.ipynb"

    if not Path(notebook_path).exists():
        print(f"  {check_mark(False)} Notebook not found")
        return False

    try:
        with Path(notebook_path).open() as f:
            notebook = json.load(f)

        n_cells = len(notebook.get("cells", []))
        has_code = any(
            cell["cell_type"] == "code" for cell in notebook.get("cells", [])
        )

        print(f"  {check_mark(n_cells > 0)} Notebook has {n_cells} cells")
        print(f"  {check_mark(has_code)} Notebook contains code cells")

        return n_cells > 5 and has_code

    except Exception as e:
        print(f"  {check_mark(False)} Could not read notebook: {e!s}")
        return False


def check_generated_files():
    """Check for generated files from running the pipeline."""
    print("\n📄 Checking generated files:")

    expected_files = [
        ("data/processed/preprocessed_data.pkl", "Cached preprocessed data"),
        ("data/processed/preprocessing_params.json", "Preprocessing parameters"),
        ("data/data_summary.txt", "Data summary report"),
    ]

    all_exist = True
    for filepath, description in expected_files:
        exists = Path(filepath).exists()
        print(f"  {check_mark(exists)} {description}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("🔍 Milestone 2 Verification")
    print("=" * 50)

    # Make sure we're in the right directory
    if not Path("src/mnist_classifier").exists():
        print("❌ Error: Not in project root directory!")
        print("Please run this from the mnist_classifier project root.")
        sys.exit(1)

    # Add project root to path
    sys.path.insert(0, str(Path.cwd()))

    # Run all checks
    checks = [
        ("Required Files", check_files_exist()),
        ("Module Imports", check_imports()),
        ("Data Loading", check_data_loading()),
        ("Preprocessing", check_preprocessing()),
        ("Pipeline Integration", check_pipeline()),
        ("Jupyter Notebook", check_notebook()),
        ("Generated Files", check_generated_files()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 2 is complete!")
        print("\nYour data pipeline is ready. You can now:")
        print("  - Load and preprocess MNIST data efficiently")
        print("  - Visualize and explore the dataset")
        print("  - Use the pipeline for model training")
        print("\nNext: Milestone 3 - Build and train the neural network!")
    else:
        print("\n⚠️  Some checks failed. Please review and fix the issues above.")
        print("\nTips:")
        print("  - Make sure your virtual environment is activated")
        print("  - Run the data_pipeline.py script to generate cache files")
        print("  - Check that all module imports are correct")


if __name__ == "__main__":
    main()
