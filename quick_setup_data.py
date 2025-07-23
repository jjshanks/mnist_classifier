#!/usr/bin/env python3
"""
Quick setup script for Milestone 2 data pipeline.
Run this to quickly test that everything is working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np

from src.mnist_classifier.data_pipeline import MNISTDataPipeline


def main():
    print("🚀 Quick Data Setup for MNIST Classifier")
    print("=" * 50)

    # Initialize and prepare data
    pipeline = MNISTDataPipeline()
    data = pipeline.prepare_data(validation_split=0.1)

    # Quick visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(10):
        # Find first occurrence of digit i
        labels = np.argmax(data["y_train"], axis=1)
        idx = np.where(labels == i)[0][0]

        axes[i].imshow(data["x_train"][idx].squeeze(), cmap="gray")
        axes[i].set_title(f"Digit: {i}")
        axes[i].axis("off")

    plt.suptitle("One Example of Each Digit (Preprocessed)")
    plt.tight_layout()
    plt.savefig("data/all_digits_sample.png")

    print("\n✅ Data pipeline setup complete!")
    print("📊 Saved digit samples to: data/all_digits_sample.png")
    print("📁 Cached data in: data/processed/")
    print("\nReady for Milestone 3: Neural Network Implementation!")


if __name__ == "__main__":
    main()
