"""
Example usage of the complete MNIST data pipeline.

This script demonstrates how to use the data pipeline
in a real training scenario.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .data_pipeline import MNISTDataPipeline


def visualize_preprocessed_batch(x_batch: npt.NDArray[np.float32], y_batch: npt.NDArray[np.float32], n_samples: int = 16) -> None:
    """Visualize a batch of preprocessed images."""
    n_show = min(n_samples, len(x_batch))
    n_cols = int(np.sqrt(n_show))
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(n_show):
        # Remove channel dimension for visualization
        img = x_batch[i].squeeze()
        label = np.argmax(y_batch[i])

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("data/preprocessed_samples.png")
    print("Saved visualization to data/preprocessed_samples.png")


def main() -> None:
    """Main example workflow."""
    print("MNIST Data Pipeline Example")
    print("=" * 50)

    # 1. Initialize pipeline
    pipeline = MNISTDataPipeline()

    # 2. Prepare data
    print("\nPreparing data...")
    data = pipeline.prepare_data(
        validation_split=0.1, normalize_method="standard", use_cache=True
    )

    # 3. Extract components
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    # 4. Print shapes and info
    print("\nData shapes:")
    print(f"  Train: {x_train.shape}, {y_train.shape}")
    print(f"  Val: {x_val.shape}, {y_val.shape}")
    print(f"  Test: {x_test.shape}, {y_test.shape}")

    # 5. Verify preprocessing
    print("\nPreprocessing verification:")
    print(f"  Pixel range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  Labels are one-hot: {y_train.shape[1] == 10}")

    # 6. Create batch generator
    batch_gen = pipeline.get_batch_generator(x_train, y_train, batch_size=32)
    x_batch, y_batch = next(batch_gen)
    print("\nBatch generator test:")
    print(f"  Batch shapes: {x_batch.shape}, {y_batch.shape}")

    # 7. Visualize preprocessed data
    print("\nVisualizing preprocessed data...")
    visualize_preprocessed_batch(x_batch, y_batch)

    # 8. Simulate model training
    print("\nSimulating model training loop:")
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")

        # Train for a few batches
        for batch_idx in range(5):
            x_batch, y_batch = next(batch_gen)
            # Here you would do: loss = model.train_on_batch(x_batch, y_batch)
            print(f"  Batch {batch_idx + 1}: shape {x_batch.shape}")

    print("\n✅ Pipeline demonstration complete!")
    print("\nYou can now use this pipeline with your neural network:")
    print("  - data['x_train'] and data['y_train'] for training")
    print("  - data['x_val'] and data['y_val'] for validation")
    print("  - data['x_test'] and data['y_test'] for final evaluation")


if __name__ == "__main__":
    main()
