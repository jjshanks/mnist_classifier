"""
Visualize and analyze model architecture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.mnist_classifier.models.cnn_model import (
    count_parameters,
    create_cnn_model,
    create_simple_model,
)


def visualize_layer_outputs():
    """Visualize intermediate layer outputs."""
    from tensorflow import keras

    from src.mnist_classifier.data_pipeline import MNISTDataPipeline

    # Load model and data
    model = create_cnn_model()
    pipeline = MNISTDataPipeline()
    data = pipeline.prepare_data()

    # Get a sample image
    sample_img = data["x_test"][0:1]  # First test image
    sample_label = np.argmax(data["y_test"][0])

    # Create model for layer outputs
    layer_names = ["conv1", "conv2", "conv3"]
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

    # Get activations
    activations = activation_model.predict(sample_img)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    # Original image
    axes[0].imshow(sample_img[0].squeeze(), cmap="gray")
    axes[0].set_title(f"Original (Label: {sample_label})")
    axes[0].axis("off")

    # Feature maps
    for i, (name, activation) in enumerate(zip(layer_names, activations, strict=False)):
        ax = axes[i + 1]
        # Show first 16 filters as 4x4 grid
        n_features = min(16, activation.shape[-1])
        size = int(np.sqrt(n_features))

        display_grid = np.zeros(
            (size * activation.shape[1], size * activation.shape[2])
        )

        for row in range(size):
            for col in range(size):
                channel_idx = row * size + col
                if channel_idx < activation.shape[-1]:
                    feature = activation[0, :, :, channel_idx]
                    # Normalize
                    feature = (feature - feature.mean()) / (feature.std() + 1e-5)
                    display_grid[
                        row * activation.shape[1] : (row + 1) * activation.shape[1],
                        col * activation.shape[2] : (col + 1) * activation.shape[2],
                    ] = feature

        ax.imshow(display_grid, cmap="viridis")
        ax.set_title(f"{name} (first {n_features} filters)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("models/layer_visualizations.png", dpi=150, bbox_inches="tight")
    plt.show()


def compare_architectures():
    """Compare different model architectures."""
    # Create models
    standard_model = create_cnn_model()
    simple_model = create_simple_model()

    # Get parameters
    standard_params = count_parameters(standard_model)
    simple_params = count_parameters(simple_model)

    # Create comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Parameter comparison
    models = ["Standard CNN", "Simple CNN"]
    params = [standard_params["total"], simple_params["total"]]

    ax1.bar(models, params, color=["blue", "green"])
    ax1.set_ylabel("Total Parameters")
    ax1.set_title("Model Size Comparison")

    # Add values on bars
    for i, v in enumerate(params):
        ax1.text(i, v + 1000, f"{v:,}", ha="center")

    # Layer count comparison
    layer_counts = [len(standard_model.layers), len(simple_model.layers)]

    ax2.bar(models, layer_counts, color=["blue", "green"])
    ax2.set_ylabel("Number of Layers")
    ax2.set_title("Model Complexity Comparison")

    # Add values on bars
    for i, v in enumerate(layer_counts):
        ax2.text(i, v + 0.5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig("models/architecture_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print detailed comparison
    print("\nDetailed Architecture Comparison:")
    print("=" * 50)
    print("Standard CNN:")
    print(f"  Total parameters: {standard_params['total']:,}")
    print(f"  Trainable: {standard_params['trainable']:,}")
    print(f"  Layers: {len(standard_model.layers)}")
    print("\nSimple CNN:")
    print(f"  Total parameters: {simple_params['total']:,}")
    print(f"  Trainable: {simple_params['trainable']:,}")
    print(f"  Layers: {len(simple_model.layers)}")
    print(
        f"\nParameter Reduction: {standard_params['total'] / simple_params['total']:.1f}x"
    )


if __name__ == "__main__":
    print("🎨 Model Architecture Visualization")
    print("=" * 50)

    # Run visualizations
    print("\n1. Comparing architectures...")
    compare_architectures()

    print("\n2. Visualizing layer outputs...")
    print("   (This requires a trained model)")
    try:
        visualize_layer_outputs()
    except Exception as e:
        print(f"   Skipped: {e}")

    print("\n✅ Visualization complete!")
