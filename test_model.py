#!/usr/bin/env python3
"""
Test a trained MNIST model on individual images or the test set.

Usage:
    python test_model.py                           # Test on test set
    python test_model.py --image path/to/digit.png # Test on single image
    python test_model.py --model path/to/model.h5  # Use specific model
"""

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# Configure GPU environment before importing TensorFlow
from src.mnist_classifier.utils import setup_gpu_environment

setup_gpu_environment()

from tensorflow import keras  # noqa: E402

from src.mnist_classifier.data_pipeline import MNISTDataPipeline  # noqa: E402
from src.mnist_classifier.preprocess import (  # noqa: E402
    normalize_pixels,
    reshape_images,
)


def load_latest_model(
    model_dir: str = "models/experiments",
) -> tuple[keras.Model, Path]:
    """Load the most recent trained model."""
    model_dir = Path(model_dir)

    # Find latest experiment
    experiments = sorted(model_dir.glob("mnist_*"), key=lambda p: p.stat().st_mtime)

    if not experiments:
        raise FileNotFoundError(f"No experiments found in {model_dir}")

    latest = experiments[-1]
    model_path = latest / "final_model.h5"

    if not model_path.exists():
        model_path = latest / "checkpoints" / "best_model.h5"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {latest}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    return model, latest


def test_on_image(model: keras.Model, image_path: str):
    """Test model on a single image."""
    # Load and preprocess image
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Resize to 28x28 if needed
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to array
    img_array = np.array(img)

    # Preprocess
    img_normalized = normalize_pixels(img_array.reshape(1, 28, 28))
    img_input = reshape_images(img_normalized, add_channel=True)

    # Predict
    predictions = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Show image
    ax1.imshow(img_array, cmap="gray")
    ax1.set_title(f"Input Image\nPredicted: {predicted_class} ({confidence:.1%})")
    ax1.axis("off")

    # Show predictions
    ax2.bar(range(10), predictions[0])
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction Probabilities")
    ax2.set_xticks(range(10))

    # Highlight prediction
    ax2.patches[predicted_class].set_color("green")

    plt.tight_layout()
    plt.show()

    # Print top 3 predictions
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    print("\nTop 3 predictions:")
    for i, idx in enumerate(top3_idx):
        print(f"  {i+1}. Digit {idx}: {predictions[0][idx]:.1%}")


def test_on_test_set(model: keras.Model):
    """Evaluate model on entire test set."""
    # Load test data
    print("Loading test data...")
    pipeline = MNISTDataPipeline()
    data = pipeline.prepare_data()

    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(data["x_test"], data["y_test"], batch_size=256, verbose=1)

    # Get metric names
    metric_names = model.metrics_names

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)

    for name, value in zip(metric_names, results, strict=False):
        if "accuracy" in name:
            print(f"{name}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{name}: {value:.4f}")

    # Analyze errors
    print("\nAnalyzing misclassifications...")
    predictions = model.predict(data["x_test"], verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(data["y_test"], axis=1)

    # Find errors
    errors = y_pred != y_true
    error_indices = np.where(errors)[0]

    print(f"\nTotal errors: {len(error_indices)} out of {len(y_true)}")
    print(f"Error rate: {len(error_indices)/len(y_true):.2%}")

    # Confusion analysis
    if len(error_indices) > 0:
        print("\nMost common confusions:")
        confusions = []
        for idx in error_indices:
            confusions.append(f"{y_true[idx]}→{y_pred[idx]}")

        confusion_counts = Counter(confusions).most_common(5)
        for confusion, count in confusion_counts:
            print(f"  {confusion}: {count} times")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test trained MNIST model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: use latest)",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image file to test"
    )
    parser.add_argument(
        "--experiment", type=str, default=None, help="Specific experiment folder name"
    )

    args = parser.parse_args()

    print("🧪 MNIST Model Testing")
    print("=" * 50)

    try:
        # Load model
        if args.model:
            print(f"Loading specified model: {args.model}")
            model = keras.models.load_model(args.model)
            exp_dir = Path(args.model).parent
        else:
            model, exp_dir = load_latest_model()

        print(f"Model loaded from: {exp_dir.name}")
        print(
            f"Model summary: {len(model.layers)} layers, "
            f"{model.count_params():,} parameters"
        )

        # Test mode
        if args.image:
            print(f"\nTesting on image: {args.image}")
            test_on_image(model, args.image)
        else:
            test_on_test_set(model)

            # Show sample predictions
            print("\nWould you like to see sample predictions? (y/n): ", end="")
            if input().lower().strip() == "y":
                # Load one batch for visualization
                pipeline = MNISTDataPipeline()
                data = pipeline.prepare_data()

                # Random samples
                indices = np.random.choice(len(data["x_test"]), 9)
                samples = data["x_test"][indices]
                true_labels = np.argmax(data["y_test"][indices], axis=1)

                # Predict
                predictions = model.predict(samples, verbose=0)
                pred_labels = np.argmax(predictions, axis=1)

                # Display
                fig, axes = plt.subplots(3, 3, figsize=(10, 10))
                axes = axes.ravel()

                for i, (img, true, pred) in enumerate(
                    zip(samples, true_labels, pred_labels, strict=False)
                ):
                    ax = axes[i]
                    ax.imshow(img.squeeze(), cmap="gray")

                    color = "green" if true == pred else "red"
                    confidence = predictions[i][pred] * 100

                    ax.set_title(
                        f"True: {true}, Pred: {pred}\n" f"Conf: {confidence:.1f}%",
                        color=color,
                    )
                    ax.axis("off")

                plt.suptitle("Sample Test Predictions", fontsize=16)
                plt.tight_layout()
                plt.show()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
