#!/usr/bin/env python3
"""
Monte Carlo Dropout Demonstration Script

This script demonstrates the Monte Carlo Dropout uncertainty estimation
feature of the MNIST classifier. It loads a trained model and runs
MC Dropout inference on sample images from the MNIST test set.

Usage:
    python demo_mc_dropout.py [--samples N] [--images N]

Example:
    python demo_mc_dropout.py --samples 100 --images 5
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.mnist_classifier.experiments.mc_dropout import (
    MCDropoutPredictor,
    get_uncertainty_level,
)


def find_model() -> Path:
    """Find the latest trained model."""
    model_paths = [
        Path("models/experiments"),
        Path("models"),
    ]
    h5_files = []
    for base_path in model_paths:
        if base_path.exists():
            h5_files.extend(base_path.glob("**/final_model.h5"))
            h5_files.extend(base_path.glob("**/best_model.h5"))
            h5_files.extend(base_path.glob("mnist_model.h5"))

    h5_files = [f for f in h5_files if "weights" not in f.name.lower()]

    if not h5_files:
        raise FileNotFoundError("No trained model found. Run: ./train_gpu.sh --quick")

    return max(h5_files, key=lambda p: p.stat().st_mtime)


def run_demo(args: argparse.Namespace) -> None:
    """Run the MC Dropout demo."""
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel("ERROR")

    # Find model
    model_path = Path(args.model) if args.model else find_model()

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded: {model.name}\n")

    # Load MNIST test data
    print("Loading MNIST test data...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Create MC Dropout predictor
    print(f"Creating MC Dropout predictor with {args.samples} samples...")
    mc_predictor = MCDropoutPredictor(model, n_samples=args.samples)
    print()

    # Select random test images
    np.random.seed(42)
    indices = np.random.choice(len(x_test), args.images, replace=False)

    print(f"Evaluating {args.images} test images...")
    print("-" * 70)

    results = []
    for i, idx in enumerate(indices):
        image = x_test[idx : idx + 1]
        true_label = y_test[idx]

        # Run MC Dropout
        result = mc_predictor.predict(image)
        results.append((true_label, result))

        # Print results
        correct = "OK" if result.predicted_class == true_label else "WRONG"
        level = get_uncertainty_level(result)
        ci_low = result.confidence_interval[0]
        ci_high = result.confidence_interval[1]

        print(f"\nImage {i + 1} (index {idx}):")
        print(f"  True label: {true_label}")
        print(f"  Prediction: {result.predicted_class} [{correct}]")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"  Entropy: {result.predictive_entropy:.4f} ({level.upper()})")
        print(f"  Mutual Info: {result.mutual_information:.4f}")

    print_summary(results, args.images)

    if args.save_plots:
        save_plots(results)


def print_summary(results: list, n_images: int) -> None:
    """Print summary statistics."""
    print("\n" + "-" * 70)
    print("Summary Statistics")
    print("-" * 70)

    correct = sum(1 for true, res in results if res.predicted_class == true)
    avg_confidence = np.mean([r.confidence for _, r in results])
    avg_entropy = np.mean([r.predictive_entropy for _, r in results])
    avg_mi = np.mean([r.mutual_information for _, r in results])

    low_unc = sum(1 for _, r in results if get_uncertainty_level(r) == "low")
    med_unc = sum(1 for _, r in results if get_uncertainty_level(r) == "medium")
    high_unc = sum(1 for _, r in results if get_uncertainty_level(r) == "high")

    print(f"\nAccuracy: {correct}/{n_images} ({100 * correct / n_images:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.1%}")
    print(f"Average Entropy: {avg_entropy:.4f}")
    print(f"Average Mutual Info: {avg_mi:.4f}")
    print("\nUncertainty Distribution:")
    print(f"  Low:    {low_unc} ({100 * low_unc / n_images:.1f}%)")
    print(f"  Medium: {med_unc} ({100 * med_unc / n_images:.1f}%)")
    print(f"  High:   {high_unc} ({100 * high_unc / n_images:.1f}%)")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("  - Run web app with MC Dropout: python run_web_app.py")
    print("  - Use CLI: python -m src.mnist_classifier.cli.mc_dropout_predict img.png")
    print("  - API endpoint: POST /predict/mc-dropout")


def save_plots(results: list) -> None:
    """Save visualization plots."""
    print("\nSaving visualization plots...")

    output_dir = Path("outputs/mc_dropout_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (true_label, result) in enumerate(results):
        _, ax = plt.subplots(figsize=(10, 6))
        std_devs = [np.sqrt(v) for v in result.prediction_variance]

        bars = ax.bar(
            range(10),
            result.mean_prediction,
            yerr=std_devs,
            capsize=5,
            color="steelblue",
            edgecolor="black",
            ecolor="darkred",
            alpha=0.8,
        )
        bars[result.predicted_class].set_color("darkgreen")

        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ci_low = result.confidence_interval[0]
        ci_high = result.confidence_interval[1]
        ax.set_title(
            f"MC Dropout (True: {true_label}, Pred: {result.predicted_class})\n"
            f"Entropy: {result.predictive_entropy:.4f}, "
            f"95% CI: [{ci_low:.1%}, {ci_high:.1%}]"
        )
        ax.set_xticks(range(10))
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"image_{i + 1}_uncertainty.png", dpi=150)
        plt.close()

    print(f"Plots saved to: {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demonstrate Monte Carlo Dropout uncertainty estimation"
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=50,
        help="Number of MC Dropout samples per prediction (default: 50)",
    )
    parser.add_argument(
        "--images",
        "-i",
        type=int,
        default=5,
        help="Number of test images to evaluate (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (auto-detects if not specified)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save visualization plots to files",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Monte Carlo Dropout Uncertainty Estimation Demo")
    print("=" * 70)
    print()

    try:
        run_demo(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
