"""CLI module for MC Dropout digit prediction with uncertainty estimation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.mnist_classifier.experiments.mc_dropout import MCDropoutPredictor

from .image_utils import load_and_preprocess_image


def load_model(model_path: Path) -> tf.keras.Model:
    """Load the trained MNIST model.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded Keras model

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        if model_path.is_dir():
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e!s}") from e


def format_output(
    mc_result: dict,
    output_format: str,
) -> str:
    """Format MC Dropout prediction output.

    Args:
        mc_result: Dictionary from MCDropoutResult.to_dict()
        output_format: One of 'verbose', 'quiet', 'json'

    Returns:
        Formatted output string
    """
    if output_format == "quiet":
        return str(mc_result["predicted_class"])

    if output_format == "json":
        return json.dumps(mc_result, indent=2)

    # verbose format (default)
    predicted = mc_result["predicted_class"]
    confidence = mc_result["confidence"]
    entropy = mc_result["uncertainty"]["predictive_entropy"]
    mi = mc_result["uncertainty"]["mutual_information"]
    ci = mc_result["uncertainty"]["confidence_interval_95"]
    n_samples = mc_result["num_samples"]

    # Determine uncertainty level
    if entropy < 0.3:
        level = "LOW"
        level_indicator = "[OK]"
    elif entropy < 1.0:
        level = "MEDIUM"
        level_indicator = "[!]"
    else:
        level = "HIGH"
        level_indicator = "[!!]"

    output_lines = [
        "=" * 60,
        "Monte Carlo Dropout Prediction Results",
        "=" * 60,
        f"\nPredicted digit: {predicted}",
        f"Mean confidence: {confidence:.2%}",
        f"95% CI: [{ci['lower']:.2%}, {ci['upper']:.2%}]",
        f"\nUncertainty Level: {level} {level_indicator}",
        f"  Predictive Entropy: {entropy:.4f}",
        f"  Mutual Information: {mi:.4f}",
        f"  MC Samples: {n_samples}",
        "\nProbabilities with variance:",
    ]

    for digit, prob in mc_result["mean_probabilities"].items():
        var = mc_result["variance_per_class"][digit]
        std = np.sqrt(var)
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        marker = " ← predicted" if int(digit) == predicted else ""
        output_lines.append(f"  {digit}: [{bar}] {prob:.2%} ±{std:.3f}{marker}")

    output_lines.append("\n" + "=" * 60)

    return "\n".join(output_lines)


def main() -> None:
    """Main CLI entry point for MC Dropout prediction."""
    parser = argparse.ArgumentParser(
        description="Classify digits with Monte Carlo Dropout uncertainty estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png
  %(prog)s image.jpg --model models/my_model.h5 --samples 100
  %(prog)s digit.png --output-format json
  %(prog)s digit.png --quiet

Monte Carlo Dropout runs multiple forward passes with dropout enabled
to estimate prediction uncertainty. Higher uncertainty indicates the
model is less confident in its prediction.
        """,
    )

    # Required arguments
    parser.add_argument(
        "image_path", type=str, help="Path to the image file to classify"
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="models/mnist_model.h5",
        help="Path to the trained model (default: models/mnist_model.h5)",
    )

    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=50,
        help="Number of MC Dropout samples (default: 50)",
    )

    parser.add_argument(
        "--output-format",
        choices=["verbose", "quiet", "json"],
        default="verbose",
        help="Output format (default: verbose)",
    )

    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=None,
        help="Maximum predictive entropy threshold (warn if exceeded)",
    )

    args = parser.parse_args()

    try:
        # Convert paths to Path objects
        image_path = Path(args.image_path)
        model_path = Path(args.model)

        # Check if image exists
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}", file=sys.stderr)
            sys.exit(1)

        # Load and preprocess image
        try:
            image_array = load_and_preprocess_image(image_path)
        except ValueError as e:
            print(f"Error: {e!s}", file=sys.stderr)
            sys.exit(1)

        # Load model
        try:
            model = load_model(model_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e!s}", file=sys.stderr)
            sys.exit(1)

        # Create MC Dropout predictor
        mc_predictor = MCDropoutPredictor(model, n_samples=args.samples)

        # Run MC Dropout inference
        mc_result = mc_predictor.predict(image_array)
        result_dict = mc_result.to_dict()

        # Check uncertainty threshold if specified
        threshold = args.uncertainty_threshold
        if threshold is not None and mc_result.predictive_entropy > threshold:
            if args.output_format == "json":
                result_dict["warning"] = "Uncertainty exceeds threshold"
                result_dict["threshold"] = threshold
                print(json.dumps(result_dict, indent=2))
            else:
                print(
                    f"Warning: Uncertainty ({mc_result.predictive_entropy:.4f}) "
                    f"exceeds threshold ({threshold:.4f})",
                    file=sys.stderr,
                )
                output = format_output(result_dict, args.output_format)
                print(output)
            sys.exit(2)

        # Format and print output
        output = format_output(result_dict, args.output_format)
        print(output)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
