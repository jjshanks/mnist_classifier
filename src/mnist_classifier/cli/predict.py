"""Main CLI module for digit prediction."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

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
        # Try loading as SavedModel format first
        if model_path.is_dir():
            model = tf.keras.models.load_model(model_path)
        else:
            # Try loading as .keras file
            model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e!s}") from e


def predict_digit(
    model: tf.keras.Model, image_array: np.ndarray
) -> tuple[int, float, np.ndarray]:
    """Predict digit from preprocessed image.

    Args:
        model: Trained Keras model
        image_array: Preprocessed image array (1, 28, 28, 1)

    Returns:
        Tuple of (predicted_digit, confidence, all_probabilities)
    """
    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    probabilities = predictions[0]

    # Get predicted class and confidence
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    return int(predicted_class), float(confidence), probabilities


def format_output(
    predicted_digit: int,
    confidence: float,
    probabilities: np.ndarray,
    output_format: str,
) -> str:
    """Format prediction output based on requested format.

    Args:
        predicted_digit: Predicted digit (0-9)
        confidence: Confidence score for prediction
        probabilities: Array of probabilities for all digits
        output_format: One of 'verbose', 'quiet', 'json'

    Returns:
        Formatted output string
    """
    if output_format == "quiet":
        return str(predicted_digit)

    if output_format == "json":
        output_dict = {
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 4),
            "probabilities": {
                str(i): round(float(prob), 4) for i, prob in enumerate(probabilities)
            },
        }
        return json.dumps(output_dict, indent=2)

    # verbose format (default)
    output_lines = [
        f"Predicted digit: {predicted_digit}",
        f"Confidence: {confidence:.2%}",
        "\nProbabilities for each digit:",
    ]

    for digit, prob in enumerate(probabilities):
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        output_lines.append(f"  {digit}: [{bar}] {prob:.2%}")

    return "\n".join(output_lines)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classify handwritten digits using a trained MNIST model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png
  %(prog)s image.jpg --model models/my_model.keras
  %(prog)s digit.png --output-format json
  %(prog)s digit.png --quiet
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
        "--output-format",
        choices=["verbose", "quiet", "json"],
        default="verbose",
        help="Output format (default: verbose)",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence threshold for prediction (0.0-1.0)",
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

        # Make prediction
        predicted_digit, confidence, probabilities = predict_digit(model, image_array)

        # Check confidence threshold
        if confidence < args.confidence_threshold:
            if args.output_format == "json":
                output = json.dumps(
                    {
                        "error": "Low confidence",
                        "predicted_digit": predicted_digit,
                        "confidence": round(confidence, 4),
                        "threshold": args.confidence_threshold,
                    },
                    indent=2,
                )
                print(output)
            else:
                print(
                    f"Error: Prediction confidence ({confidence:.2%}) below "
                    f"threshold ({args.confidence_threshold:.2%})",
                    file=sys.stderr,
                )
            sys.exit(2)

        # Format and print output
        output = format_output(
            predicted_digit, confidence, probabilities, args.output_format
        )
        print(output)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
