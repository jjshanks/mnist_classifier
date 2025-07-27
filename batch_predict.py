#!/usr/bin/env python3
"""
Batch prediction tool for MNIST digit classification.

This script processes multiple image files and outputs predictions
in various formats including CSV.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mnist_classifier.cli.image_utils import (
    load_and_preprocess_image,
    validate_image_file,
)
from mnist_classifier.cli.predict import load_model


def process_images(image_paths: list[Path], model_path: Path) -> list[dict]:
    """Process multiple images and return predictions.

    Args:
        image_paths: List of paths to image files
        model_path: Path to the trained model

    Returns:
        List of prediction results
    """
    # Load model once
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e!s}", file=sys.stderr)
        sys.exit(1)

    results = []

    for image_path in image_paths:
        result = {
            "filename": str(image_path),
            "status": "success",
            "predicted_digit": None,
            "confidence": None,
            "error": None,
        }

        try:
            # Validate image file
            if not validate_image_file(image_path):
                raise ValueError("Invalid or unsupported image format")

            # Load and preprocess image
            image_array = load_and_preprocess_image(image_path)

            # Make prediction
            predictions = model.predict(image_array, verbose=0)
            probabilities = predictions[0]

            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]

            result["predicted_digit"] = int(predicted_class)
            result["confidence"] = float(confidence)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        results.append(result)

    return results


def save_results_csv(results: list[dict], output_path: Path) -> None:
    """Save results to CSV file.

    Args:
        results: List of prediction results
        output_path: Path to save CSV file
    """
    with Path(output_path).open("w", newline="") as csvfile:
        fieldnames = ["filename", "predicted_digit", "confidence", "status", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


def save_results_json(results: list[dict], output_path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: List of prediction results
        output_path: Path to save JSON file
    """
    # Round confidence values for cleaner output
    for result in results:
        if result["confidence"] is not None:
            result["confidence"] = round(result["confidence"], 4)

    with Path(output_path).open("w") as jsonfile:
        json.dump(results, jsonfile, indent=2)


def print_results_table(results: list[dict]) -> None:
    """Print results in a formatted table.

    Args:
        results: List of prediction results
    """
    # Calculate column widths
    max_filename = max(len(r["filename"]) for r in results)
    max_filename = max(max_filename, 8)  # Minimum width for "Filename"

    # Print header
    print(
        f"{'Filename':<{max_filename}} | {'Digit':>5} | "
        f"{'Confidence':>10} | {'Status':>7}"
    )
    print("-" * (max_filename + 30))

    # Print results
    for result in results:
        filename = Path(result["filename"]).name
        if result["status"] == "success":
            digit = str(result["predicted_digit"])
            confidence = f"{result['confidence']:.2%}"
        else:
            digit = "---"
            confidence = "---"

        print(
            f"{filename:<{max_filename}} | {digit:>5} | {confidence:>10} | "
            f"{result['status']:>7}"
        )

    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    print(
        f"\nProcessed {len(results)} images: {successful} successful, "
        f"{len(results) - successful} failed"
    )


def main():  # noqa: PLR0912
    """Main entry point for batch prediction."""
    parser = argparse.ArgumentParser(
        description="Batch prediction tool for MNIST digit classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s images/*.png
  %(prog)s image1.jpg image2.png image3.gif
  %(prog)s images/*.jpg --output results.csv
  %(prog)s images/*.png --output results.json --format json
        """,
    )

    # Required arguments
    parser.add_argument(
        "images", nargs="+", help="Image files to process (supports wildcards)"
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="models/mnist_model",
        help="Path to the trained model (default: models/mnist_model)",
    )

    parser.add_argument(
        "--output", type=str, help="Output file path (CSV or JSON based on format)"
    )

    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for images",
    )

    args = parser.parse_args()

    # Collect image paths
    image_paths = []
    for pattern in args.images:
        path = Path(pattern)

        if path.is_file():
            image_paths.append(path)
        elif path.is_dir():
            # Search directory for images
            if args.recursive:
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]:
                    image_paths.extend(path.rglob(ext))
            else:
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]:
                    image_paths.extend(path.glob(ext))
        else:
            # Try as glob pattern
            parent = path.parent
            glob_pattern = path.name
            if parent.exists():
                image_paths.extend(parent.glob(glob_pattern))

    # Remove duplicates and sort
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print("Error: No image files found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process...")

    # Process images
    model_path = Path(args.model)
    results = process_images(image_paths, model_path)

    # Output results
    if args.format == "csv" or (args.output and args.output.endswith(".csv")):
        if args.output:
            save_results_csv(results, Path(args.output))
            print(f"Results saved to {args.output}")
        else:
            # Print CSV to stdout
            writer = csv.DictWriter(
                sys.stdout,
                fieldnames=[
                    "filename",
                    "predicted_digit",
                    "confidence",
                    "status",
                    "error",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    elif args.format == "json" or (args.output and args.output.endswith(".json")):
        if args.output:
            save_results_json(results, Path(args.output))
            print(f"Results saved to {args.output}")
        else:
            # Print JSON to stdout
            for result in results:
                if result["confidence"] is not None:
                    result["confidence"] = round(result["confidence"], 4)
            print(json.dumps(results, indent=2))

    else:  # table format
        print_results_table(results)

        if args.output:
            # Save as CSV by default for table format with output file
            save_results_csv(results, Path(args.output))
            print(f"\nResults also saved to {args.output}")


if __name__ == "__main__":
    main()
