#!/usr/bin/env python3
"""
Create test images for the MNIST digit classifier.

This script generates sample digit images from the MNIST test set
or creates synthetic digit images for testing the CLI.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def create_synthetic_digit(
    digit: int, size: int = 200, background: str = "white", foreground: str = "black"
) -> Image.Image:
    """Create a synthetic image of a digit.

    Args:
        digit: Digit to create (0-9)
        size: Size of the square image
        background: Background color
        foreground: Foreground (digit) color

    Returns:
        PIL Image object
    """
    # Create image
    image = Image.new("L", (size, size), background)
    draw = ImageDraw.Draw(image)

    # Try to use a nice font, fall back to default if not available
    font_size = int(size * 0.7)
    try:
        # Try to use a monospace font
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size
        )
    except Exception:
        try:
            # Try another common font
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            # Use default font
            font = ImageFont.load_default()
            # Scale digit drawing manually for default font
            font_size = 20

    # Get text bounding box
    text = str(digit)
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # Fallback for older PIL versions
        text_width, text_height = draw.textsize(text, font=font)

    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2

    # Draw the digit
    draw.text((x, y), text, fill=foreground, font=font)

    return image


def create_from_mnist(output_dir: Path, count: int = 10) -> None:
    """Create test images from MNIST dataset.

    Args:
        output_dir: Directory to save images
        count: Number of images to create (max 10 per digit)
    """
    try:
        import tensorflow as tf

        # Load MNIST data
        print("Loading MNIST dataset...")
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Create images for each digit
        for digit in range(10):
            # Find indices of this digit in test set
            digit_indices = np.where(y_test == digit)[0]

            # Select up to 'count' examples
            selected_indices = digit_indices[: min(count, len(digit_indices))]

            for i, idx in enumerate(selected_indices):
                # Get image
                image_array = x_test[idx]

                # Convert to PIL Image
                image = Image.fromarray(image_array, mode="L")

                # Save with descriptive filename
                filename = output_dir / f"mnist_digit_{digit}_{i+1}.png"
                image.save(filename)
                print(f"Created {filename}")

    except ImportError:
        print("TensorFlow not found. Creating synthetic images instead.")
        create_synthetic_images(output_dir, count)
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        print("Creating synthetic images instead.")
        create_synthetic_images(output_dir, count)


def create_synthetic_images(output_dir: Path, count: int = 3) -> None:
    """Create synthetic test images.

    Args:
        output_dir: Directory to save images
        count: Number of variations per digit
    """
    variations = [
        {"background": "white", "foreground": "black", "suffix": "normal"},
        {"background": "black", "foreground": "white", "suffix": "inverted"},
        {"background": "gray", "foreground": "black", "suffix": "gray_bg"},
    ]

    for digit in range(10):
        for _, var in enumerate(variations[:count]):
            # Create synthetic digit
            image = create_synthetic_digit(
                digit, background=var["background"], foreground=var["foreground"]
            )

            # Resize to 28x28 for more realistic testing
            image_small = image.resize((28, 28), Image.Resampling.LANCZOS)

            # Save both sizes
            filename_large = (
                output_dir / f"synthetic_digit_{digit}_{var['suffix']}_large.png"
            )
            filename_small = output_dir / f"synthetic_digit_{digit}_{var['suffix']}.png"

            image.save(filename_large)
            image_small.save(filename_small)

            print(f"Created {filename_small} and {filename_large}")


def create_edge_cases(output_dir: Path) -> None:
    """Create edge case test images.

    Args:
        output_dir: Directory to save images
    """
    # Blank image
    blank = Image.new("L", (28, 28), "white")
    blank.save(output_dir / "edge_case_blank.png")
    print("Created edge_case_blank.png")

    # Very faint digit
    faint = Image.new("L", (28, 28), 250)  # Almost white
    draw = ImageDraw.Draw(faint)
    draw.text((10, 5), "5", fill=240)  # Very light gray
    faint.save(output_dir / "edge_case_faint.png")
    print("Created edge_case_faint.png")

    # Noisy image
    noise = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    noisy = Image.fromarray(noise, mode="L")
    noisy.save(output_dir / "edge_case_noise.png")
    print("Created edge_case_noise.png")

    # Multiple digits (should confuse the model)
    multi = Image.new("L", (56, 28), "white")
    draw = ImageDraw.Draw(multi)
    draw.text((5, 5), "2", fill="black")
    draw.text((35, 5), "7", fill="black")
    multi.save(output_dir / "edge_case_multiple.png")
    print("Created edge_case_multiple.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create test images for MNIST digit classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates three types of test images:
  1. Real MNIST images from the test dataset
  2. Synthetic digit images with various styles
  3. Edge cases to test robustness

Examples:
  %(prog)s test_images/
  %(prog)s test_images/ --synthetic --count 5
  %(prog)s test_images/ --mnist --count 10
  %(prog)s test_images/ --all
        """,
    )

    parser.add_argument("output_dir", type=str, help="Directory to save test images")

    parser.add_argument(
        "--mnist", action="store_true", help="Create images from MNIST dataset"
    )

    parser.add_argument(
        "--synthetic", action="store_true", help="Create synthetic digit images"
    )

    parser.add_argument(
        "--edge-cases", action="store_true", help="Create edge case test images"
    )

    parser.add_argument(
        "--all", action="store_true", help="Create all types of test images"
    )

    parser.add_argument(
        "--count", type=int, default=3, help="Number of images per digit (default: 3)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to create
    if args.all:
        create_mnist = True
        create_synthetic = True
        create_edges = True
    else:
        create_mnist = args.mnist
        create_synthetic = args.synthetic
        create_edges = args.edge_cases

        # If nothing specified, default to synthetic
        if not any([create_mnist, create_synthetic, create_edges]):
            create_synthetic = True

    # Create requested images
    if create_mnist:
        print("\nCreating MNIST test images...")
        create_from_mnist(output_dir, args.count)

    if create_synthetic:
        print("\nCreating synthetic test images...")
        create_synthetic_images(output_dir, min(args.count, 3))

    if create_edges:
        print("\nCreating edge case test images...")
        create_edge_cases(output_dir)

    print(f"\nTest images created in {output_dir}/")
    print("You can now test the classifier with:")
    print(f"  python predict_digit.py {output_dir}/<image_file>")
    print(f"  python batch_predict.py {output_dir}/*.png")


if __name__ == "__main__":
    main()
