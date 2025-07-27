"""Image processing utilities for digit classification."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load and preprocess an image for MNIST model prediction.

    This function:
    1. Loads the image using PIL
    2. Converts to grayscale if needed
    3. Resizes to 28x28 pixels
    4. Normalizes pixel values to [0, 1]
    5. Inverts if necessary (dark background with light digit)
    6. Reshapes for model input

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image array with shape (1, 28, 28, 1)

    Raises:
        ValueError: If image cannot be loaded or processed
    """
    try:
        # Load image
        image = Image.open(image_path)

        # Convert to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        image_array = image_array / 255.0

        # Check if image needs inversion
        # MNIST has black background (0) and white digits (1)
        # If the image has more white pixels than black, it likely needs inversion
        if np.mean(image_array) > 0.5:
            image_array = 1.0 - image_array

        # Reshape for model input: (28, 28) -> (1, 28, 28, 1)
        return image_array.reshape(1, 28, 28, 1)

    except Exception as e:
        raise ValueError(f"Failed to load or preprocess image: {e!s}") from e


def preprocess_batch(image_paths: list[Path | str]) -> np.ndarray:
    """Preprocess multiple images for batch prediction.

    Args:
        image_paths: List of paths to image files

    Returns:
        Batch of preprocessed images with shape (n, 28, 28, 1)

    Raises:
        ValueError: If any image cannot be processed
    """
    preprocessed_images = []

    for image_path in image_paths:
        try:
            # Load and preprocess single image
            image_array = load_and_preprocess_image(image_path)
            # Remove batch dimension for stacking
            preprocessed_images.append(image_array[0])
        except ValueError as e:
            raise ValueError(f"Error processing {image_path}: {e!s}") from e

    # Stack into batch
    return np.array(preprocessed_images)


def save_preprocessed_image(image_array: np.ndarray, output_path: str | Path) -> None:
    """Save a preprocessed image for debugging purposes.

    Args:
        image_array: Preprocessed image array (28, 28) or (1, 28, 28, 1)
        output_path: Path to save the image
    """
    # Handle different input shapes
    if image_array.shape == (1, 28, 28, 1):
        image_array = image_array[0, :, :, 0]
    elif image_array.shape == (28, 28, 1):
        image_array = image_array[:, :, 0]

    # Convert back to 0-255 range
    image_array = (image_array * 255).astype(np.uint8)

    # Create PIL image and save
    image = Image.fromarray(image_array, mode="L")
    image.save(output_path)


def validate_image_file(image_path: str | Path) -> bool:
    """Validate that a file is a supported image format.

    Args:
        image_path: Path to the image file

    Returns:
        True if the file is a valid image, False otherwise
    """
    supported_formats = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

    path = Path(image_path)

    # Check if file exists
    if not path.exists():
        return False

    # Check if it's a file (not directory)
    if not path.is_file():
        return False

    # Check file extension
    if path.suffix.lower() not in supported_formats:
        return False

    # Try to open the file to verify it's a valid image
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
