"""
MNIST Data Loading Module

This module handles downloading and loading the MNIST dataset.
The data is automatically cached after first download.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_directory(data_dir: str = "data/") -> str:
    """
    Create data directory if it doesn't exist.

    Args:
        data_dir: Path to data directory

    Returns:
        Absolute path to data directory
    """
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        data_path.mkdir(parents=True)
        logger.info(f"Created data directory: {data_path}")
    return str(data_path)


def load_mnist(
    data_dir: str = "data/",
) -> tuple[
    tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]],
    tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]],
]:
    """
    Load MNIST dataset and return train/test splits.

    This function:
    1. Downloads MNIST if not already cached
    2. Loads the data into memory
    3. Returns properly formatted numpy arrays

    Args:
        data_dir: Directory to cache the downloaded data

    Returns:
        Tuple of (x_train, y_train), (x_test, y_test) where:
        - x_train: Training images, shape (60000, 28, 28), dtype uint8, values 0-255
        - y_train: Training labels, shape (60000,), dtype uint8, values 0-9
        - x_test: Test images, shape (10000, 28, 28), dtype uint8, values 0-255
        - y_test: Test labels, shape (10000,), dtype uint8, values 0-9

    Example:
        >>> (x_train, y_train), (x_test, y_test) = load_mnist()
        >>> print(f"Training data shape: {x_train.shape}")
        >>> print(f"Training labels shape: {y_train.shape}")
    """
    # Create data directory
    create_data_directory(data_dir)

    logger.info("Loading MNIST dataset...")

    try:
        # Load MNIST using Keras
        # This automatically downloads to ~/.keras/datasets/ if not present
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Log dataset information
        logger.info("Successfully loaded MNIST dataset")
        logger.info(f"Training set: {x_train.shape[0]} samples")
        logger.info(f"Test set: {x_test.shape[0]} samples")
        logger.info(f"Image shape: {x_train.shape[1:]} (height, width)")
        logger.info(f"Pixel value range: {x_train.min()} to {x_train.max()}")

        # Verify data integrity
        assert x_train.shape == (
            60000,
            28,
            28,
        ), f"Unexpected training data shape: {x_train.shape}"
        assert y_train.shape == (
            60000,
        ), f"Unexpected training labels shape: {y_train.shape}"
        assert x_test.shape == (
            10000,
            28,
            28,
        ), f"Unexpected test data shape: {x_test.shape}"
        assert y_test.shape == (10000,), f"Unexpected test labels shape: {y_test.shape}"

        # Verify label ranges
        assert y_train.min() >= 0, "Training labels min out of range"
        assert y_train.max() <= 9, "Training labels max out of range"
        assert y_test.min() >= 0, "Test labels min out of range"
        assert y_test.max() <= 9, "Test labels max out of range"

        return (x_train, y_train), (x_test, y_test)

    except Exception as e:
        logger.error(f"Failed to load MNIST dataset: {e!s}")
        raise


def get_dataset_info(
    x_train: npt.NDArray[np.uint8],
    y_train: npt.NDArray[np.uint8],
    x_test: npt.NDArray[np.uint8],
    y_test: npt.NDArray[np.uint8],
) -> dict[str, Any]:
    """
    Get comprehensive information about the dataset.

    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels

    Returns:
        Dictionary containing dataset statistics
    """
    # Count samples per class
    train_class_counts = np.bincount(y_train)
    test_class_counts = np.bincount(y_test)

    return {
        "num_classes": 10,
        "image_shape": x_train.shape[1:],
        "train_samples": x_train.shape[0],
        "test_samples": x_test.shape[0],
        "total_samples": x_train.shape[0] + x_test.shape[0],
        "pixel_dtype": x_train.dtype,
        "pixel_range": (x_train.min(), x_train.max()),
        "train_class_distribution": train_class_counts.tolist(),
        "test_class_distribution": test_class_counts.tolist(),
        "memory_usage_mb": {
            "train_images": x_train.nbytes / 1024 / 1024,
            "train_labels": y_train.nbytes / 1024 / 1024,
            "test_images": x_test.nbytes / 1024 / 1024,
            "test_labels": y_test.nbytes / 1024 / 1024,
            "total": (x_train.nbytes + y_train.nbytes + x_test.nbytes + y_test.nbytes)
            / 1024
            / 1024,
        },
    }


def save_sample_images(
    x_data: npt.NDArray[np.uint8],
    y_data: npt.NDArray[np.uint8],
    num_samples: int = 10,
    output_dir: str = "data/samples/",
) -> None:
    """
    Save sample images to disk for inspection.

    Args:
        x_data: Image data
        y_data: Label data
        num_samples: Number of samples to save
        output_dir: Directory to save samples
    """
    from PIL import Image

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save random samples
    indices = np.random.choice(len(x_data), num_samples, replace=False)

    for i, idx in enumerate(indices):
        img = Image.fromarray(x_data[idx])
        label = y_data[idx]
        filename = f"sample_{i:02d}_label_{label}.png"
        filepath = Path(output_dir) / filename
        img.save(str(filepath))
        logger.info(f"Saved sample image: {filename}")


if __name__ == "__main__":
    """
    Test the data loading functionality.
    Run this script directly to verify everything works.
    """
    print("Testing MNIST data loader...")
    print("-" * 50)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Get and display dataset info
    info = get_dataset_info(x_train, y_train, x_test, y_test)

    print("\n📊 Dataset Information:")
    print(f"Total samples: {info['total_samples']:,}")
    print(f"Training samples: {info['train_samples']:,}")
    print(f"Test samples: {info['test_samples']:,}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Image shape: {info['image_shape']}")
    print(f"Pixel range: {info['pixel_range']}")
    print(f"Memory usage: {info['memory_usage_mb']['total']:.2f} MB")

    print("\n📈 Class distribution (training set):")
    for digit, count in enumerate(info["train_class_distribution"]):
        bar = "█" * int(count / 1000)
        print(f"  {digit}: {bar} {count:,}")

    # Save some sample images
    print("\n💾 Saving sample images...")
    save_sample_images(x_train, y_train, num_samples=10)
    print("Sample images saved to data/samples/")

    print("\n✅ Data loading test completed successfully!")
