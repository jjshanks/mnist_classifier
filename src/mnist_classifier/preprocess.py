"""
Data Preprocessing Module for MNIST

This module provides functions to prepare MNIST data for neural network training.
All transformations are designed to be reversible for visualization purposes.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def normalize_pixels(images: npt.NDArray[np.uint8], method: str = "standard") -> npt.NDArray[np.float32]:
    """
    Normalize pixel values to prepare for neural network input.

    Args:
        images: Input images with pixel values in [0, 255]
        method: Normalization method
            - 'standard': Scale to [0, 1] by dividing by 255
            - 'centered': Scale to [-1, 1] by dividing by 127.5 and subtracting 1

    Returns:
        Normalized images as float32

    Example:
        >>> images = np.array([[[0, 128, 255]]])  # Sample pixels
        >>> normalized = normalize_pixels(images)
        >>> print(normalized)  # [[0.0, 0.502, 1.0]]
    """
    # Ensure float32 for neural network compatibility
    images = images.astype(np.float32)

    if method == "standard":
        # Scale to [0, 1] - most common for images
        normalized = images / 255.0
        logger.info(f"Normalized {len(images)} images to [0, 1] range")

    elif method == "centered":
        # Scale to [-1, 1] - sometimes better for certain architectures
        normalized = (images - 127.5) / 127.5
        logger.info(f"Normalized {len(images)} images to [-1, 1] range")

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Verify output range
    if method == "standard":
        assert (
            normalized.min() >= 0 and normalized.max() <= 1
        ), f"Normalization failed: range [{normalized.min()}, {normalized.max()}]"

    return normalized


def denormalize_pixels(images: npt.NDArray[np.float32], method: str = "standard") -> npt.NDArray[np.uint8]:
    """
    Reverse normalization for visualization.

    Args:
        images: Normalized images
        method: Same method used for normalization

    Returns:
        Images with pixel values in [0, 255] as uint8
    """
    if method == "standard":
        denormalized = images * 255.0
    elif method == "centered":
        denormalized = (images + 1) * 127.5
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Clip values and convert to uint8
    denormalized = np.clip(denormalized, 0, 255)
    return denormalized.astype(np.uint8)


def reshape_images(images: npt.NDArray[np.float32], add_channel: bool = True) -> npt.NDArray[np.float32]:
    """
    Reshape images for CNN input.

    Args:
        images: Input images of shape (n_samples, height, width)
        add_channel: If True, add channel dimension for CNN

    Returns:
        Reshaped images
        - If add_channel=True: (n_samples, height, width, 1)
        - If add_channel=False: (n_samples, height * width)

    Example:
        >>> images = np.zeros((100, 28, 28))  # 100 MNIST images
        >>> reshaped = reshape_images(images, add_channel=True)
        >>> print(reshaped.shape)  # (100, 28, 28, 1)
    """
    n_samples = images.shape[0]

    if add_channel:
        # Add channel dimension for CNN
        if len(images.shape) == 3:
            # Images are (n, h, w), add channel to make (n, h, w, 1)
            reshaped = images.reshape(n_samples, 28, 28, 1)
            logger.info(f"Reshaped images from {images.shape} to {reshaped.shape}")
        else:
            # Images already have channel dimension
            reshaped = images
    else:
        # Flatten for traditional neural networks
        reshaped = images.reshape(n_samples, -1)
        logger.info(f"Flattened images from {images.shape} to {reshaped.shape}")

    return reshaped


def one_hot_encode_labels(labels: npt.NDArray[np.int32], num_classes: int = 10) -> npt.NDArray[np.float32]:
    """
    Convert integer labels to one-hot encoded vectors.

    One-hot encoding transforms categorical labels into binary vectors where
    only one element is 1 (hot) and the rest are 0 (cold).

    Args:
        labels: Integer labels of shape (n_samples,)
        num_classes: Total number of classes

    Returns:
        One-hot encoded labels of shape (n_samples, num_classes)

    Example:
        >>> labels = np.array([0, 1, 2, 3])
        >>> one_hot = one_hot_encode_labels(labels, num_classes=4)
        >>> print(one_hot)
        [[1 0 0 0]   # 0 → [1, 0, 0, 0]
         [0 1 0 0]   # 1 → [0, 1, 0, 0]
         [0 0 1 0]   # 2 → [0, 0, 1, 0]
         [0 0 0 1]]  # 3 → [0, 0, 0, 1]
    """
    # Ensure labels are integers
    labels = labels.astype(np.int32)

    # Check label validity
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Labels must be in range [0, {num_classes-1}], "
            f"got [{labels.min()}, {labels.max()}]"
        )

    # Create one-hot encoding
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1

    logger.info(f"One-hot encoded {len(labels)} labels into {one_hot.shape}")

    # Verify encoding
    assert np.all(one_hot.sum(axis=1) == 1), "One-hot encoding failed"

    return one_hot


def decode_one_hot_labels(one_hot: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
    """
    Convert one-hot encoded labels back to integers.

    Args:
        one_hot: One-hot encoded labels of shape (n_samples, num_classes)

    Returns:
        Integer labels of shape (n_samples,)
    """
    return np.argmax(one_hot, axis=1).astype(np.int64)  # type: ignore[no-any-return]


def create_train_validation_split(
    x_data: npt.NDArray[np.float32],
    y_data: npt.NDArray[np.int32],
    validation_split: float = 0.1,
    random_seed: int | None = 42,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """
    Split training data into train and validation sets.

    Args:
        x_data: Input features
        y_data: Labels
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (x_train, y_train, x_val, y_val)

    Example:
        >>> x = np.random.rand(1000, 28, 28)
        >>> y = np.random.randint(0, 10, 1000)
        >>> x_train, y_train, x_val, y_val = create_train_validation_split(x, y, 0.2)
        >>> print(len(x_train), len(x_val))  # 800, 200
    """
    if not 0 < validation_split < 1:
        raise ValueError(
            f"validation_split must be between 0 and 1, got {validation_split}"
        )

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Get number of samples
    n_samples = len(x_data)
    n_val = int(n_samples * validation_split)

    # Create shuffled indices
    indices = np.random.permutation(n_samples)

    # Split indices
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Create splits
    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    x_val = x_data[val_indices]
    y_val = y_data[val_indices]

    logger.info(
        f"Created train/validation split: "
        f"{len(x_train)} train, {len(x_val)} validation samples"
    )

    # Verify class distribution in both sets
    train_classes = np.unique(y_train)
    val_classes = np.unique(y_val)

    if len(train_classes) != len(val_classes):
        logger.warning(
            f"Class imbalance detected: "
            f"train has {len(train_classes)} classes, "
            f"val has {len(val_classes)} classes"
        )

    return x_train, y_train, x_val, y_val


def prepare_data_for_training(
    x_train: npt.NDArray[np.uint8],
    y_train: npt.NDArray[np.int32],
    x_test: npt.NDArray[np.uint8],
    y_test: npt.NDArray[np.int32],
    validation_split: float = 0.1,
    normalize_method: str = "standard",
) -> dict[str, Any]:
    """
    Complete preprocessing pipeline for MNIST data.

    This function applies all necessary preprocessing steps:
    1. Normalization
    2. Reshaping for CNN
    3. One-hot encoding
    4. Train/validation split

    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        validation_split: Fraction for validation
        normalize_method: Normalization method

    Returns:
        Dictionary with preprocessed data
    """
    logger.info("Starting complete preprocessing pipeline...")

    # 1. Normalize pixel values
    x_train_norm = normalize_pixels(x_train, method=normalize_method)
    x_test_norm = normalize_pixels(x_test, method=normalize_method)

    # 2. Reshape for CNN input
    x_train_reshaped = reshape_images(x_train_norm, add_channel=True)
    x_test_reshaped = reshape_images(x_test_norm, add_channel=True)

    # 3. Create train/validation split
    x_train_final, y_train_final, x_val, y_val = create_train_validation_split(
        x_train_reshaped, y_train, validation_split=validation_split
    )

    # 4. One-hot encode labels
    y_train_encoded = one_hot_encode_labels(y_train_final)
    y_val_encoded = one_hot_encode_labels(y_val)
    y_test_encoded = one_hot_encode_labels(y_test)

    # Package results
    preprocessed_data = {
        "x_train": x_train_final,
        "y_train": y_train_encoded,
        "x_val": x_val,
        "y_val": y_val_encoded,
        "x_test": x_test_reshaped,
        "y_test": y_test_encoded,
        "preprocessing_params": {
            "normalize_method": normalize_method,
            "validation_split": validation_split,
            "input_shape": x_train_final.shape[1:],
            "num_classes": y_train_encoded.shape[1],
        },
    }

    logger.info("Preprocessing complete!")
    logger.info(
        f"Final shapes - Train: {x_train_final.shape}, "
        f"Val: {x_val.shape}, Test: {x_test_reshaped.shape}"
    )

    return preprocessed_data


if __name__ == "__main__":
    """
    Test preprocessing functions with sample data.
    """
    print("Testing preprocessing functions...")
    print("-" * 50)

    # Create sample data
    sample_images = np.random.randint(0, 256, size=(5, 28, 28), dtype=np.uint8)
    sample_labels = np.array([0, 1, 2, 3, 4])

    print("1. Testing normalization:")
    normalized = normalize_pixels(sample_images)
    print(f"   Original range: [{sample_images.min()}, {sample_images.max()}]")
    print(f"   Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print("   ✓ Normalization successful!")

    print("\n2. Testing reshaping:")
    reshaped = reshape_images(normalized)
    print(f"   Original shape: {normalized.shape}")
    print(f"   Reshaped shape: {reshaped.shape}")
    print("   ✓ Reshaping successful!")

    print("\n3. Testing one-hot encoding:")
    one_hot = one_hot_encode_labels(sample_labels, num_classes=10)
    print(f"   Original labels: {sample_labels}")
    print(f"   One-hot shape: {one_hot.shape}")
    print("   Sample encoding for label 3:")
    print(f"   {one_hot[3]}")
    print("   ✓ One-hot encoding successful!")

    print("\n4. Testing train/validation split:")
    x_train, y_train, x_val, y_val = create_train_validation_split(
        reshaped, sample_labels, validation_split=0.4
    )
    print(f"   Training samples: {len(x_train)}")
    print(f"   Validation samples: {len(x_val)}")
    print("   ✓ Split successful!")

    print("\n✅ All preprocessing functions working correctly!")
