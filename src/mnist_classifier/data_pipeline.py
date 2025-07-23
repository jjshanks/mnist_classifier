"""
Integrated Data Pipeline for MNIST

This module combines data loading and preprocessing into a single,
easy-to-use pipeline for training neural networks.
"""

import json
import logging
import pickle
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .load_data import get_dataset_info, load_mnist
from .preprocess import (
    normalize_pixels,
    prepare_data_for_training,
    reshape_images,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNISTDataPipeline:
    """
    Complete data pipeline for MNIST digit classification.

    This class encapsulates all data operations:
    - Loading
    - Preprocessing
    - Saving/loading preprocessed data
    - Providing data for training

    Example:
        >>> pipeline = MNISTDataPipeline()
        >>> data = pipeline.prepare_data(validation_split=0.1)
        >>> x_train, y_train = data['x_train'], data['y_train']
    """

    def __init__(self, data_dir: str = "data/", cache_dir: str = "data/processed/"):
        """
        Initialize the data pipeline.

        Args:
            data_dir: Directory for raw data
            cache_dir: Directory for preprocessed data cache
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.data_loaded = False
        self.preprocessing_params: dict[str, Any] | None = None

        # Create directories if needed
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def prepare_data(
        self,
        validation_split: float = 0.1,
        normalize_method: str = "standard",
        use_cache: bool = True,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        """
        Prepare complete dataset for training.

        Args:
            validation_split: Fraction for validation set
            normalize_method: Normalization method
            use_cache: Whether to use cached preprocessed data
            force_reload: Force reprocessing even if cache exists

        Returns:
            Dictionary with all preprocessed data and parameters
        """
        cache_file = Path(self.cache_dir) / "preprocessed_data.pkl"
        params_file = Path(self.cache_dir) / "preprocessing_params.json"

        # Try to load from cache
        if use_cache and not force_reload and cache_file.exists():
            logger.info("Loading preprocessed data from cache...")
            return self._load_from_cache(cache_file, params_file)

        # Load raw data
        logger.info("Loading raw MNIST data...")
        (x_train, y_train), (x_test, y_test) = load_mnist(self.data_dir)

        # Get dataset info
        info = get_dataset_info(x_train, y_train, x_test, y_test)
        logger.info(f"Dataset loaded: {info['total_samples']} total samples")

        # Apply preprocessing
        logger.info("Applying preprocessing pipeline...")
        preprocessed_data = prepare_data_for_training(
            x_train,
            y_train.astype(np.int32),
            x_test,
            y_test.astype(np.int32),
            validation_split=validation_split,
            normalize_method=normalize_method,
        )

        # Add dataset info
        preprocessed_data["dataset_info"] = info

        # Save to cache if requested
        if use_cache:
            self._save_to_cache(preprocessed_data, cache_file, params_file)

        self.data_loaded = True
        self.preprocessing_params = preprocessed_data["preprocessing_params"]

        return preprocessed_data

    def _save_to_cache(
        self, data: dict[str, Any], cache_file: Path, params_file: Path
    ) -> None:
        """Save preprocessed data to cache."""
        logger.info(f"Saving preprocessed data to {cache_file}...")

        # Save data as pickle
        with cache_file.open("wb") as f:
            pickle.dump(data, f)

        # Save parameters as JSON for readability
        with params_file.open("w") as f:
            json.dump(data["preprocessing_params"], f, indent=2)

        logger.info("Cache saved successfully!")

    def _load_from_cache(self, cache_file: Path, params_file: Path) -> dict[str, Any]:
        """Load preprocessed data from cache."""
        with cache_file.open("rb") as f:
            data = pickle.load(f)

        with params_file.open() as f:
            params = json.load(f)

        # Verify parameters match
        if data["preprocessing_params"] != params:
            logger.warning("Cache parameters mismatch! Reprocessing...")
            raise ValueError("Cache invalid")

        logger.info("Cache loaded successfully!")
        self.data_loaded = True
        self.preprocessing_params = params

        return data  # type: ignore[no-any-return]

    def get_batch_generator(
        self,
        x_data: npt.NDArray[np.float32],
        y_data: npt.NDArray[np.float32],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Generator[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], None, None]:
        """
        Create a generator for mini-batch training.

        Args:
            x_data: Input features
            y_data: Labels
            batch_size: Size of each batch
            shuffle: Whether to shuffle data each epoch

        Yields:
            Tuples of (x_batch, y_batch)
        """
        n_samples = len(x_data)
        indices = np.arange(n_samples)

        while True:
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                yield x_data[batch_indices], y_data[batch_indices]

    def prepare_single_image(
        self, image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        """
        Preprocess a single image for inference.

        Args:
            image: Raw image of shape (28, 28) with values 0-255

        Returns:
            Preprocessed image ready for model input
        """
        if self.preprocessing_params is None:
            raise ValueError("Pipeline not initialized. Call prepare_data() first.")

        # Apply same preprocessing as training
        normalized = normalize_pixels(
            image.reshape(1, 28, 28),
            method=self.preprocessing_params["normalize_method"],
        )
        return reshape_images(normalized, add_channel=True)

    def get_sample_batch(
        self, n_samples: int = 32
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Get a sample batch for testing or visualization.

        Args:
            n_samples: Number of samples

        Returns:
            Tuple of (images, labels)
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call prepare_data() first.")

        # This would need access to the loaded data
        # Implementation depends on how you store the data
        _ = n_samples  # Acknowledge unused parameter for future implementation
        raise NotImplementedError("get_sample_batch not yet implemented")


def create_data_summary_report(
    data: dict[str, Any], output_file: str = "data/data_summary.txt"
) -> None:
    """
    Create a comprehensive summary report of the processed data.

    Args:
        data: Preprocessed data dictionary
        output_file: Where to save the report
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write("MNIST Data Pipeline Summary Report\n")
        f.write("=" * 50 + "\n\n")

        # Dataset info
        info = data["dataset_info"]
        f.write("Dataset Information:\n")
        f.write(f"  Total samples: {info['total_samples']:,}\n")
        f.write(f"  Image shape: {info['image_shape']}\n")
        f.write(f"  Number of classes: {info['num_classes']}\n")
        f.write(f"  Memory usage: {info['memory_usage_mb']['total']:.2f} MB\n\n")

        # Preprocessing parameters
        params = data["preprocessing_params"]
        f.write("Preprocessing Parameters:\n")
        f.write(f"  Normalization method: {params['normalize_method']}\n")
        f.write(f"  Validation split: {params['validation_split']:.1%}\n")
        f.write(f"  Input shape for model: {params['input_shape']}\n")
        f.write(f"  Output classes: {params['num_classes']}\n\n")

        # Data splits
        f.write("Data Splits:\n")
        f.write(f"  Training samples: {data['x_train'].shape[0]:,}\n")
        f.write(f"  Validation samples: {data['x_val'].shape[0]:,}\n")
        f.write(f"  Test samples: {data['x_test'].shape[0]:,}\n\n")

        # Class distribution
        f.write("Class Distribution (Training Set):\n")
        train_labels = np.argmax(data["y_train"], axis=1)
        for digit in range(10):
            count = np.sum(train_labels == digit)
            percentage = count / len(train_labels) * 100
            f.write(f"  Digit {digit}: {count:,} ({percentage:.1f}%)\n")

    logger.info(f"Data summary report saved to {output_file}")


def verify_data_pipeline() -> dict[str, Any]:
    """
    Comprehensive test of the entire data pipeline.
    """
    print("🧪 Testing Complete Data Pipeline")
    print("=" * 50)

    # Initialize pipeline
    pipeline = MNISTDataPipeline()

    # Test 1: Basic data preparation
    print("\n1. Testing basic data preparation...")
    data = pipeline.prepare_data(validation_split=0.15, force_reload=True)
    print("   ✓ Data prepared successfully")
    print(f"   ✓ Training shape: {data['x_train'].shape}")
    print(f"   ✓ Validation shape: {data['x_val'].shape}")
    print(f"   ✓ Test shape: {data['x_test'].shape}")

    # Test 2: Cache functionality
    print("\n2. Testing cache functionality...")
    import time

    start_time = time.time()
    pipeline.prepare_data(validation_split=0.15)
    cache_time = time.time() - start_time
    print(f"   ✓ Cache loading took {cache_time:.2f} seconds")

    # Test 3: Single image preprocessing
    print("\n3. Testing single image preprocessing...")
    test_image = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
    processed = pipeline.prepare_single_image(test_image)
    print(f"   ✓ Single image processed: {test_image.shape} → {processed.shape}")

    # Test 4: Batch generator
    print("\n4. Testing batch generator...")
    gen = pipeline.get_batch_generator(data["x_train"], data["y_train"], batch_size=64)
    x_batch, y_batch = next(gen)
    print(f"   ✓ Batch shapes: X={x_batch.shape}, Y={y_batch.shape}")

    # Test 5: Data summary report
    print("\n5. Creating data summary report...")
    create_data_summary_report(data)
    print("   ✓ Report saved to data/data_summary.txt")

    print("\n✅ All pipeline tests passed!")
    return data


if __name__ == "__main__":
    # Run comprehensive pipeline test
    data = verify_data_pipeline()

    # Print final summary
    print("\n" + "=" * 50)
    print("📊 Pipeline ready for model training!")
    print("Next steps:")
    print("  1. Design CNN architecture")
    print("  2. Train model using prepared data")
    print("  3. Evaluate on test set")
