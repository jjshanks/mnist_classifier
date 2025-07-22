# Milestone 2: Data Acquisition and Preparation - Detailed Implementation Guide

## Overview

Welcome to Milestone 2! In this milestone, you'll learn how to work with the famous MNIST dataset - a collection of 70,000 handwritten digit images that has become the "Hello World" of machine learning. You'll download the data, explore it visually, and prepare it for training a neural network.

### What You'll Learn
- **Data Loading**: How to efficiently load and manage ML datasets
- **Data Exploration**: Techniques for understanding your data through visualization
- **Preprocessing**: Essential transformations to prepare data for neural networks
- **Data Splitting**: Creating proper train/validation/test sets
- **Best Practices**: Industry-standard approaches to data handling

### The MNIST Dataset
MNIST (Modified National Institute of Standards and Technology) contains:
- 60,000 training images
- 10,000 test images
- Each image is 28×28 pixels (grayscale)
- Labels from 0-9 representing the digit in each image

Fun fact: MNIST was created by Yann LeCun and has been used to benchmark ML algorithms since 1998!

### Prerequisites
Before starting, ensure you have:
- ✅ Completed Milestone 1 (all directories, dependencies installed)
- ✅ Virtual environment activated (you should see `(.venv)` in your prompt)
- ✅ Basic Python knowledge (functions, imports, numpy arrays)
- ✅ About 45-60 minutes to complete all tasks

### Success Criteria
By the end of this milestone, you will have:
- ✅ A robust data loading module that downloads and caches MNIST
- ✅ Interactive visualizations showing sample digits and statistics
- ✅ Preprocessing functions that prepare data for neural networks
- ✅ Proper train/validation/test splits
- ✅ Well-documented, reusable code

---

## Task 2.1: Create Data Loading Script

### Why This Matters
Data loading is the foundation of any ML project. A good data loader should:
- Download data automatically if not present
- Cache data to avoid re-downloading
- Handle errors gracefully
- Return data in a consistent format
- Be memory efficient

### Understanding TensorFlow's Data Loading
TensorFlow provides built-in datasets through `tf.keras.datasets`. This is convenient because:
- Handles downloading and caching automatically
- Returns data in NumPy array format
- Splits data into train/test sets
- Widely used and well-tested

### Step-by-Step Instructions

1. **Create the source directory structure**:
   ```bash
   # Make sure you're in the project root
   cd mnist_classifier

   # Create the package structure
   mkdir -p src/mnist_classifier
   touch src/__init__.py
   touch src/mnist_classifier/__init__.py
   ```

2. **Create the data loader module**:
   ```bash
   touch src/mnist_classifier/load_data.py
   ```

3. **Implement the data loading functionality**:

   Add this code to `src/mnist_classifier/load_data.py`:

   ```python
   """
   MNIST Data Loading Module

   This module handles downloading and loading the MNIST dataset.
   The data is automatically cached after first download.
   """

   import os
   import logging
   from typing import Tuple, Optional
   import numpy as np
   import tensorflow as tf

   # Configure logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)


   def create_data_directory(data_dir: str = 'data/') -> str:
       """
       Create data directory if it doesn't exist.

       Args:
           data_dir: Path to data directory

       Returns:
           Absolute path to data directory
       """
       abs_data_dir = os.path.abspath(data_dir)
       if not os.path.exists(abs_data_dir):
           os.makedirs(abs_data_dir)
           logger.info(f"Created data directory: {abs_data_dir}")
       return abs_data_dir


   def load_mnist(data_dir: str = 'data/') -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                     Tuple[np.ndarray, np.ndarray]]:
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
       abs_data_dir = create_data_directory(data_dir)

       logger.info("Loading MNIST dataset...")

       try:
           # Load MNIST using Keras
           # This automatically downloads to ~/.keras/datasets/ if not present
           (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

           # Log dataset information
           logger.info(f"Successfully loaded MNIST dataset")
           logger.info(f"Training set: {x_train.shape[0]} samples")
           logger.info(f"Test set: {x_test.shape[0]} samples")
           logger.info(f"Image shape: {x_train.shape[1:]} (height, width)")
           logger.info(f"Pixel value range: {x_train.min()} to {x_train.max()}")

           # Verify data integrity
           assert x_train.shape == (60000, 28, 28), f"Unexpected training data shape: {x_train.shape}"
           assert y_train.shape == (60000,), f"Unexpected training labels shape: {y_train.shape}"
           assert x_test.shape == (10000, 28, 28), f"Unexpected test data shape: {x_test.shape}"
           assert y_test.shape == (10000,), f"Unexpected test labels shape: {y_test.shape}"

           # Verify label ranges
           assert y_train.min() >= 0 and y_train.max() <= 9, "Training labels out of range"
           assert y_test.min() >= 0 and y_test.max() <= 9, "Test labels out of range"

           return (x_train, y_train), (x_test, y_test)

       except Exception as e:
           logger.error(f"Failed to load MNIST dataset: {str(e)}")
           raise


   def get_dataset_info(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray) -> dict:
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

       info = {
           'num_classes': 10,
           'image_shape': x_train.shape[1:],
           'train_samples': x_train.shape[0],
           'test_samples': x_test.shape[0],
           'total_samples': x_train.shape[0] + x_test.shape[0],
           'pixel_dtype': x_train.dtype,
           'pixel_range': (x_train.min(), x_train.max()),
           'train_class_distribution': train_class_counts.tolist(),
           'test_class_distribution': test_class_counts.tolist(),
           'memory_usage_mb': {
               'train_images': x_train.nbytes / 1024 / 1024,
               'train_labels': y_train.nbytes / 1024 / 1024,
               'test_images': x_test.nbytes / 1024 / 1024,
               'test_labels': y_test.nbytes / 1024 / 1024,
               'total': (x_train.nbytes + y_train.nbytes +
                        x_test.nbytes + y_test.nbytes) / 1024 / 1024
           }
       }

       return info


   def save_sample_images(x_data: np.ndarray, y_data: np.ndarray,
                         num_samples: int = 10,
                         output_dir: str = 'data/samples/') -> None:
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
       os.makedirs(output_dir, exist_ok=True)

       # Save random samples
       indices = np.random.choice(len(x_data), num_samples, replace=False)

       for i, idx in enumerate(indices):
           img = Image.fromarray(x_data[idx])
           label = y_data[idx]
           filename = f"sample_{i:02d}_label_{label}.png"
           filepath = os.path.join(output_dir, filename)
           img.save(filepath)
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
       for digit, count in enumerate(info['train_class_distribution']):
           bar = "█" * int(count / 1000)
           print(f"  {digit}: {bar} {count:,}")

       # Save some sample images
       print("\n💾 Saving sample images...")
       save_sample_images(x_train, y_train, num_samples=10)
       print(f"Sample images saved to data/samples/")

       print("\n✅ Data loading test completed successfully!")
   ```

4. **Test the data loader**:
   ```bash
   # Make sure your virtual environment is activated
   python src/mnist_classifier/load_data.py
   ```

   Expected output:
   ```
   Testing MNIST data loader...
   --------------------------------------------------
   INFO:__main__:Loading MNIST dataset...
   INFO:__main__:Successfully loaded MNIST dataset
   INFO:__main__:Training set: 60000 samples
   INFO:__main__:Test set: 10000 samples
   ...
   ```

### Understanding the Code

Let's break down the key concepts:

1. **Type Hints**:
   ```python
   def load_mnist(data_dir: str = 'data/') -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
   ```
   - Type hints make code more readable and help IDEs provide better suggestions
   - `Tuple[Tuple[...], Tuple[...]]` indicates nested tuples

2. **Logging**:
   ```python
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```
   - Better than print statements for production code
   - Can be configured to write to files, filter by level, etc.

3. **Data Validation**:
   ```python
   assert x_train.shape == (60000, 28, 28), f"Unexpected shape: {x_train.shape}"
   ```
   - Catches data corruption or API changes early
   - Fails fast with clear error messages

4. **Memory Efficiency**:
   - MNIST is small (~11MB), but good habits matter
   - The loader returns references, not copies
   - For larger datasets, consider generators or tf.data pipelines

### Common Issues and Solutions

**Issue**: "No module named 'tensorflow'"
- **Solution**: Activate your virtual environment and ensure TensorFlow is installed

**Issue**: Download fails or hangs
- **Solution**: Check internet connection; TensorFlow will retry automatically

**Issue**: Out of memory error
- **Solution**: Close other applications; MNIST is small but still needs ~50MB RAM

---

## Task 2.2: Create Data Exploration Notebook

### Why Data Exploration Matters
Before building any model, you need to understand your data:
- What do the images look like?
- Are classes balanced?
- Are there any anomalies?
- What preprocessing might be needed?

### Introduction to Jupyter Notebooks
Jupyter notebooks are interactive documents that combine:
- Code cells (executable Python)
- Markdown cells (formatted text)
- Output displays (plots, tables, images)

They're perfect for exploration because you can experiment and see results immediately.

### Step-by-Step Instructions

1. **Start Jupyter Notebook**:
   ```bash
   # Make sure your virtual environment is activated
   jupyter notebook
   ```

   This opens a browser window. Navigate to the `notebooks/` directory.

2. **Create a new notebook**:
   - Click "New" → "Python 3"
   - Rename it to "01_data_exploration.ipynb" (click on "Untitled")

3. **Add the exploration code**:

   Copy these cells into your notebook:

   **Cell 1 - Imports and Setup**:
   ```python
   """
   MNIST Data Exploration Notebook

   This notebook explores the MNIST dataset to understand:
   - Image characteristics
   - Label distribution
   - Data quality
   - Preprocessing needs
   """

   # Standard imports
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from PIL import Image
   import sys
   import os

   # Add project root to path so we can import our modules
   sys.path.insert(0, os.path.abspath('..'))

   # Import our data loader
   from src.mnist_classifier.load_data import load_mnist, get_dataset_info

   # Configure visualization settings
   plt.style.use('default')  # Use default style for consistency
   sns.set_palette("husl")
   plt.rcParams['figure.figsize'] = (10, 8)
   plt.rcParams['font.size'] = 12

   print("Imports successful! 🎉")
   ```

   **Cell 2 - Load Data**:
   ```python
   # Load the MNIST dataset
   print("Loading MNIST dataset...")
   (x_train, y_train), (x_test, y_test) = load_mnist()

   # Get dataset information
   info = get_dataset_info(x_train, y_train, x_test, y_test)

   print(f"\n✅ Dataset loaded successfully!")
   print(f"Training samples: {info['train_samples']:,}")
   print(f"Test samples: {info['test_samples']:,}")
   print(f"Image shape: {info['image_shape']}")
   print(f"Memory usage: {info['memory_usage_mb']['total']:.2f} MB")
   ```

   **Cell 3 - Visualize Sample Images**:
   ```python
   def plot_digit_samples(images, labels, n_samples=25, title="Sample Digits"):
       """
       Plot a grid of sample digit images with their labels.
       """
       # Calculate grid dimensions
       n_cols = int(np.sqrt(n_samples))
       n_rows = int(np.ceil(n_samples / n_cols))

       # Create figure
       fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
       fig.suptitle(title, fontsize=16)

       # Flatten axes array for easy iteration
       axes = axes.ravel()

       # Plot each sample
       for i in range(n_samples):
           # Select random image
           idx = np.random.randint(0, len(images))

           # Plot image
           axes[i].imshow(images[idx], cmap='gray')
           axes[i].set_title(f'Label: {labels[idx]}')
           axes[i].axis('off')

       # Hide any unused subplots
       for i in range(n_samples, len(axes)):
           axes[i].axis('off')

       plt.tight_layout()
       plt.show()

   # Display random samples
   plot_digit_samples(x_train, y_train, n_samples=25,
                     title="Random Training Samples")
   ```

   **Cell 4 - Analyze Pixel Value Distribution**:
   ```python
   def analyze_pixel_distribution(images, sample_size=1000):
       """
       Analyze the distribution of pixel values in the dataset.
       """
       # Sample images for efficiency
       sample_indices = np.random.choice(len(images), sample_size, replace=False)
       sample_images = images[sample_indices]

       # Flatten all pixels
       all_pixels = sample_images.flatten()

       # Create figure with subplots
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

       # Histogram of pixel values
       ax1.hist(all_pixels, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
       ax1.set_xlabel('Pixel Value')
       ax1.set_ylabel('Density')
       ax1.set_title('Distribution of Pixel Values')
       ax1.axvline(all_pixels.mean(), color='red', linestyle='--',
                   label=f'Mean: {all_pixels.mean():.1f}')
       ax1.axvline(all_pixels.std(), color='green', linestyle='--',
                   label=f'Std: {all_pixels.std():.1f}')
       ax1.legend()

       # Box plot by digit class
       pixel_by_class = []
       for digit in range(10):
           digit_images = images[sample_indices[y_train[sample_indices] == digit]]
           if len(digit_images) > 0:
               pixel_by_class.append(digit_images.flatten())

       ax2.boxplot(pixel_by_class, labels=range(10))
       ax2.set_xlabel('Digit Class')
       ax2.set_ylabel('Pixel Value')
       ax2.set_title('Pixel Value Distribution by Digit Class')

       plt.tight_layout()
       plt.show()

       # Print statistics
       print("📊 Pixel Value Statistics:")
       print(f"Min value: {all_pixels.min()}")
       print(f"Max value: {all_pixels.max()}")
       print(f"Mean value: {all_pixels.mean():.2f}")
       print(f"Std deviation: {all_pixels.std():.2f}")
       print(f"Percentage of zero pixels (black): {(all_pixels == 0).mean() * 100:.1f}%")
       print(f"Percentage of max pixels (white): {(all_pixels == 255).mean() * 100:.1f}%")

   analyze_pixel_distribution(x_train)
   ```

   **Cell 5 - Class Distribution Analysis**:
   ```python
   def plot_class_distribution(y_train, y_test):
       """
       Visualize the distribution of classes in train and test sets.
       """
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

       # Training set distribution
       train_counts = np.bincount(y_train)
       ax1.bar(range(10), train_counts, color='skyblue', edgecolor='black')
       ax1.set_xlabel('Digit')
       ax1.set_ylabel('Count')
       ax1.set_title('Training Set Class Distribution')
       ax1.set_xticks(range(10))

       # Add count labels on bars
       for i, count in enumerate(train_counts):
           ax1.text(i, count + 100, str(count), ha='center')

       # Test set distribution
       test_counts = np.bincount(y_test)
       ax2.bar(range(10), test_counts, color='lightcoral', edgecolor='black')
       ax2.set_xlabel('Digit')
       ax2.set_ylabel('Count')
       ax2.set_title('Test Set Class Distribution')
       ax2.set_xticks(range(10))

       # Add count labels on bars
       for i, count in enumerate(test_counts):
           ax2.text(i, count + 20, str(count), ha='center')

       plt.tight_layout()
       plt.show()

       # Calculate and display balance metrics
       train_balance = train_counts.std() / train_counts.mean()
       test_balance = test_counts.std() / test_counts.mean()

       print("⚖️ Class Balance Analysis:")
       print(f"Training set - Coefficient of variation: {train_balance:.3f}")
       print(f"Test set - Coefficient of variation: {test_balance:.3f}")
       print(f"(Lower values indicate better balance, 0 = perfect balance)")

   plot_class_distribution(y_train, y_test)
   ```

   **Cell 6 - Examine Individual Digits**:
   ```python
   def examine_digit_variations(images, labels, digit=7, n_samples=10):
       """
       Show variations in how a specific digit is written.
       """
       # Find all instances of the digit
       digit_indices = np.where(labels == digit)[0]

       # Sample some instances
       sample_indices = np.random.choice(digit_indices,
                                       min(n_samples, len(digit_indices)),
                                       replace=False)

       # Plot the variations
       fig, axes = plt.subplots(2, 5, figsize=(12, 6))
       fig.suptitle(f'Different Ways People Write "{digit}"', fontsize=16)

       axes = axes.ravel()
       for i, idx in enumerate(sample_indices):
           axes[i].imshow(images[idx], cmap='gray')
           axes[i].axis('off')

           # Calculate some basic statistics for each image
           img = images[idx]
           center_of_mass_y = np.sum(np.arange(28) * img.sum(axis=1)) / img.sum()
           center_of_mass_x = np.sum(np.arange(28) * img.sum(axis=0)) / img.sum()

           axes[i].plot(center_of_mass_x, center_of_mass_y, 'r+', markersize=10)

       plt.tight_layout()
       plt.show()

       print(f"Note: Red crosses show the center of mass of each digit.")
       print(f"This helps visualize how digits are positioned within the 28x28 frame.")

   # Examine variations for each digit
   for digit in [1, 7, 9]:  # These digits often have interesting variations
       examine_digit_variations(x_train, y_train, digit=digit)
   ```

   **Cell 7 - Image Quality Analysis**:
   ```python
   def analyze_image_quality(images, labels, n_samples=1000):
       """
       Analyze image quality metrics like contrast and centering.
       """
       # Sample for efficiency
       sample_indices = np.random.choice(len(images), n_samples, replace=False)
       sample_images = images[sample_indices]
       sample_labels = labels[sample_indices]

       # Calculate metrics for each image
       contrasts = []
       center_offsets = []
       ink_percentages = []

       for img in sample_images:
           # Contrast (std of pixel values)
           contrasts.append(img.std())

           # Center offset
           y_coords, x_coords = np.mgrid[0:28, 0:28]
           center_y = np.sum(y_coords * img) / np.sum(img) if np.sum(img) > 0 else 14
           center_x = np.sum(x_coords * img) / np.sum(img) if np.sum(img) > 0 else 14
           offset = np.sqrt((center_x - 14)**2 + (center_y - 14)**2)
           center_offsets.append(offset)

           # Ink percentage (non-zero pixels)
           ink_percentages.append((img > 0).mean() * 100)

       # Create visualization
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))

       # Contrast distribution
       axes[0, 0].hist(contrasts, bins=30, alpha=0.7, color='green', edgecolor='black')
       axes[0, 0].set_xlabel('Contrast (Std Dev)')
       axes[0, 0].set_ylabel('Count')
       axes[0, 0].set_title('Image Contrast Distribution')

       # Center offset distribution
       axes[0, 1].hist(center_offsets, bins=30, alpha=0.7, color='orange', edgecolor='black')
       axes[0, 1].set_xlabel('Distance from Center (pixels)')
       axes[0, 1].set_ylabel('Count')
       axes[0, 1].set_title('Digit Centering Distribution')

       # Ink percentage distribution
       axes[1, 0].hist(ink_percentages, bins=30, alpha=0.7, color='purple', edgecolor='black')
       axes[1, 0].set_xlabel('Ink Coverage (%)')
       axes[1, 0].set_ylabel('Count')
       axes[1, 0].set_title('Ink Coverage Distribution')

       # Examples of outliers
       # Find low contrast image
       low_contrast_idx = sample_indices[np.argmin(contrasts)]
       axes[1, 1].imshow(images[low_contrast_idx], cmap='gray')
       axes[1, 1].set_title(f'Low Contrast Example (Label: {labels[low_contrast_idx]})')
       axes[1, 1].axis('off')

       plt.tight_layout()
       plt.show()

       print("📊 Image Quality Statistics:")
       print(f"Average contrast: {np.mean(contrasts):.2f}")
       print(f"Average center offset: {np.mean(center_offsets):.2f} pixels")
       print(f"Average ink coverage: {np.mean(ink_percentages):.1f}%")

   analyze_image_quality(x_train, y_train)
   ```

   **Cell 8 - Data Augmentation Preview**:
   ```python
   def preview_augmentations(image):
       """
       Preview potential data augmentations for training.
       """
       from scipy.ndimage import rotate, shift

       fig, axes = plt.subplots(2, 4, figsize=(12, 6))
       fig.suptitle('Potential Data Augmentations', fontsize=16)

       # Original
       axes[0, 0].imshow(image, cmap='gray')
       axes[0, 0].set_title('Original')
       axes[0, 0].axis('off')

       # Slight rotation
       rotated = rotate(image, angle=15, reshape=False)
       axes[0, 1].imshow(rotated, cmap='gray')
       axes[0, 1].set_title('Rotated +15°')
       axes[0, 1].axis('off')

       # Slight shift
       shifted = shift(image, shift=(2, -2))
       axes[0, 2].imshow(shifted, cmap='gray')
       axes[0, 2].set_title('Shifted')
       axes[0, 2].axis('off')

       # Add noise
       noisy = image + np.random.normal(0, 10, image.shape)
       noisy = np.clip(noisy, 0, 255)
       axes[0, 3].imshow(noisy, cmap='gray')
       axes[0, 3].set_title('With Noise')
       axes[0, 3].axis('off')

       # Brightness adjustment
       bright = np.clip(image * 1.2, 0, 255)
       axes[1, 0].imshow(bright, cmap='gray')
       axes[1, 0].set_title('Brighter')
       axes[1, 0].axis('off')

       # Contrast adjustment
       mean = image.mean()
       contrast = np.clip((image - mean) * 1.5 + mean, 0, 255)
       axes[1, 1].imshow(contrast, cmap='gray')
       axes[1, 1].set_title('Higher Contrast')
       axes[1, 1].axis('off')

       # Slight blur (simulate different pen styles)
       from scipy.ndimage import gaussian_filter
       blurred = gaussian_filter(image, sigma=0.5)
       axes[1, 2].imshow(blurred, cmap='gray')
       axes[1, 2].set_title('Slight Blur')
       axes[1, 2].axis('off')

       # Elastic deformation preview
       axes[1, 3].text(0.5, 0.5, 'Elastic\nDeformation\n(Advanced)',
                       ha='center', va='center', transform=axes[1, 3].transAxes)
       axes[1, 3].axis('off')

       plt.tight_layout()
       plt.show()

   # Show augmentation possibilities
   sample_idx = np.random.randint(0, len(x_train))
   print(f"Showing augmentation preview for a '{y_train[sample_idx]}':")
   preview_augmentations(x_train[sample_idx])
   ```

   **Cell 9 - Summary and Insights**:
   ```python
   def create_exploration_summary():
       """
       Summarize key findings from our exploration.
       """
       print("📋 MNIST Dataset Exploration Summary")
       print("=" * 50)

       print("\n1. Dataset Characteristics:")
       print(f"   - Total samples: {len(x_train) + len(x_test):,}")
       print(f"   - Image dimensions: 28x28 pixels (784 features)")
       print(f"   - Grayscale values: 0 (black) to 255 (white)")
       print(f"   - Highly imbalanced: No (all classes ~10% ± 1%)")

       print("\n2. Key Observations:")
       print("   - Most pixels are black (background)")
       print("   - Digits are generally well-centered")
       print("   - Significant variation in writing styles")
       print("   - Good contrast between digits and background")

       print("\n3. Preprocessing Recommendations:")
       print("   ✓ Normalize pixel values to [0, 1] range")
       print("   ✓ Reshape for CNN: (28, 28) → (28, 28, 1)")
       print("   ✓ One-hot encode labels for classification")
       print("   ✓ Consider data augmentation for better generalization")

       print("\n4. Potential Challenges:")
       print("   - Similar looking digits (e.g., 1 vs 7, 3 vs 8)")
       print("   - Varying stroke thickness and styles")
       print("   - Some digits are off-center or rotated")

       print("\n5. Next Steps:")
       print("   → Implement preprocessing pipeline")
       print("   → Create train/validation split")
       print("   → Design CNN architecture")
       print("   → Train and evaluate model")

   create_exploration_summary()
   ```

4. **Save the notebook**:
   - File → Save and Checkpoint (or Ctrl+S)
   - The notebook auto-saves periodically

### Understanding Jupyter Notebooks

**Key Concepts**:
1. **Cell Types**:
   - Code cells: Execute Python code
   - Markdown cells: Documentation (double-click to edit)

2. **Execution Order**:
   - Cells can be run in any order (but be careful!)
   - Numbers in brackets show execution order: `[1]`, `[2]`, etc.

3. **Keyboard Shortcuts**:
   - `Shift + Enter`: Run cell and move to next
   - `Ctrl + Enter`: Run cell and stay
   - `Esc` then `A`: Insert cell above
   - `Esc` then `B`: Insert cell below

4. **Best Practices**:
   - Run cells in order for reproducibility
   - Clear output before sharing: Kernel → Restart & Clear Output
   - Use meaningful variable names
   - Add markdown cells to explain your thinking

### Common Notebook Issues

**Issue**: "No module named 'src'"
- **Solution**: Make sure you're running the notebook from the project root

**Issue**: Plots not showing
- **Solution**: Add `%matplotlib inline` at the beginning

**Issue**: Kernel died
- **Solution**: Restart kernel (Kernel → Restart) and re-run cells

---

## Task 2.3: Create Preprocessing Module

### Why Preprocessing is Critical
Neural networks are sensitive to input data format and scale. Proper preprocessing:
- Improves convergence speed
- Prevents numerical instability
- Ensures all features contribute equally
- Matches the assumptions of activation functions

### Key Preprocessing Steps for MNIST

1. **Normalization**: Scale pixel values from [0, 255] to [0, 1]
   - Why: Neural networks work better with small numbers
   - How: Divide by 255

2. **Reshaping**: Add channel dimension (28, 28) → (28, 28, 1)
   - Why: CNNs expect channel information (like RGB)
   - How: Use numpy reshape or expand_dims

3. **One-Hot Encoding**: Convert labels to binary vectors
   - Why: Treats classification as 10 binary problems
   - How: Label 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

### Step-by-Step Instructions

1. **Create the preprocessing module**:
   ```bash
   touch src/mnist_classifier/preprocess.py
   ```

2. **Implement preprocessing functions**:

   Add this code to `src/mnist_classifier/preprocess.py`:

   ```python
   """
   Data Preprocessing Module for MNIST

   This module provides functions to prepare MNIST data for neural network training.
   All transformations are designed to be reversible for visualization purposes.
   """

   import numpy as np
   import tensorflow as tf
   from typing import Tuple, Optional, Union
   import logging

   logger = logging.getLogger(__name__)


   def normalize_pixels(images: np.ndarray,
                       method: str = 'standard') -> np.ndarray:
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

       if method == 'standard':
           # Scale to [0, 1] - most common for images
           normalized = images / 255.0
           logger.info(f"Normalized {len(images)} images to [0, 1] range")

       elif method == 'centered':
           # Scale to [-1, 1] - sometimes better for certain architectures
           normalized = (images - 127.5) / 127.5
           logger.info(f"Normalized {len(images)} images to [-1, 1] range")

       else:
           raise ValueError(f"Unknown normalization method: {method}")

       # Verify output range
       if method == 'standard':
           assert normalized.min() >= 0 and normalized.max() <= 1, \
               f"Normalization failed: range [{normalized.min()}, {normalized.max()}]"

       return normalized


   def denormalize_pixels(images: np.ndarray,
                         method: str = 'standard') -> np.ndarray:
       """
       Reverse normalization for visualization.

       Args:
           images: Normalized images
           method: Same method used for normalization

       Returns:
           Images with pixel values in [0, 255] as uint8
       """
       if method == 'standard':
           denormalized = images * 255.0
       elif method == 'centered':
           denormalized = (images + 1) * 127.5
       else:
           raise ValueError(f"Unknown normalization method: {method}")

       # Clip values and convert to uint8
       denormalized = np.clip(denormalized, 0, 255)
       return denormalized.astype(np.uint8)


   def reshape_images(images: np.ndarray,
                     add_channel: bool = True) -> np.ndarray:
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


   def one_hot_encode_labels(labels: np.ndarray,
                           num_classes: int = 10) -> np.ndarray:
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
           raise ValueError(f"Labels must be in range [0, {num_classes-1}], "
                          f"got [{labels.min()}, {labels.max()}]")

       # Create one-hot encoding
       one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
       one_hot[np.arange(labels.shape[0]), labels] = 1

       logger.info(f"One-hot encoded {len(labels)} labels into {one_hot.shape}")

       # Verify encoding
       assert np.all(one_hot.sum(axis=1) == 1), "One-hot encoding failed"

       return one_hot


   def decode_one_hot_labels(one_hot: np.ndarray) -> np.ndarray:
       """
       Convert one-hot encoded labels back to integers.

       Args:
           one_hot: One-hot encoded labels of shape (n_samples, num_classes)

       Returns:
           Integer labels of shape (n_samples,)
       """
       return np.argmax(one_hot, axis=1)


   def create_train_validation_split(x_data: np.ndarray,
                                   y_data: np.ndarray,
                                   validation_split: float = 0.1,
                                   random_seed: Optional[int] = 42) -> Tuple:
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
           raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")

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

       logger.info(f"Created train/validation split: "
                  f"{len(x_train)} train, {len(x_val)} validation samples")

       # Verify class distribution in both sets
       train_classes = np.unique(y_train)
       val_classes = np.unique(y_val)

       if len(train_classes) != len(val_classes):
           logger.warning(f"Class imbalance detected: "
                         f"train has {len(train_classes)} classes, "
                         f"val has {len(val_classes)} classes")

       return x_train, y_train, x_val, y_val


   def prepare_data_for_training(x_train: np.ndarray, y_train: np.ndarray,
                               x_test: np.ndarray, y_test: np.ndarray,
                               validation_split: float = 0.1,
                               normalize_method: str = 'standard') -> dict:
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
           'x_train': x_train_final,
           'y_train': y_train_encoded,
           'x_val': x_val,
           'y_val': y_val_encoded,
           'x_test': x_test_reshaped,
           'y_test': y_test_encoded,
           'preprocessing_params': {
               'normalize_method': normalize_method,
               'validation_split': validation_split,
               'input_shape': x_train_final.shape[1:],
               'num_classes': y_train_encoded.shape[1]
           }
       }

       logger.info("Preprocessing complete!")
       logger.info(f"Final shapes - Train: {x_train_final.shape}, "
                  f"Val: {x_val.shape}, Test: {x_test_reshaped.shape}")

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
       print(f"   ✓ Normalization successful!")

       print("\n2. Testing reshaping:")
       reshaped = reshape_images(normalized)
       print(f"   Original shape: {normalized.shape}")
       print(f"   Reshaped shape: {reshaped.shape}")
       print(f"   ✓ Reshaping successful!")

       print("\n3. Testing one-hot encoding:")
       one_hot = one_hot_encode_labels(sample_labels, num_classes=10)
       print(f"   Original labels: {sample_labels}")
       print(f"   One-hot shape: {one_hot.shape}")
       print(f"   Sample encoding for label 3:")
       print(f"   {one_hot[3]}")
       print(f"   ✓ One-hot encoding successful!")

       print("\n4. Testing train/validation split:")
       x_train, y_train, x_val, y_val = create_train_validation_split(
           reshaped, sample_labels, validation_split=0.4
       )
       print(f"   Training samples: {len(x_train)}")
       print(f"   Validation samples: {len(x_val)}")
       print(f"   ✓ Split successful!")

       print("\n✅ All preprocessing functions working correctly!")
   ```

3. **Test the preprocessing module**:
   ```bash
   python src/mnist_classifier/preprocess.py
   ```

### Understanding the Preprocessing

**1. Normalization Deep Dive**:
```python
normalized = images / 255.0
```
- Pixels are 8-bit integers (0-255)
- Neural networks prefer small floating-point numbers
- Division by 255 maps to [0, 1]
- Alternative: StandardScaler for mean=0, std=1

**2. Why One-Hot Encoding?**
```
Label: 3
One-hot: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
         [0  1  2  3  4  5  6  7  8  9] <- indices
```
- Treats each class independently
- No implicit ordering (3 is not "greater than" 2)
- Matches softmax output perfectly
- Enables use of categorical crossentropy loss

**3. Train/Validation Split**:
- Never train on validation data!
- Validation helps detect overfitting
- Common splits: 80/20, 90/10
- Stratified split ensures class balance

### Common Preprocessing Mistakes

❌ **Forgetting to normalize test data**
- Always apply same preprocessing to train and test

❌ **Normalizing before splitting**
- Can leak information between train/validation

❌ **Using training statistics on test data**
- Test data should be treated as "unseen"

❌ **Not preserving preprocessing parameters**
- Need same parameters for inference

---

## Task 2.4: Update Data Loading with Preprocessing

### Integration Best Practices
When integrating components:
- Keep functions modular and testable
- Document the data flow clearly
- Validate outputs at each step
- Make pipelines reproducible

### Step-by-Step Instructions

1. **Create an integrated data pipeline script**:
   ```bash
   touch src/mnist_classifier/data_pipeline.py
   ```

2. **Implement the integrated pipeline**:

   Add this code to `src/mnist_classifier/data_pipeline.py`:

   ```python
   """
   Integrated Data Pipeline for MNIST

   This module combines data loading and preprocessing into a single,
   easy-to-use pipeline for training neural networks.
   """

   import os
   import json
   import pickle
   from typing import Dict, Tuple, Optional
   import numpy as np
   import logging

   from .load_data import load_mnist, get_dataset_info
   from .preprocess import (
       prepare_data_for_training,
       normalize_pixels,
       reshape_images,
       one_hot_encode_labels
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

       def __init__(self, data_dir: str = 'data/',
                    cache_dir: str = 'data/processed/'):
           """
           Initialize the data pipeline.

           Args:
               data_dir: Directory for raw data
               cache_dir: Directory for preprocessed data cache
           """
           self.data_dir = data_dir
           self.cache_dir = cache_dir
           self.data_loaded = False
           self.preprocessing_params = None

           # Create directories if needed
           os.makedirs(self.cache_dir, exist_ok=True)

       def prepare_data(self,
                       validation_split: float = 0.1,
                       normalize_method: str = 'standard',
                       use_cache: bool = True,
                       force_reload: bool = False) -> Dict:
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
           cache_file = os.path.join(self.cache_dir, 'preprocessed_data.pkl')
           params_file = os.path.join(self.cache_dir, 'preprocessing_params.json')

           # Try to load from cache
           if use_cache and not force_reload and os.path.exists(cache_file):
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
               x_train, y_train, x_test, y_test,
               validation_split=validation_split,
               normalize_method=normalize_method
           )

           # Add dataset info
           preprocessed_data['dataset_info'] = info

           # Save to cache if requested
           if use_cache:
               self._save_to_cache(preprocessed_data, cache_file, params_file)

           self.data_loaded = True
           self.preprocessing_params = preprocessed_data['preprocessing_params']

           return preprocessed_data

       def _save_to_cache(self, data: Dict, cache_file: str, params_file: str):
           """Save preprocessed data to cache."""
           logger.info(f"Saving preprocessed data to {cache_file}...")

           # Save data as pickle
           with open(cache_file, 'wb') as f:
               pickle.dump(data, f)

           # Save parameters as JSON for readability
           with open(params_file, 'w') as f:
               json.dump(data['preprocessing_params'], f, indent=2)

           logger.info("Cache saved successfully!")

       def _load_from_cache(self, cache_file: str, params_file: str) -> Dict:
           """Load preprocessed data from cache."""
           with open(cache_file, 'rb') as f:
               data = pickle.load(f)

           with open(params_file, 'r') as f:
               params = json.load(f)

           # Verify parameters match
           if data['preprocessing_params'] != params:
               logger.warning("Cache parameters mismatch! Reprocessing...")
               raise ValueError("Cache invalid")

           logger.info("Cache loaded successfully!")
           self.data_loaded = True
           self.preprocessing_params = params

           return data

       def get_batch_generator(self, x_data: np.ndarray, y_data: np.ndarray,
                             batch_size: int = 32, shuffle: bool = True):
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

       def prepare_single_image(self, image: np.ndarray) -> np.ndarray:
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
           normalized = normalize_pixels(image.reshape(1, 28, 28),
                                       method=self.preprocessing_params['normalize_method'])
           reshaped = reshape_images(normalized, add_channel=True)

           return reshaped

       def get_sample_batch(self, dataset: str = 'train',
                          n_samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
           """
           Get a sample batch for testing or visualization.

           Args:
               dataset: Which dataset to sample from ('train', 'val', 'test')
               n_samples: Number of samples

           Returns:
               Tuple of (images, labels)
           """
           if not self.data_loaded:
               raise ValueError("Data not loaded. Call prepare_data() first.")

           # This would need access to the loaded data
           # Implementation depends on how you store the data
           pass


   def create_data_summary_report(data: Dict, output_file: str = 'data/data_summary.txt'):
       """
       Create a comprehensive summary report of the processed data.

       Args:
           data: Preprocessed data dictionary
           output_file: Where to save the report
       """
       os.makedirs(os.path.dirname(output_file), exist_ok=True)

       with open(output_file, 'w') as f:
           f.write("MNIST Data Pipeline Summary Report\n")
           f.write("=" * 50 + "\n\n")

           # Dataset info
           info = data['dataset_info']
           f.write("Dataset Information:\n")
           f.write(f"  Total samples: {info['total_samples']:,}\n")
           f.write(f"  Image shape: {info['image_shape']}\n")
           f.write(f"  Number of classes: {info['num_classes']}\n")
           f.write(f"  Memory usage: {info['memory_usage_mb']['total']:.2f} MB\n\n")

           # Preprocessing parameters
           params = data['preprocessing_params']
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
           train_labels = np.argmax(data['y_train'], axis=1)
           for digit in range(10):
               count = np.sum(train_labels == digit)
               percentage = count / len(train_labels) * 100
               f.write(f"  Digit {digit}: {count:,} ({percentage:.1f}%)\n")

       logger.info(f"Data summary report saved to {output_file}")


   def verify_data_pipeline():
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
       print(f"   ✓ Data prepared successfully")
       print(f"   ✓ Training shape: {data['x_train'].shape}")
       print(f"   ✓ Validation shape: {data['x_val'].shape}")
       print(f"   ✓ Test shape: {data['x_test'].shape}")

       # Test 2: Cache functionality
       print("\n2. Testing cache functionality...")
       import time
       start_time = time.time()
       data_cached = pipeline.prepare_data(validation_split=0.15)
       cache_time = time.time() - start_time
       print(f"   ✓ Cache loading took {cache_time:.2f} seconds")

       # Test 3: Single image preprocessing
       print("\n3. Testing single image preprocessing...")
       test_image = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
       processed = pipeline.prepare_single_image(test_image)
       print(f"   ✓ Single image processed: {test_image.shape} → {processed.shape}")

       # Test 4: Batch generator
       print("\n4. Testing batch generator...")
       gen = pipeline.get_batch_generator(data['x_train'], data['y_train'],
                                        batch_size=64)
       x_batch, y_batch = next(gen)
       print(f"   ✓ Batch shapes: X={x_batch.shape}, Y={y_batch.shape}")

       # Test 5: Data summary report
       print("\n5. Creating data summary report...")
       create_data_summary_report(data)
       print(f"   ✓ Report saved to data/data_summary.txt")

       print("\n✅ All pipeline tests passed!")
       return data


   if __name__ == "__main__":
       # Run comprehensive pipeline test
       data = verify_data_pipeline()

       # Print final summary
       print("\n" + "=" * 50)
       print("📊 Pipeline ready for model training!")
       print(f"Next steps:")
       print(f"  1. Design CNN architecture")
       print(f"  2. Train model using prepared data")
       print(f"  3. Evaluate on test set")
   ```

3. **Create a complete example script**:
   ```bash
   touch src/mnist_classifier/example_usage.py
   ```

   Add this code to show how everything works together:

   ```python
   """
   Example usage of the complete MNIST data pipeline.

   This script demonstrates how to use the data pipeline
   in a real training scenario.
   """

   import numpy as np
   import matplotlib.pyplot as plt
   from data_pipeline import MNISTDataPipeline


   def visualize_preprocessed_batch(x_batch, y_batch, n_samples=16):
       """Visualize a batch of preprocessed images."""
       n_show = min(n_samples, len(x_batch))
       n_cols = int(np.sqrt(n_show))
       n_rows = int(np.ceil(n_show / n_cols))

       fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
       axes = axes.ravel()

       for i in range(n_show):
           # Remove channel dimension for visualization
           img = x_batch[i].squeeze()
           label = np.argmax(y_batch[i])

           axes[i].imshow(img, cmap='gray')
           axes[i].set_title(f'Label: {label}')
           axes[i].axis('off')

       plt.tight_layout()
       plt.savefig('data/preprocessed_samples.png')
       print("Saved visualization to data/preprocessed_samples.png")


   def main():
       """Main example workflow."""
       print("MNIST Data Pipeline Example")
       print("=" * 50)

       # 1. Initialize pipeline
       pipeline = MNISTDataPipeline()

       # 2. Prepare data
       print("\nPreparing data...")
       data = pipeline.prepare_data(
           validation_split=0.1,
           normalize_method='standard',
           use_cache=True
       )

       # 3. Extract components
       x_train = data['x_train']
       y_train = data['y_train']
       x_val = data['x_val']
       y_val = data['y_val']
       x_test = data['x_test']
       y_test = data['y_test']

       # 4. Print shapes and info
       print(f"\nData shapes:")
       print(f"  Train: {x_train.shape}, {y_train.shape}")
       print(f"  Val: {x_val.shape}, {y_val.shape}")
       print(f"  Test: {x_test.shape}, {y_test.shape}")

       # 5. Verify preprocessing
       print(f"\nPreprocessing verification:")
       print(f"  Pixel range: [{x_train.min():.3f}, {x_train.max():.3f}]")
       print(f"  Labels are one-hot: {y_train.shape[1] == 10}")

       # 6. Create batch generator
       batch_gen = pipeline.get_batch_generator(x_train, y_train, batch_size=32)
       x_batch, y_batch = next(batch_gen)
       print(f"\nBatch generator test:")
       print(f"  Batch shapes: {x_batch.shape}, {y_batch.shape}")

       # 7. Visualize preprocessed data
       print("\nVisualizing preprocessed data...")
       visualize_preprocessed_batch(x_batch, y_batch)

       # 8. Simulate model training
       print("\nSimulating model training loop:")
       for epoch in range(3):
           print(f"\nEpoch {epoch + 1}/3")

           # Train for a few batches
           for batch_idx in range(5):
               x_batch, y_batch = next(batch_gen)
               # Here you would do: loss = model.train_on_batch(x_batch, y_batch)
               print(f"  Batch {batch_idx + 1}: shape {x_batch.shape}")

       print("\n✅ Pipeline demonstration complete!")
       print("\nYou can now use this pipeline with your neural network:")
       print("  - data['x_train'] and data['y_train'] for training")
       print("  - data['x_val'] and data['y_val'] for validation")
       print("  - data['x_test'] and data['y_test'] for final evaluation")


   if __name__ == "__main__":
       main()
   ```

### Understanding the Integration

**Key Design Decisions**:

1. **Class-based Pipeline**:
   - Encapsulates all data operations
   - Maintains state (preprocessing parameters)
   - Provides consistent interface

2. **Caching Strategy**:
   - Saves time on repeated runs
   - Stores parameters for reproducibility
   - Uses pickle for efficiency

3. **Batch Generator**:
   - Memory efficient for large datasets
   - Supports shuffling each epoch
   - Infinite loop for training

4. **Single Image Processing**:
   - Uses same parameters as training
   - Essential for inference/deployment

### Testing the Complete Pipeline

1. **Run the pipeline test**:
   ```bash
   python src/mnist_classifier/data_pipeline.py
   ```

2. **Run the example usage**:
   ```bash
   cd src/mnist_classifier
   python example_usage.py
   cd ../..
   ```

3. **Check the generated files**:
   ```bash
   ls data/
   ls data/processed/
   cat data/data_summary.txt
   ```

---

## Helper Scripts

### quick_setup_data.py - Quick Data Setup

Create this in your project root:

```python
#!/usr/bin/env python3
"""
Quick setup script for Milestone 2 data pipeline.
Run this to quickly test that everything is working.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.mnist_classifier.data_pipeline import MNISTDataPipeline
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("🚀 Quick Data Setup for MNIST Classifier")
    print("=" * 50)

    # Initialize and prepare data
    pipeline = MNISTDataPipeline()
    data = pipeline.prepare_data(validation_split=0.1)

    # Quick visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(10):
        # Find first occurrence of digit i
        labels = np.argmax(data['y_train'], axis=1)
        idx = np.where(labels == i)[0][0]

        axes[i].imshow(data['x_train'][idx].squeeze(), cmap='gray')
        axes[i].set_title(f'Digit: {i}')
        axes[i].axis('off')

    plt.suptitle('One Example of Each Digit (Preprocessed)')
    plt.tight_layout()
    plt.savefig('data/all_digits_sample.png')

    print("\n✅ Data pipeline setup complete!")
    print(f"📊 Saved digit samples to: data/all_digits_sample.png")
    print(f"📁 Cached data in: data/processed/")
    print(f"\nReady for Milestone 3: Neural Network Implementation!")


if __name__ == "__main__":
    main()
```

### verify_milestone2.py - Comprehensive Verification

Create this script to verify everything is working:

```python
#!/usr/bin/env python3
"""
Verification script for Milestone 2 completion.
Checks all requirements and provides detailed feedback.
"""

import os
import sys
import importlib.util
import subprocess
import json


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def check_files_exist():
    """Check if all required files exist."""
    print("\n📁 Checking required files:")

    required_files = [
        ('src/mnist_classifier/__init__.py', 'Package init file'),
        ('src/mnist_classifier/load_data.py', 'Data loading module'),
        ('src/mnist_classifier/preprocess.py', 'Preprocessing module'),
        ('src/mnist_classifier/data_pipeline.py', 'Integrated pipeline'),
        ('notebooks/01_data_exploration.ipynb', 'Exploration notebook'),
    ]

    all_exist = True
    for filepath, description in required_files:
        exists = os.path.exists(filepath)
        print(f"  {check_mark(exists)} {description}: {filepath}")
        if not exists:
            all_exist = False

    return all_exist


def check_imports():
    """Check if all modules can be imported."""
    print("\n📦 Checking module imports:")

    modules = [
        'src.mnist_classifier.load_data',
        'src.mnist_classifier.preprocess',
        'src.mnist_classifier.data_pipeline'
    ]

    all_importable = True
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  {check_mark(True)} {module_name}")
        except Exception as e:
            print(f"  {check_mark(False)} {module_name}: {str(e)}")
            all_importable = False

    return all_importable


def check_data_loading():
    """Test data loading functionality."""
    print("\n💾 Testing data loading:")

    try:
        from src.mnist_classifier.load_data import load_mnist
        (x_train, y_train), (x_test, y_test) = load_mnist()

        checks = [
            (x_train.shape == (60000, 28, 28), f"Training data shape: {x_train.shape}"),
            (y_train.shape == (60000,), f"Training labels shape: {y_train.shape}"),
            (x_test.shape == (10000, 28, 28), f"Test data shape: {x_test.shape}"),
            (y_test.shape == (10000,), f"Test labels shape: {y_test.shape}"),
        ]

        all_good = True
        for condition, description in checks:
            print(f"  {check_mark(condition)} {description}")
            if not condition:
                all_good = False

        return all_good

    except Exception as e:
        print(f"  {check_mark(False)} Failed to load data: {str(e)}")
        return False


def check_preprocessing():
    """Test preprocessing functionality."""
    print("\n🔧 Testing preprocessing:")

    try:
        import numpy as np
        from src.mnist_classifier.preprocess import (
            normalize_pixels, reshape_images, one_hot_encode_labels
        )

        # Test with sample data
        test_images = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
        test_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Test normalization
        normalized = normalize_pixels(test_images)
        norm_ok = normalized.min() >= 0 and normalized.max() <= 1
        print(f"  {check_mark(norm_ok)} Normalization: [{normalized.min():.3f}, {normalized.max():.3f}]")

        # Test reshaping
        reshaped = reshape_images(normalized)
        reshape_ok = reshaped.shape == (10, 28, 28, 1)
        print(f"  {check_mark(reshape_ok)} Reshaping: {reshaped.shape}")

        # Test one-hot encoding
        one_hot = one_hot_encode_labels(test_labels)
        onehot_ok = one_hot.shape == (10, 10) and np.all(one_hot.sum(axis=1) == 1)
        print(f"  {check_mark(onehot_ok)} One-hot encoding: {one_hot.shape}")

        return norm_ok and reshape_ok and onehot_ok

    except Exception as e:
        print(f"  {check_mark(False)} Preprocessing failed: {str(e)}")
        return False


def check_pipeline():
    """Test integrated pipeline."""
    print("\n🔄 Testing integrated pipeline:")

    try:
        from src.mnist_classifier.data_pipeline import MNISTDataPipeline

        pipeline = MNISTDataPipeline()
        data = pipeline.prepare_data(validation_split=0.1, force_reload=True)

        checks = [
            ('x_train' in data, "Training data present"),
            ('x_val' in data, "Validation data present"),
            ('x_test' in data, "Test data present"),
            (data['x_train'].shape[0] + data['x_val'].shape[0] == 60000,
             "Train/val split correct"),
            (data['x_train'].min() >= 0 and data['x_train'].max() <= 1,
             "Data normalized correctly"),
        ]

        all_good = True
        for condition, description in checks:
            print(f"  {check_mark(condition)} {description}")
            if not condition:
                all_good = False

        return all_good

    except Exception as e:
        print(f"  {check_mark(False)} Pipeline failed: {str(e)}")
        return False


def check_notebook():
    """Check if Jupyter notebook exists and has content."""
    print("\n📓 Checking Jupyter notebook:")

    notebook_path = 'notebooks/01_data_exploration.ipynb'

    if not os.path.exists(notebook_path):
        print(f"  {check_mark(False)} Notebook not found")
        return False

    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)

        n_cells = len(notebook.get('cells', []))
        has_code = any(cell['cell_type'] == 'code' for cell in notebook.get('cells', []))

        print(f"  {check_mark(n_cells > 0)} Notebook has {n_cells} cells")
        print(f"  {check_mark(has_code)} Notebook contains code cells")

        return n_cells > 5 and has_code

    except Exception as e:
        print(f"  {check_mark(False)} Could not read notebook: {str(e)}")
        return False


def check_generated_files():
    """Check for generated files from running the pipeline."""
    print("\n📄 Checking generated files:")

    expected_files = [
        ('data/processed/preprocessed_data.pkl', 'Cached preprocessed data'),
        ('data/processed/preprocessing_params.json', 'Preprocessing parameters'),
        ('data/data_summary.txt', 'Data summary report'),
    ]

    all_exist = True
    for filepath, description in expected_files:
        exists = os.path.exists(filepath)
        print(f"  {check_mark(exists)} {description}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("🔍 Milestone 2 Verification")
    print("=" * 50)

    # Make sure we're in the right directory
    if not os.path.exists('src/mnist_classifier'):
        print("❌ Error: Not in project root directory!")
        print("Please run this from the mnist_classifier project root.")
        sys.exit(1)

    # Add project root to path
    sys.path.insert(0, os.path.abspath('.'))

    # Run all checks
    checks = [
        ("Required Files", check_files_exist()),
        ("Module Imports", check_imports()),
        ("Data Loading", check_data_loading()),
        ("Preprocessing", check_preprocessing()),
        ("Pipeline Integration", check_pipeline()),
        ("Jupyter Notebook", check_notebook()),
        ("Generated Files", check_generated_files()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 2 is complete!")
        print("\nYour data pipeline is ready. You can now:")
        print("  - Load and preprocess MNIST data efficiently")
        print("  - Visualize and explore the dataset")
        print("  - Use the pipeline for model training")
        print("\nNext: Milestone 3 - Build and train the neural network!")
    else:
        print("\n⚠️  Some checks failed. Please review and fix the issues above.")
        print("\nTips:")
        print("  - Make sure your virtual environment is activated")
        print("  - Run the data_pipeline.py script to generate cache files")
        print("  - Check that all module imports are correct")


if __name__ == "__main__":
    main()
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Import Errors

**Problem**: "ModuleNotFoundError: No module named 'src'"
```bash
# Solution 1: Make sure you're in the project root
cd /path/to/mnist_classifier

# Solution 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Problem**: "No module named 'tensorflow'"
```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install tensorflow
```

#### Data Loading Issues

**Problem**: Download hangs or fails
- Check internet connection
- Try manual download:
  ```python
  import tensorflow as tf
  tf.keras.datasets.mnist.load_data()
  ```

**Problem**: Out of memory
- MNIST is small, but if you have issues:
  ```python
  # Process in smaller batches
  batch_size = 1000
  for i in range(0, len(x_train), batch_size):
      batch = x_train[i:i+batch_size]
      # Process batch
  ```

#### Jupyter Notebook Issues

**Problem**: Kernel keeps dying
- Reduce image count in visualizations
- Clear output: Kernel → Restart & Clear Output
- Check available memory

**Problem**: Plots not showing
```python
# Add at notebook start:
%matplotlib inline
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode on
```

#### File Path Issues

**Problem**: Can't find data files
```python
# Use absolute paths
import os
project_root = os.path.abspath('.')
data_path = os.path.join(project_root, 'data')
```

### Performance Optimization Tips

1. **Use caching aggressively**:
   ```python
   # Cache preprocessed data
   pipeline.prepare_data(use_cache=True)
   ```

2. **Process in batches**:
   ```python
   # Don't load entire dataset into memory at once
   for batch in data_generator:
       process(batch)
   ```

3. **Use efficient data types**:
   ```python
   # float32 instead of float64
   data = data.astype(np.float32)
   ```

---

## Learning Resources

### Essential Reading
1. **MNIST Database**: [Original MNIST page](http://yann.lecun.com/exdb/mnist/)
2. **NumPy Tutorial**: [Official NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
3. **Data Preprocessing**: [Google's ML Crash Course on Data Prep](https://developers.google.com/machine-learning/data-prep)

### Video Tutorials
1. **Understanding MNIST**: Search for "MNIST dataset explained" on YouTube
2. **Data Preprocessing**: "Feature scaling and normalization in ML"
3. **One-Hot Encoding**: "Categorical data encoding techniques"

### Hands-On Practice
1. **Kaggle MNIST**: Join the Digit Recognizer competition
2. **TensorFlow Tutorials**: Official MNIST tutorials
3. **Fast.ai Course**: Practical Deep Learning course

### Next Steps
1. Review the preprocessed data characteristics
2. Research CNN architectures for MNIST
3. Plan your model architecture
4. Prepare for Milestone 3: Neural Network Implementation

---

## Milestone 2 Checklist

Before moving to Milestone 3, ensure you've completed:

- [ ] Created data loading module (`load_data.py`)
- [ ] Implemented `load_mnist()` function with error handling
- [ ] Created data exploration notebook
- [ ] Visualized sample images and distributions
- [ ] Analyzed pixel values and class balance
- [ ] Created preprocessing module (`preprocess.py`)
- [ ] Implemented normalization function
- [ ] Implemented reshape function for CNN input
- [ ] Implemented one-hot encoding
- [ ] Created train/validation split function
- [ ] Integrated everything in `data_pipeline.py`
- [ ] Implemented caching for efficiency
- [ ] Created batch generator for training
- [ ] Tested single image preprocessing
- [ ] Generated data summary report
- [ ] All verification tests pass

🎉 **Congratulations on completing Milestone 2!**

You now have a robust, production-ready data pipeline that:
- Efficiently loads and caches data
- Applies consistent preprocessing
- Provides clean interfaces for training
- Includes comprehensive error handling
- Is well-documented and tested

The foundation is set for building your neural network. In Milestone 3, you'll design and train a CNN that achieves >98% accuracy on this well-prepared data!

### What You've Learned
- How to structure a data pipeline professionally
- The importance of data exploration before modeling
- Essential preprocessing techniques for neural networks
- How to create reusable, testable code modules
- Best practices for ML data handling

Ready for the exciting part - building the neural network? Head to `MILESTONE_3.md`!
