# Milestone 4: Command-Line Interface (CLI) - Detailed Implementation Guide

## Overview

Welcome to Milestone 4! Now that you have a trained neural network achieving >98% accuracy, it's time to make it useful. In this milestone, you'll build a command-line interface that allows anyone to classify handwritten digits from image files. This bridges the gap between a trained model and a practical application.

### What You'll Learn
- **Command-Line Interface Design**: Building user-friendly CLI tools with Python
- **Argument Parsing**: Using argparse to handle command-line arguments professionally
- **Image Processing**: Loading and preprocessing arbitrary image files for model input
- **Model Integration**: Loading saved models and running inference
- **Error Handling**: Gracefully handling user errors and edge cases
- **Documentation**: Writing clear usage instructions for CLI tools

### Why CLI Tools Matter
Command-line interfaces are essential because they:
- Enable automation and scripting
- Provide quick access to functionality
- Work on servers without GUI
- Form the basis for web APIs and services
- Are the standard for developer tools

### Prerequisites
Before starting, ensure you have:
- ✅ Completed Milestone 3 (trained model with >98% accuracy)
- ✅ Saved model file (`.h5` format) from training
- ✅ Understanding of Python functions and error handling
- ✅ Basic command-line familiarity
- ✅ About 45-60 minutes to complete all tasks

### Success Criteria
By the end of this milestone, you will have:
- ✅ A working CLI tool for digit classification
- ✅ Support for common image formats (PNG, JPG, BMP)
- ✅ Proper error handling and user feedback
- ✅ Clear documentation and help messages
- ✅ Batch processing capabilities
- ✅ Professional code structure

---

## Task 4.1: Create CLI Script

### Understanding Command-Line Interfaces

A good CLI tool should:
1. **Be intuitive**: Users should understand how to use it from the help message
2. **Provide feedback**: Clear error messages and progress indicators
3. **Handle errors gracefully**: Don't crash on bad input
4. **Follow conventions**: Use standard argument formats (--long-option, -s)

### Introduction to argparse

Python's `argparse` module is the standard way to handle command-line arguments. It provides:
- Automatic help generation
- Type checking and validation
- Error messages
- Support for subcommands

### Step-by-Step Instructions

1. **Create the CLI directory structure**:
   ```bash
   mkdir -p src/mnist_classifier/cli
   touch src/mnist_classifier/cli/__init__.py
   ```

2. **Create the main CLI module**:
   ```bash
   touch src/mnist_classifier/cli/predict.py
   ```

3. **Implement the CLI script**:

   Add this code to `src/mnist_classifier/cli/predict.py`:

   ```python
   """
   Command-Line Interface for MNIST Digit Classification

   This module provides a CLI tool for classifying handwritten digits
   from image files using a trained CNN model.
   """

   import argparse
   import sys
   import os
   from pathlib import Path
   from typing import Optional, Tuple, List, Dict
   import json
   import numpy as np
   from PIL import Image
   import tensorflow as tf
   from tensorflow import keras

   # Add project root to path
   project_root = Path(__file__).parent.parent.parent.parent
   sys.path.insert(0, str(project_root))

   from src.mnist_classifier.preprocess import normalize_pixels, reshape_images


   class DigitClassifier:
       """
       Command-line interface for MNIST digit classification.

       This class handles:
       - Model loading and caching
       - Image preprocessing
       - Predictions with confidence scores
       - Error handling and user feedback
       """

       def __init__(self, model_path: Optional[str] = None, verbose: bool = False):
           """
           Initialize the digit classifier.

           Args:
               model_path: Path to the trained model file
               verbose: Enable verbose output
           """
           self.verbose = verbose
           self.model = None
           self.model_path = model_path

           # Find and load model
           if model_path:
               self._load_model(model_path)
           else:
               self._find_and_load_model()

       def _find_and_load_model(self):
           """Find and load the most recent trained model."""
           if self.verbose:
               print("🔍 Searching for trained model...")

           # Search paths in order of preference
           search_paths = [
               Path("models/experiments"),
               Path("models"),
               Path(".")
           ]

           model_files = []
           for search_path in search_paths:
               if search_path.exists():
                   # Find all .h5 files
                   model_files.extend(search_path.glob("**/*.h5"))

           if not model_files:
               raise FileNotFoundError(
                   "No trained model found. Please either:\n"
                   "1. Train a model first (python train_model.py)\n"
                   "2. Specify a model path with --model"
               )

           # Use the most recent model
           latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

           if self.verbose:
               print(f"📦 Found model: {latest_model}")

           self._load_model(str(latest_model))

       def _load_model(self, model_path: str):
           """Load a trained model from file."""
           if not os.path.exists(model_path):
               raise FileNotFoundError(f"Model file not found: {model_path}")

           try:
               if self.verbose:
                   print(f"📥 Loading model from {model_path}...")

               self.model = keras.models.load_model(model_path)
               self.model_path = model_path

               if self.verbose:
                   print(f"✅ Model loaded successfully")
                   print(f"   Architecture: {len(self.model.layers)} layers")
                   print(f"   Input shape: {self.model.input_shape}")

           except Exception as e:
               raise RuntimeError(f"Failed to load model: {str(e)}")

       def preprocess_image(self, image_path: str) -> np.ndarray:
           """
           Load and preprocess an image for model input.

           Args:
               image_path: Path to the image file

           Returns:
               Preprocessed image array ready for prediction
           """
           if self.verbose:
               print(f"\n🖼️  Processing image: {image_path}")

           # Check if file exists
           if not os.path.exists(image_path):
               raise FileNotFoundError(f"Image file not found: {image_path}")

           try:
               # Load image
               img = Image.open(image_path)

               if self.verbose:
                   print(f"   Original size: {img.size}")
                   print(f"   Original mode: {img.mode}")

               # Convert to grayscale if needed
               if img.mode != 'L':
                   img = img.convert('L')
                   if self.verbose:
                       print("   Converted to grayscale")

               # Resize to 28x28
               if img.size != (28, 28):
                   # Use high-quality downsampling
                   img = img.resize((28, 28), Image.Resampling.LANCZOS)
                   if self.verbose:
                       print("   Resized to 28x28")

               # Convert to numpy array
               img_array = np.array(img, dtype=np.uint8)

               # Handle inverted images (white on black vs black on white)
               # MNIST uses white digits on black background
               mean_value = np.mean(img_array)
               if mean_value > 127:  # Mostly white = probably black on white
                   img_array = 255 - img_array
                   if self.verbose:
                       print("   Inverted colors (detected black on white)")

               # Reshape for batch dimension
               img_array = img_array.reshape(1, 28, 28)

               # Apply same preprocessing as training
               img_normalized = normalize_pixels(img_array)
               img_input = reshape_images(img_normalized, add_channel=True)

               if self.verbose:
                   print(f"   Final shape: {img_input.shape}")
                   print(f"   Value range: [{img_input.min():.3f}, {img_input.max():.3f}]")

               return img_input

           except Exception as e:
               raise RuntimeError(f"Failed to process image: {str(e)}")

       def predict(self, image_path: str) -> Dict:
           """
           Predict the digit in an image.

           Args:
               image_path: Path to the image file

           Returns:
               Dictionary with prediction results
           """
           # Preprocess image
           img_input = self.preprocess_image(image_path)

           # Make prediction
           if self.verbose:
               print("\n🤖 Running prediction...")

           predictions = self.model.predict(img_input, verbose=0)

           # Get results
           predicted_class = int(np.argmax(predictions[0]))
           confidence = float(predictions[0][predicted_class])

           # Get top 3 predictions
           top3_indices = np.argsort(predictions[0])[-3:][::-1]
           top3_predictions = [
               {
                   "digit": int(idx),
                   "confidence": float(predictions[0][idx])
               }
               for idx in top3_indices
           ]

           results = {
               "image_path": image_path,
               "predicted_digit": predicted_class,
               "confidence": confidence,
               "top_3_predictions": top3_predictions,
               "all_probabilities": {
                   str(i): float(predictions[0][i])
                   for i in range(10)
               }
           }

           return results

       def predict_batch(self, image_paths: List[str]) -> List[Dict]:
           """
           Predict digits for multiple images.

           Args:
               image_paths: List of paths to image files

           Returns:
               List of prediction results
           """
           results = []

           print(f"\n📊 Processing {len(image_paths)} images...")

           for i, image_path in enumerate(image_paths, 1):
               try:
                   print(f"\n[{i}/{len(image_paths)}] {image_path}")
                   result = self.predict(image_path)
                   result["status"] = "success"
                   results.append(result)

               except Exception as e:
                   print(f"   ❌ Error: {str(e)}")
                   results.append({
                       "image_path": image_path,
                       "status": "error",
                       "error": str(e)
                   })

           return results


   def format_prediction_output(result: Dict, verbose: bool = False) -> str:
       """
       Format prediction results for display.

       Args:
           result: Prediction results dictionary
           verbose: Show detailed output

       Returns:
           Formatted string for display
       """
       output_lines = []

       output_lines.append("\n" + "="*50)
       output_lines.append("PREDICTION RESULTS")
       output_lines.append("="*50)

       output_lines.append(f"\nImage: {result['image_path']}")
       output_lines.append(f"Predicted Digit: {result['predicted_digit']}")
       output_lines.append(f"Confidence: {result['confidence']:.1%}")

       if verbose:
           output_lines.append("\nTop 3 Predictions:")
           for i, pred in enumerate(result['top_3_predictions'], 1):
               output_lines.append(
                   f"  {i}. Digit {pred['digit']}: {pred['confidence']:.1%}"
               )

           output_lines.append("\nAll Probabilities:")
           for digit in range(10):
               prob = result['all_probabilities'][str(digit)]
               bar = "█" * int(prob * 20)
               output_lines.append(f"  {digit}: {bar:<20} {prob:.1%}")

       output_lines.append("")
       return "\n".join(output_lines)


   def create_argument_parser() -> argparse.ArgumentParser:
       """
       Create and configure argument parser.

       Returns:
           Configured ArgumentParser instance
       """
       parser = argparse.ArgumentParser(
           description="MNIST Digit Classifier - Classify handwritten digits from images",
           formatter_class=argparse.RawDescriptionHelpFormatter,
           epilog="""
   Examples:
     %(prog)s image.png                    # Classify a single image
     %(prog)s -v image.png                 # Verbose output with probabilities
     %(prog)s -m model.h5 image.png        # Use specific model
     %(prog)s *.png                        # Classify multiple images
     %(prog)s -o results.json *.png        # Save results to JSON file

   Supported image formats: PNG, JPG, JPEG, BMP, GIF, TIFF

   Tips:
     - Images should contain a single digit
     - Works best with dark digits on light background
     - Images are automatically resized to 28x28
     - Use -v/--verbose for debugging
           """
       )

       # Positional arguments
       parser.add_argument(
           'image_paths',
           nargs='+',
           help='Path(s) to image file(s) to classify'
       )

       # Optional arguments
       parser.add_argument(
           '-m', '--model',
           type=str,
           default=None,
           help='Path to trained model file (default: auto-detect latest)'
       )

       parser.add_argument(
           '-v', '--verbose',
           action='store_true',
           help='Enable verbose output with detailed probabilities'
       )

       parser.add_argument(
           '-o', '--output',
           type=str,
           default=None,
           help='Save results to JSON file'
       )

       parser.add_argument(
           '-q', '--quiet',
           action='store_true',
           help='Minimal output (only show predictions)'
       )

       parser.add_argument(
           '--batch',
           action='store_true',
           help='Process images in batch mode'
       )

       return parser


   def main():
       """Main entry point for CLI."""
       # Parse arguments
       parser = create_argument_parser()
       args = parser.parse_args()

       # Expand wildcards on Windows
       import glob
       image_paths = []
       for path in args.image_paths:
           if '*' in path or '?' in path:
               image_paths.extend(glob.glob(path))
           else:
               image_paths.append(path)

       if not image_paths:
           print("Error: No image files found")
           sys.exit(1)

       try:
           # Initialize classifier
           classifier = DigitClassifier(
               model_path=args.model,
               verbose=args.verbose
           )

           # Single image mode
           if len(image_paths) == 1 and not args.batch:
               result = classifier.predict(image_paths[0])

               if not args.quiet:
                   print(format_prediction_output(result, args.verbose))
               else:
                   print(f"{result['predicted_digit']}")

               # Save to file if requested
               if args.output:
                   with open(args.output, 'w') as f:
                       json.dump(result, f, indent=2)
                   if not args.quiet:
                       print(f"\n💾 Results saved to {args.output}")

           # Batch mode
           else:
               results = classifier.predict_batch(image_paths)

               # Display summary
               if not args.quiet:
                   print("\n" + "="*50)
                   print("BATCH PROCESSING SUMMARY")
                   print("="*50)

                   successful = sum(1 for r in results if r.get('status') == 'success')
                   print(f"\nTotal images: {len(image_paths)}")
                   print(f"Successful: {successful}")
                   print(f"Failed: {len(image_paths) - successful}")

                   if successful > 0:
                       print("\nPredictions:")
                       for result in results:
                           if result.get('status') == 'success':
                               print(f"  {result['image_path']}: {result['predicted_digit']} "
                                     f"({result['confidence']:.1%} confidence)")

               # Save to file if requested
               if args.output:
                   with open(args.output, 'w') as f:
                       json.dump(results, f, indent=2)
                   if not args.quiet:
                       print(f"\n💾 Results saved to {args.output}")

       except KeyboardInterrupt:
           print("\n\nOperation cancelled by user")
           sys.exit(1)

       except Exception as e:
           print(f"\n❌ Error: {str(e)}")
           if args.verbose:
               import traceback
               traceback.print_exc()
           sys.exit(1)


   if __name__ == "__main__":
       main()
   ```

### Understanding the Code

Let's break down the key components:

**1. Argument Parser Setup**:
```python
parser.add_argument('image_paths', nargs='+', help='...')
```
- `nargs='+'`: Accept one or more image paths
- Positional argument: No dash prefix needed

**2. Model Discovery**:
```python
model_files.extend(search_path.glob("**/*.h5"))
latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
```
- Searches common locations for model files
- Uses the most recent model by default

**3. Image Preprocessing**:
```python
if mean_value > 127:  # Mostly white
    img_array = 255 - img_array  # Invert
```
- Handles both black-on-white and white-on-black images
- Automatically detects and inverts if needed

**4. Error Handling**:
- FileNotFoundError for missing files
- RuntimeError for processing errors
- Graceful handling with user-friendly messages

### CLI Design Best Practices

1. **Progressive Disclosure**:
   - Basic usage is simple: `predict digit.png`
   - Advanced features available via flags

2. **Helpful Error Messages**:
   ```python
   raise FileNotFoundError(
       "No trained model found. Please either:\n"
       "1. Train a model first (python train_model.py)\n"
       "2. Specify a model path with --model"
   )
   ```

3. **Multiple Output Formats**:
   - Human-readable by default
   - JSON for programmatic use
   - Quiet mode for scripts

---

## Task 4.2: Implement Image Processing

### Understanding Image Preprocessing Requirements

The model was trained on MNIST images with specific characteristics:
- **Size**: 28×28 pixels
- **Color**: Grayscale (single channel)
- **Background**: Black (value 0)
- **Digits**: White (value 255)
- **Normalization**: Pixel values scaled to [0, 1]

User-provided images might have:
- Different sizes (need resizing)
- Color (need grayscale conversion)
- Different backgrounds (might need inversion)
- Various formats (PNG, JPG, etc.)

### Advanced Image Processing Implementation

1. **Create an enhanced image processing module**:
   ```bash
   touch src/mnist_classifier/cli/image_utils.py
   ```

2. **Implement advanced image processing**:

   Add this code to `src/mnist_classifier/cli/image_utils.py`:

   ```python
   """
   Image processing utilities for MNIST digit classification.

   This module provides functions for:
   - Loading images in various formats
   - Preprocessing for model input
   - Handling edge cases and variations
   """

   import numpy as np
   from PIL import Image, ImageOps, ImageFilter
   from typing import Tuple, Optional
   import cv2


   def load_and_preprocess_image(
       image_path: str,
       target_size: Tuple[int, int] = (28, 28),
       invert_threshold: int = 127,
       apply_threshold: bool = True,
       denoise: bool = True
   ) -> np.ndarray:
       """
       Load and preprocess an image for MNIST digit classification.

       This function handles:
       - Various image formats
       - Color to grayscale conversion
       - Size normalization
       - Background/foreground detection
       - Noise reduction
       - Centering and padding

       Args:
           image_path: Path to input image
           target_size: Target dimensions (width, height)
           invert_threshold: Threshold for detecting inverted images
           apply_threshold: Apply binary thresholding
           denoise: Apply denoising filters

       Returns:
           Preprocessed image array (1, 28, 28)
       """
       # Load image
       img = Image.open(image_path)

       # Convert to grayscale if needed
       if img.mode != 'L':
           # Handle RGBA by compositing on white background
           if img.mode == 'RGBA':
               background = Image.new('RGB', img.size, (255, 255, 255))
               background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
               img = background

           img = img.convert('L')

       # Apply denoising if requested
       if denoise:
           # Slight Gaussian blur to reduce noise
           img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

       # Convert to numpy array
       img_array = np.array(img, dtype=np.uint8)

       # Apply thresholding for cleaner digits
       if apply_threshold:
           # Otsu's thresholding for optimal threshold
           _, img_array = cv2.threshold(
               img_array, 0, 255,
               cv2.THRESH_BINARY + cv2.THRESH_OTSU
           )

       # Detect if image needs inversion
       # (MNIST has white digits on black background)
       if np.mean(img_array) > invert_threshold:
           img_array = 255 - img_array

       # Find bounding box of the digit
       img_array = center_and_resize_digit(img_array, target_size)

       # Ensure correct shape
       img_array = img_array.reshape(1, 28, 28)

       return img_array


   def center_and_resize_digit(
       img_array: np.ndarray,
       target_size: Tuple[int, int] = (28, 28)
   ) -> np.ndarray:
       """
       Center the digit and resize to target size.

       This function:
       1. Finds the bounding box of the digit
       2. Centers it in a square canvas
       3. Adds padding to match MNIST style
       4. Resizes to target dimensions

       Args:
           img_array: Input image array
           target_size: Target dimensions

       Returns:
           Centered and resized image
       """
       # Find bounding box of non-zero pixels
       coords = np.column_stack(np.where(img_array > 0))

       if len(coords) == 0:
           # Empty image, return zeros
           return np.zeros(target_size, dtype=np.uint8)

       # Get bounding box
       y_min, x_min = coords.min(axis=0)
       y_max, x_max = coords.max(axis=0)

       # Extract digit region
       digit = img_array[y_min:y_max+1, x_min:x_max+1]

       # Calculate size maintaining aspect ratio
       h, w = digit.shape
       if h > w:
           new_h = 20  # MNIST digits are typically 20x20 within 28x28
           new_w = int(w * (20 / h))
       else:
           new_w = 20
           new_h = int(h * (20 / w))

       # Resize digit
       digit_resized = cv2.resize(
           digit, (new_w, new_h),
           interpolation=cv2.INTER_AREA
       )

       # Create blank canvas
       canvas = np.zeros(target_size, dtype=np.uint8)

       # Calculate position to center the digit
       y_offset = (target_size[0] - new_h) // 2
       x_offset = (target_size[1] - new_w) // 2

       # Place digit on canvas
       canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

       return canvas


   def preprocess_for_model(
       img_array: np.ndarray,
       normalize: bool = True
   ) -> np.ndarray:
       """
       Final preprocessing for model input.

       Args:
           img_array: Image array from previous preprocessing
           normalize: Whether to normalize to [0, 1]

       Returns:
           Model-ready input array (1, 28, 28, 1)
       """
       # Ensure float32 for model
       img_array = img_array.astype(np.float32)

       # Normalize if requested
       if normalize:
           img_array = img_array / 255.0

       # Add channel dimension
       if len(img_array.shape) == 3:
           img_array = np.expand_dims(img_array, axis=-1)

       return img_array


   def validate_image(image_path: str) -> Tuple[bool, Optional[str]]:
       """
       Validate an image file before processing.

       Args:
           image_path: Path to image file

       Returns:
           Tuple of (is_valid, error_message)
       """
       import os

       # Check file exists
       if not os.path.exists(image_path):
           return False, "File not found"

       # Check file size (reasonable limit)
       file_size = os.path.getsize(image_path)
       if file_size > 10 * 1024 * 1024:  # 10MB
           return False, "File too large (>10MB)"

       # Try to open as image
       try:
           img = Image.open(image_path)
           img.verify()  # Verify it's a valid image

           # Re-open after verify (verify closes the file)
           img = Image.open(image_path)

           # Check dimensions aren't too large
           if img.width > 5000 or img.height > 5000:
               return False, "Image dimensions too large (>5000px)"

           return True, None

       except Exception as e:
           return False, f"Invalid image format: {str(e)}"


   def create_debug_visualization(
       original: np.ndarray,
       processed: np.ndarray,
       output_path: str
   ):
       """
       Create a side-by-side visualization for debugging.

       Args:
           original: Original image
           processed: Processed image
           output_path: Where to save visualization
       """
       import matplotlib.pyplot as plt

       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

       # Original image
       ax1.imshow(original.squeeze(), cmap='gray')
       ax1.set_title('Original Image')
       ax1.axis('off')

       # Processed image
       ax2.imshow(processed.squeeze(), cmap='gray')
       ax2.set_title('Processed (Model Input)')
       ax2.axis('off')

       plt.tight_layout()
       plt.savefig(output_path, dpi=150, bbox_inches='tight')
       plt.close()

       print(f"Debug visualization saved to: {output_path}")
   ```

### Handling Common Image Issues

**1. Different Backgrounds**:
```python
# Black digit on white background (opposite of MNIST)
if np.mean(img_array) > 127:
    img_array = 255 - img_array
```

**2. Centered Digits**:
```python
# Find bounding box and center the digit
coords = np.column_stack(np.where(img_array > 0))
# ... extract and center
```

**3. Noise Reduction**:
```python
# Gaussian blur for slight noise reduction
img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
```

**4. Aspect Ratio Preservation**:
```python
# Maintain aspect ratio when resizing
if h > w:
    new_h = 20
    new_w = int(w * (20 / h))
```

### Testing Image Preprocessing

Create a test script to verify image processing:

```python
#!/usr/bin/env python3
"""Test image preprocessing functionality."""

from src.mnist_classifier.cli.image_utils import (
    load_and_preprocess_image,
    create_debug_visualization
)

# Test with various images
test_images = [
    "test_images/digit_photo.jpg",      # Photo of handwritten digit
    "test_images/digit_scan.png",       # Scanned digit
    "test_images/digit_digital.png",    # Digitally created
    "test_images/digit_inverted.png",   # White background
]

for img_path in test_images:
    try:
        # Load and preprocess
        processed = load_and_preprocess_image(img_path)

        # Create visualization
        original = Image.open(img_path).convert('L')
        create_debug_visualization(
            np.array(original),
            processed,
            f"debug_{Path(img_path).stem}.png"
        )

        print(f"✅ Processed: {img_path}")

    except Exception as e:
        print(f"❌ Failed: {img_path} - {str(e)}")
```

---

## Task 4.3: Update Documentation

### Adding CLI Usage to README

Good documentation is crucial for CLI tools. Users should be able to understand how to use your tool without reading the source code.

### Step-by-Step Instructions

1. **Update the main README.md**:

   Add this section to your `README.md` after the Quick Start section:

   ```markdown
   ## 🖥️ Command-Line Usage

   After training a model, you can classify digit images using the CLI:

   ### Basic Usage

   ```bash
   # Classify a single image
   python -m src.mnist_classifier.cli.predict digit.png

   # Or use the convenience script
   python predict_digit.py digit.png
   ```

   ### Advanced Options

   ```bash
   # Verbose output with all probabilities
   python predict_digit.py -v digit.png

   # Use a specific model
   python predict_digit.py --model models/my_model.h5 digit.png

   # Batch processing
   python predict_digit.py *.png

   # Save results to JSON
   python predict_digit.py -o results.json digit.png

   # Quiet mode (only output predicted digit)
   python predict_digit.py -q digit.png
   ```

   ### Supported Image Formats

   The CLI supports common image formats:
   - PNG (recommended)
   - JPEG/JPG
   - BMP
   - GIF
   - TIFF

   ### Image Requirements

   For best results:
   - Use dark digits on light background (or vice versa)
   - Ensure good contrast between digit and background
   - Center the digit in the image
   - Avoid excessive noise or artifacts

   The tool automatically:
   - ✅ Converts color images to grayscale
   - ✅ Resizes images to 28×28 pixels
   - ✅ Inverts colors if needed (white/black background)
   - ✅ Centers and pads digits appropriately

   ### Examples

   1. **Quick classification**:
      ```bash
      $ python predict_digit.py my_digit.png

      ================================================
      PREDICTION RESULTS
      ================================================

      Image: my_digit.png
      Predicted Digit: 7
      Confidence: 99.8%
      ```

   2. **Detailed analysis**:
      ```bash
      $ python predict_digit.py -v my_digit.png

      ================================================
      PREDICTION RESULTS
      ================================================

      Image: my_digit.png
      Predicted Digit: 7
      Confidence: 99.8%

      Top 3 Predictions:
        1. Digit 7: 99.8%
        2. Digit 1: 0.1%
        3. Digit 9: 0.1%

      All Probabilities:
        0: ░░░░░░░░░░░░░░░░░░░░ 0.0%
        1: ░░░░░░░░░░░░░░░░░░░░ 0.1%
        2: ░░░░░░░░░░░░░░░░░░░░ 0.0%
        3: ░░░░░░░░░░░░░░░░░░░░ 0.0%
        4: ░░░░░░░░░░░░░░░░░░░░ 0.0%
        5: ░░░░░░░░░░░░░░░░░░░░ 0.0%
        6: ░░░░░░░░░░░░░░░░░░░░ 0.0%
        7: ████████████████████ 99.8%
        8: ░░░░░░░░░░░░░░░░░░░░ 0.0%
        9: ░░░░░░░░░░░░░░░░░░░░ 0.1%
      ```

   3. **Batch processing**:
      ```bash
      $ python predict_digit.py digits/*.png

      📊 Processing 5 images...

      [1/5] digits/sample1.png
      [2/5] digits/sample2.png
      [3/5] digits/sample3.png
      [4/5] digits/sample4.png
      [5/5] digits/sample5.png

      ================================================
      BATCH PROCESSING SUMMARY
      ================================================

      Total images: 5
      Successful: 5
      Failed: 0

      Predictions:
        digits/sample1.png: 3 (98.7% confidence)
        digits/sample2.png: 7 (99.9% confidence)
        digits/sample3.png: 1 (97.3% confidence)
        digits/sample4.png: 9 (99.5% confidence)
        digits/sample5.png: 2 (98.2% confidence)
      ```

   ### Troubleshooting

   **"No trained model found"**
   - Train a model first: `python train_model.py`
   - Or specify a model path: `--model path/to/model.h5`

   **"Image file not found"**
   - Check the file path is correct
   - Use absolute paths if needed

   **Poor predictions**
   - Ensure image has good contrast
   - Try inverting colors manually
   - Check if digit is centered and clear
   - Use `-v` flag to see confidence scores

   **"Failed to process image"**
   - Ensure file is a valid image format
   - Check file isn't corrupted
   - Try converting to PNG format
   ```

2. **Create a dedicated CLI documentation file**:

   ```bash
   touch docs/cli_usage.md
   ```

   Add comprehensive CLI documentation:

   ```markdown
   # MNIST Classifier CLI Documentation

   ## Overview

   The MNIST Classifier CLI provides a command-line interface for classifying handwritten digits using a trained neural network model. It supports single image classification, batch processing, and various output formats.

   ## Installation

   Ensure you have completed the project setup and trained a model:

   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Train a model (if not already done)
   python train_model.py
   ```

   ## Basic Usage

   ### Single Image Classification

   The simplest usage is to classify a single image:

   ```bash
   python predict_digit.py image.png
   ```

   This will:
   1. Load the most recent trained model
   2. Preprocess the image
   3. Make a prediction
   4. Display the result

   ### Command Structure

   ```
   python predict_digit.py [OPTIONS] IMAGE_PATH [IMAGE_PATH ...]
   ```

   ## Options

   ### `-m, --model MODEL_PATH`
   Specify a custom model file to use instead of auto-detecting the latest model.

   ```bash
   python predict_digit.py --model models/best_model.h5 digit.png
   ```

   ### `-v, --verbose`
   Enable verbose output showing detailed prediction probabilities.

   ```bash
   python predict_digit.py -v digit.png
   ```

   ### `-o, --output OUTPUT_FILE`
   Save prediction results to a JSON file.

   ```bash
   python predict_digit.py -o results.json digit.png
   ```

   ### `-q, --quiet`
   Minimal output mode - only prints the predicted digit.

   ```bash
   python predict_digit.py -q digit.png
   # Output: 7
   ```

   ### `--batch`
   Force batch processing mode even for a single image.

   ```bash
   python predict_digit.py --batch digit.png
   ```

   ## Image Preprocessing

   The CLI automatically preprocesses images to match the MNIST format:

   ### Automatic Preprocessing Steps

   1. **Color Conversion**: RGB/RGBA images are converted to grayscale
   2. **Resizing**: Images are resized to 28×28 pixels using high-quality resampling
   3. **Inversion Detection**: Automatically detects and inverts if needed (MNIST uses white digits on black)
   4. **Normalization**: Pixel values are normalized to [0, 1] range
   5. **Centering**: Digits are centered within the 28×28 frame

   ### Supported Formats

   - **PNG** (Portable Network Graphics) - Recommended
   - **JPEG/JPG** (Joint Photographic Experts Group)
   - **BMP** (Bitmap)
   - **GIF** (Graphics Interchange Format)
   - **TIFF/TIF** (Tagged Image File Format)
   - **WEBP** (Web Picture format)

   ## Output Formats

   ### Human-Readable (Default)

   ```
   ================================================
   PREDICTION RESULTS
   ================================================

   Image: digit.png
   Predicted Digit: 7
   Confidence: 99.8%
   ```

   ### Verbose Output

   With `-v` flag:

   ```
   ================================================
   PREDICTION RESULTS
   ================================================

   Image: digit.png
   Predicted Digit: 7
   Confidence: 99.8%

   Top 3 Predictions:
     1. Digit 7: 99.8%
     2. Digit 1: 0.1%
     3. Digit 9: 0.1%

   All Probabilities:
     0: ░░░░░░░░░░░░░░░░░░░░ 0.0%
     1: ░░░░░░░░░░░░░░░░░░░░ 0.1%
     ...
     7: ████████████████████ 99.8%
     ...
   ```

   ### JSON Output

   With `-o results.json`:

   ```json
   {
     "image_path": "digit.png",
     "predicted_digit": 7,
     "confidence": 0.998,
     "top_3_predictions": [
       {"digit": 7, "confidence": 0.998},
       {"digit": 1, "confidence": 0.001},
       {"digit": 9, "confidence": 0.001}
     ],
     "all_probabilities": {
       "0": 0.0001,
       "1": 0.001,
       ...
     }
   }
   ```

   ## Batch Processing

   Process multiple images at once:

   ```bash
   # Using wildcards
   python predict_digit.py *.png

   # Multiple specific files
   python predict_digit.py digit1.png digit2.jpg digit3.bmp

   # From a directory
   python predict_digit.py images/*.png
   ```

   ### Batch Output

   ```
   📊 Processing 3 images...

   [1/3] digit1.png
   [2/3] digit2.jpg
   [3/3] digit3.bmp

   ================================================
   BATCH PROCESSING SUMMARY
   ================================================

   Total images: 3
   Successful: 3
   Failed: 0

   Predictions:
     digit1.png: 3 (98.7% confidence)
     digit2.jpg: 7 (99.9% confidence)
     digit3.bmp: 1 (97.3% confidence)
   ```

   ## Error Handling

   The CLI provides clear error messages:

   ### Model Not Found
   ```
   ❌ Error: No trained model found. Please either:
   1. Train a model first (python train_model.py)
   2. Specify a model path with --model
   ```

   ### Invalid Image
   ```
   ❌ Error: Failed to process image: Invalid image format
   ```

   ### File Not Found
   ```
   ❌ Error: Image file not found: missing.png
   ```

   ## Best Practices

   ### For Best Results

   1. **Image Quality**:
      - Use clear, high-contrast images
      - Ensure digit is well-defined
      - Avoid blurry or pixelated images

   2. **Image Content**:
      - Single digit per image
      - Digit should fill most of the frame
      - Minimal background noise

   3. **Format**:
      - PNG format recommended for quality
      - Avoid heavily compressed JPEGs

   ### Creating Test Images

   You can create test images using:
   - Drawing apps (MS Paint, GIMP)
   - Tablet/phone drawing apps
   - Scanning handwritten digits
   - Screenshots of digit displays

   ## Integration Examples

   ### Python Script Integration

   ```python
   import subprocess
   import json

   # Run prediction
   result = subprocess.run(
       ['python', 'predict_digit.py', '-q', 'digit.png'],
       capture_output=True,
       text=True
   )

   predicted_digit = int(result.stdout.strip())
   ```

   ### Bash Script Integration

   ```bash
   #!/bin/bash
   # Process all images in a directory

   for img in images/*.png; do
       digit=$(python predict_digit.py -q "$img")
       echo "$img: $digit"
   done
   ```

   ### JSON Processing

   ```python
   import subprocess
   import json

   # Get detailed results
   subprocess.run([
       'python', 'predict_digit.py',
       '-o', 'results.json',
       'digit.png'
   ])

   # Load and process results
   with open('results.json') as f:
       data = json.load(f)
       print(f"Digit: {data['predicted_digit']}")
       print(f"Confidence: {data['confidence']:.1%}")
   ```

   ## Performance Considerations

   - **Model Loading**: First prediction takes longer due to model loading
   - **Batch Processing**: More efficient for multiple images
   - **Image Size**: Large images take longer to process
   - **GPU**: Uses GPU if available for faster predictions

   ## Debugging

   Use verbose mode to debug issues:

   ```bash
   python predict_digit.py -v problematic_image.png
   ```

   This shows:
   - Image loading details
   - Preprocessing steps
   - Model information
   - Detailed predictions

   ## Future Enhancements

   Planned features:
   - Real-time camera input
   - Confidence threshold filtering
   - Multiple digit detection
   - Web API endpoint
   - GUI interface option
   ```

---

## Helper Scripts

### predict_digit.py - Main CLI Entry Point

Create this convenience script in the project root:

```python
#!/usr/bin/env python3
"""
Convenience script for digit prediction CLI.

This script provides a simple entry point for the MNIST digit classifier.
It's equivalent to running: python -m src.mnist_classifier.cli.predict
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the CLI
from src.mnist_classifier.cli.predict import main

if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x predict_digit.py
```

### batch_predict.py - Batch Processing Script

Create this script for convenient batch processing:

```python
#!/usr/bin/env python3
"""
Batch prediction script for processing multiple digit images.

This script provides enhanced batch processing capabilities:
- Directory traversal
- Result summarization
- CSV export
- Error reporting
"""

import argparse
import sys
import os
import csv
import json
from pathlib import Path
from typing import List, Dict
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mnist_classifier.cli.predict import DigitClassifier


def find_images(input_paths: List[str], recursive: bool = False) -> List[str]:
    """
    Find all image files from input paths.

    Args:
        input_paths: List of paths (files or directories)
        recursive: Search directories recursively

    Returns:
        List of image file paths
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}
    image_files = []

    for path in input_paths:
        path_obj = Path(path)

        if path_obj.is_file():
            # Single file
            if path_obj.suffix.lower() in image_extensions:
                image_files.append(str(path_obj))

        elif path_obj.is_dir():
            # Directory
            if recursive:
                # Recursive search
                for ext in image_extensions:
                    image_files.extend([
                        str(p) for p in path_obj.rglob(f"*{ext}")
                    ])
            else:
                # Non-recursive
                for ext in image_extensions:
                    image_files.extend([
                        str(p) for p in path_obj.glob(f"*{ext}")
                    ])

        else:
            print(f"Warning: Path not found: {path}")

    return sorted(list(set(image_files)))  # Remove duplicates and sort


def process_images(
    image_paths: List[str],
    model_path: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Process multiple images and return results.

    Args:
        image_paths: List of image file paths
        model_path: Optional path to model file
        verbose: Enable verbose output

    Returns:
        List of prediction results
    """
    # Initialize classifier
    print("🚀 Initializing classifier...")
    classifier = DigitClassifier(model_path=model_path, verbose=verbose)

    results = []
    start_time = time.time()

    print(f"\n📊 Processing {len(image_paths)} images...")
    print("-" * 50)

    for i, image_path in enumerate(image_paths, 1):
        # Progress indicator
        progress = i / len(image_paths) * 100
        print(f"\r[{i}/{len(image_paths)}] {progress:.1f}% - {Path(image_path).name}",
              end="", flush=True)

        try:
            result = classifier.predict(image_path)
            result["status"] = "success"
            result["error"] = None
            results.append(result)

        except Exception as e:
            results.append({
                "image_path": image_path,
                "status": "error",
                "error": str(e),
                "predicted_digit": None,
                "confidence": None
            })

    elapsed_time = time.time() - start_time
    print(f"\n\n✅ Processing complete in {elapsed_time:.1f} seconds")
    print(f"   Average: {elapsed_time/len(image_paths):.3f} seconds per image")

    return results


def save_results(results: List[Dict], output_format: str, output_path: str):
    """
    Save results in specified format.

    Args:
        results: List of prediction results
        output_format: Format ('json' or 'csv')
        output_path: Output file path
    """
    if output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    elif output_format == 'csv':
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'image_path', 'predicted_digit', 'confidence',
                'status', 'error'
            ])

            # Data rows
            for result in results:
                writer.writerow([
                    result['image_path'],
                    result.get('predicted_digit', ''),
                    f"{result.get('confidence', 0):.3f}" if result.get('confidence') else '',
                    result.get('status', 'error'),
                    result.get('error', '')
                ])

    print(f"\n💾 Results saved to: {output_path}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"\nTotal images: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        # Confidence statistics
        confidences = [r['confidence'] for r in successful]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        print(f"\nConfidence Statistics:")
        print(f"  Average: {avg_confidence:.1%}")
        print(f"  Min: {min_confidence:.1%}")
        print(f"  Max: {max_confidence:.1%}")

        # Digit distribution
        digit_counts = {}
        for r in successful:
            digit = r['predicted_digit']
            digit_counts[digit] = digit_counts.get(digit, 0) + 1

        print(f"\nDigit Distribution:")
        for digit in sorted(digit_counts.keys()):
            count = digit_counts[digit]
            bar = "█" * int(count / len(successful) * 40)
            print(f"  {digit}: {bar} {count}")

    if failed:
        print(f"\nFailed Images:")
        for r in failed[:5]:  # Show first 5
            print(f"  - {r['image_path']}: {r['error']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process digit images for classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s images/                    # Process all images in directory
  %(prog)s -r images/                 # Recursive directory search
  %(prog)s -o results.csv images/     # Save results to CSV
  %(prog)s --format json -o results.json *.png  # Save as JSON

Output formats:
  CSV: Spreadsheet-compatible format with one row per image
  JSON: Structured data with full prediction details
        """
    )

    parser.add_argument(
        'paths',
        nargs='+',
        help='Image files or directories to process'
    )

    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Search directories recursively'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='Path to model file (default: auto-detect)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Save results to file'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json'],
        default='csv',
        help='Output format (default: csv)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary statistics'
    )

    args = parser.parse_args()

    try:
        # Find all image files
        image_paths = find_images(args.paths, args.recursive)

        if not image_paths:
            print("❌ No image files found")
            sys.exit(1)

        print(f"📁 Found {len(image_paths)} image(s)")

        # Process images
        results = process_images(
            image_paths,
            model_path=args.model,
            verbose=args.verbose
        )

        # Save results if requested
        if args.output:
            save_results(results, args.format, args.output)

        # Print summary
        if not args.no_summary:
            print_summary(results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### create_test_images.py - Test Image Generator

Create this helper script to generate test images:

```python
#!/usr/bin/env python3
"""
Generate test images for the MNIST digit classifier.

This script creates various test images to verify the CLI functionality:
- Different sizes
- Different backgrounds
- Different styles
- Edge cases
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def create_test_images_directory():
    """Create directory for test images."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    return test_dir


def create_digit_image(
    digit: int,
    size: tuple = (28, 28),
    background: str = 'black',
    style: str = 'normal'
) -> Image.Image:
    """
    Create a test digit image.

    Args:
        digit: Digit to draw (0-9)
        size: Image size
        background: 'black' or 'white'
        style: Drawing style

    Returns:
        PIL Image
    """
    # Create base image
    bg_color = 0 if background == 'black' else 255
    fg_color = 255 if background == 'black' else 0

    # Create larger image for better quality
    large_size = (size[0] * 4, size[1] * 4)
    img = Image.new('L', large_size, bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a font, fall back to default if not available
    try:
        font_size = int(large_size[1] * 0.7)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Draw digit
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = (
        (large_size[0] - text_width) // 2,
        (large_size[1] - text_height) // 2
    )

    draw.text(position, text, fill=fg_color, font=font)

    # Apply style modifications
    if style == 'noisy':
        # Add noise
        pixels = np.array(img)
        noise = np.random.normal(0, 25, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)

    elif style == 'blurry':
        # Apply blur
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=2))

    # Resize to target size
    img = img.resize(size, Image.Resampling.LANCZOS)

    return img


def generate_test_set():
    """Generate a comprehensive test set."""
    test_dir = create_test_images_directory()

    print("🎨 Generating test images...")

    # Standard test images (28x28, black background)
    print("\n1. Standard MNIST-style images:")
    for digit in range(10):
        img = create_digit_image(digit)
        path = test_dir / f"standard_digit_{digit}.png"
        img.save(path)
        print(f"   Created: {path}")

    # Inverted images (white background)
    print("\n2. Inverted (white background) images:")
    for digit in [0, 1, 7, 9]:  # Test subset
        img = create_digit_image(digit, background='white')
        path = test_dir / f"inverted_digit_{digit}.png"
        img.save(path)
        print(f"   Created: {path}")

    # Different sizes
    print("\n3. Different sized images:")
    sizes = [(56, 56), (100, 100), (14, 14)]
    for size in sizes:
        img = create_digit_image(5, size=size)
        path = test_dir / f"size_{size[0]}x{size[1]}_digit_5.png"
        img.save(path)
        print(f"   Created: {path}")

    # Noisy images
    print("\n4. Noisy images:")
    for digit in [3, 8]:
        img = create_digit_image(digit, style='noisy')
        path = test_dir / f"noisy_digit_{digit}.png"
        img.save(path)
        print(f"   Created: {path}")

    # Edge cases
    print("\n5. Edge cases:")

    # Empty image
    empty = Image.new('L', (28, 28), 0)
    path = test_dir / "empty_black.png"
    empty.save(path)
    print(f"   Created: {path} (empty black)")

    # All white
    white = Image.new('L', (28, 28), 255)
    path = test_dir / "empty_white.png"
    white.save(path)
    print(f"   Created: {path} (empty white)")

    # Very small digit
    img = Image.new('L', (28, 28), 0)
    draw = ImageDraw.Draw(img)
    draw.text((12, 12), "1", fill=255)
    path = test_dir / "tiny_digit_1.png"
    img.save(path)
    print(f"   Created: {path} (tiny digit)")

    print(f"\n✅ Generated {len(list(test_dir.glob('*.png')))} test images in {test_dir}/")
    print("\nTest the CLI with:")
    print(f"  python predict_digit.py {test_dir}/*.png")


if __name__ == "__main__":
    generate_test_set()
```

### verify_milestone4.py - Verification Script

Create this comprehensive verification script:

```python
#!/usr/bin/env python3
"""
Verification script for Milestone 4 completion.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import importlib.util


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def check_files_exist():
    """Check if all required files exist."""
    print("\n📁 Checking required files:")

    required_files = [
        ('src/mnist_classifier/cli/__init__.py', 'CLI package init'),
        ('src/mnist_classifier/cli/predict.py', 'Main CLI module'),
        ('predict_digit.py', 'CLI entry point script'),
        ('docs/cli_usage.md', 'CLI documentation'),
    ]

    optional_files = [
        ('src/mnist_classifier/cli/image_utils.py', 'Image utilities'),
        ('batch_predict.py', 'Batch processing script'),
        ('create_test_images.py', 'Test image generator'),
    ]

    all_exist = True
    for filepath, description in required_files:
        exists = os.path.exists(filepath)
        print(f"  {check_mark(exists)} {description}: {filepath}")
        if not exists:
            all_exist = False

    print("\n📄 Optional files:")
    for filepath, description in optional_files:
        exists = os.path.exists(filepath)
        print(f"  {check_mark(exists)} {description}: {filepath}")

    return all_exist


def check_cli_imports():
    """Check if CLI modules can be imported."""
    print("\n📦 Checking CLI imports:")

    try:
        # Add project root to path
        sys.path.insert(0, os.path.abspath('.'))

        # Try importing the CLI module
        from src.mnist_classifier.cli.predict import (
            DigitClassifier, create_argument_parser, main
        )
        print(f"  {check_mark(True)} CLI module imports successfully")

        # Check key components
        components = [
            (DigitClassifier, "DigitClassifier class"),
            (create_argument_parser, "Argument parser function"),
            (main, "Main entry point"),
        ]

        all_good = True
        for component, name in components:
            exists = component is not None
            print(f"  {check_mark(exists)} {name} exists")
            if not exists:
                all_good = False

        return all_good

    except Exception as e:
        print(f"  {check_mark(False)} Failed to import CLI: {str(e)}")
        return False


def check_cli_functionality():
    """Test basic CLI functionality."""
    print("\n🔧 Testing CLI functionality:")

    # Check if predict_digit.py exists and is executable
    if not os.path.exists('predict_digit.py'):
        print(f"  {check_mark(False)} predict_digit.py not found")
        return False

    # Test help message
    try:
        result = subprocess.run(
            [sys.executable, 'predict_digit.py', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )

        help_ok = result.returncode == 0 and 'usage:' in result.stdout.lower()
        print(f"  {check_mark(help_ok)} Help message displays correctly")

        # Check for key arguments in help
        expected_args = ['image_paths', '--model', '--verbose', '--output']
        for arg in expected_args:
            present = arg in result.stdout
            print(f"  {check_mark(present)} Argument '{arg}' documented")

        return help_ok

    except Exception as e:
        print(f"  {check_mark(False)} CLI test failed: {str(e)}")
        return False


def check_image_processing():
    """Check if image processing works."""
    print("\n🖼️  Testing image processing:")

    try:
        from src.mnist_classifier.cli.predict import DigitClassifier

        # Create a simple test image
        from PIL import Image
        import numpy as np

        # Create a test image
        test_img = Image.new('L', (28, 28), 0)  # Black background
        pixels = np.array(test_img)
        # Draw a simple vertical line (like digit 1)
        pixels[5:23, 13:15] = 255
        test_img = Image.fromarray(pixels)

        test_path = "test_digit_temp.png"
        test_img.save(test_path)

        # Test preprocessing
        try:
            classifier = DigitClassifier(verbose=False)
            processed = classifier.preprocess_image(test_path)

            processing_ok = (
                processed.shape == (1, 28, 28, 1) and
                processed.min() >= 0 and
                processed.max() <= 1
            )

            print(f"  {check_mark(processing_ok)} Image preprocessing works")
            print(f"    Shape: {processed.shape}")
            print(f"    Range: [{processed.min():.3f}, {processed.max():.3f}]")

            # Clean up
            os.remove(test_path)

            return processing_ok

        except Exception as e:
            print(f"  {check_mark(False)} Preprocessing failed: {str(e)}")
            if os.path.exists(test_path):
                os.remove(test_path)
            return False

    except ImportError:
        print(f"  {check_mark(False)} Could not import required modules")
        return False


def check_documentation():
    """Check if documentation is comprehensive."""
    print("\n📚 Checking documentation:")

    # Check README.md updates
    readme_path = 'README.md'
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            readme_content = f.read()

        cli_section = '## 🖥️ Command-Line Usage' in readme_content or 'CLI' in readme_content
        print(f"  {check_mark(cli_section)} README.md includes CLI section")
    else:
        print(f"  {check_mark(False)} README.md not found")
        cli_section = False

    # Check dedicated CLI documentation
    cli_docs_path = 'docs/cli_usage.md'
    if os.path.exists(cli_docs_path):
        with open(cli_docs_path, 'r') as f:
            cli_docs = f.read()

        sections = [
            ('## Basic Usage', 'Basic usage section'),
            ('## Options', 'Options documentation'),
            ('## Image Preprocessing', 'Preprocessing explanation'),
            ('## Troubleshooting', 'Troubleshooting section'),
        ]

        all_sections = True
        for section, description in sections:
            present = section in cli_docs
            print(f"  {check_mark(present)} {description}")
            if not present:
                all_sections = False

        # Check length
        word_count = len(cli_docs.split())
        adequate = word_count > 500
        print(f"  {check_mark(adequate)} Adequate documentation ({word_count} words)")

        return cli_section and all_sections and adequate
    else:
        print(f"  {check_mark(False)} CLI documentation not found")
        return False


def test_end_to_end():
    """Test end-to-end CLI functionality if possible."""
    print("\n🚀 Testing end-to-end functionality:")

    # First check if we have a trained model
    model_paths = [
        Path("models/experiments"),
        Path("models"),
    ]

    model_found = False
    for path in model_paths:
        if path.exists():
            h5_files = list(path.glob("**/*.h5"))
            if h5_files:
                model_found = True
                break

    if not model_found:
        print(f"  ℹ️  No trained model found - skipping end-to-end test")
        print("     Train a model first: python train_model.py")
        return True  # Not a failure, just skipped

    # Create a test image
    try:
        from PIL import Image, ImageDraw

        # Create a clear digit image
        img = Image.new('L', (28, 28), 0)
        draw = ImageDraw.Draw(img)
        # Draw a '7'
        draw.line([(8, 8), (20, 8)], fill=255, width=2)  # Top line
        draw.line([(20, 8), (14, 20)], fill=255, width=2)  # Diagonal

        test_path = "test_e2e_digit.png"
        img.save(test_path)

        # Run prediction
        result = subprocess.run(
            [sys.executable, 'predict_digit.py', '-q', test_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check result
        success = result.returncode == 0
        if success:
            try:
                predicted = int(result.stdout.strip())
                print(f"  {check_mark(True)} End-to-end test successful")
                print(f"    Predicted digit: {predicted}")
            except:
                print(f"  {check_mark(False)} Invalid output format")
                success = False
        else:
            print(f"  {check_mark(False)} Prediction failed")
            if result.stderr:
                print(f"    Error: {result.stderr}")

        # Clean up
        os.remove(test_path)

        return success

    except Exception as e:
        print(f"  {check_mark(False)} End-to-end test error: {str(e)}")
        if os.path.exists(test_path):
            os.remove(test_path)
        return False


def main():
    """Run all verification checks."""
    print("🔍 Milestone 4 Verification")
    print("=" * 50)

    # Make sure we're in the right directory
    if not os.path.exists('src/mnist_classifier'):
        print("❌ Error: Not in project root directory!")
        print("Please run this from the mnist_classifier project root.")
        sys.exit(1)

    # Run all checks
    checks = [
        ("Required Files", check_files_exist()),
        ("CLI Imports", check_cli_imports()),
        ("CLI Functionality", check_cli_functionality()),
        ("Image Processing", check_image_processing()),
        ("Documentation", check_documentation()),
        ("End-to-End Test", test_end_to_end()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 4 is complete!")
        print("\nYou have successfully:")
        print("  ✓ Built a command-line interface for digit classification")
        print("  ✓ Implemented robust image preprocessing")
        print("  ✓ Added comprehensive error handling")
        print("  ✓ Created clear documentation")
        print("\nYour classifier is now usable from the command line!")
        print("\nTry it out:")
        print("  python predict_digit.py test_images/*.png")
        print("\nNext: Milestone 5 - Build an interactive web interface!")
    else:
        print("\n⚠️  Some checks failed. Please review and fix the issues above.")
        print("\nTips:")
        print("  - Ensure all required files are created")
        print("  - Check that imports are correct")
        print("  - Test the CLI manually: python predict_digit.py --help")
        print("  - Make sure you have a trained model")


if __name__ == "__main__":
    main()
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Import Errors

**Problem**: "ModuleNotFoundError: No module named 'src'"
```bash
# Solution 1: Run from project root
cd /path/to/mnist_classifier
python predict_digit.py image.png

# Solution 2: Install package in development mode
pip install -e .
```

**Problem**: "No module named 'PIL'"
```bash
# Install Pillow
pip install Pillow
```

#### Model Loading Issues

**Problem**: "No trained model found"
```bash
# Train a model first
python train_model.py --quick  # Quick 5-epoch training

# Or specify model path
python predict_digit.py --model path/to/model.h5 image.png
```

**Problem**: "Failed to load model"
- Check TensorFlow version compatibility
- Ensure model file isn't corrupted
- Try loading in Python directly

#### Image Processing Issues

**Problem**: "Failed to process image: Invalid image format"
```bash
# Check file is actually an image
file image.png

# Try converting to PNG
convert image.jpg image.png  # Using ImageMagick

# Or use PIL
python -c "from PIL import Image; Image.open('image.jpg').save('image.png')"
```

**Problem**: Poor predictions on user images
1. **Check image quality**:
   - Is the digit clear and well-defined?
   - Good contrast between digit and background?

2. **Try preprocessing**:
   ```python
   # Invert colors
   from PIL import Image, ImageOps
   img = Image.open('digit.png')
   img_inverted = ImageOps.invert(img)
   img_inverted.save('digit_inverted.png')
   ```

3. **Debug with verbose mode**:
   ```bash
   python predict_digit.py -v problematic_image.png
   ```

#### Performance Issues

**Problem**: Slow first prediction
- Normal behavior - model loading takes time
- Consider keeping model in memory for batch processing

**Problem**: Slow image processing
```python
# Profile the code
python -m cProfile predict_digit.py image.png

# Use batch mode for multiple images
python batch_predict.py images/*.png
```

### Platform-Specific Issues

#### Windows

**Problem**: Wildcard expansion doesn't work
```batch
REM Windows doesn't expand wildcards by default
REM Use batch script or Python glob

python -c "import glob; print(' '.join(glob.glob('*.png')))" > files.txt
python predict_digit.py @files.txt
```

#### macOS

**Problem**: Font issues in test image generation
```python
# Use system fonts
font_paths = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
```

#### Linux

**Problem**: Missing image libraries
```bash
# Install system dependencies
sudo apt-get install libpng-dev libjpeg-dev

# Reinstall Pillow
pip uninstall Pillow
pip install --no-cache-dir Pillow
```

---

## Learning Resources

### Command-Line Interface Design

1. **Python argparse**:
   - [Official argparse tutorial](https://docs.python.org/3/howto/argparse.html)
   - [Real Python argparse guide](https://realpython.com/command-line-interfaces-python-argparse/)
   - [Click library](https://click.palletsprojects.com/) (alternative to argparse)

2. **CLI Best Practices**:
   - [12 Factor CLI Apps](https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46)
   - [Command Line Interface Guidelines](https://clig.dev/)
   - Unix Philosophy and CLI design

3. **Error Handling**:
   - Python exception handling best practices
   - User-friendly error messages
   - Logging vs printing

### Image Processing

1. **Pillow (PIL)**:
   - [Pillow documentation](https://pillow.readthedocs.io/)
   - Image manipulation tutorials
   - Format conversion guides

2. **OpenCV** (optional):
   - [OpenCV Python tutorials](https://opencv-python-tutroals.readthedocs.io/)
   - Advanced image preprocessing
   - Computer vision techniques

3. **Image Preprocessing for ML**:
   - Normalization techniques
   - Data augmentation
   - Handling different image formats

### Testing and Validation

1. **Testing CLI Applications**:
   - pytest with CLI testing
   - Mocking file systems
   - Integration testing

2. **Image Testing**:
   - Creating test datasets
   - Edge case handling
   - Performance testing

---

## Milestone 4 Checklist

Before moving to Milestone 5, ensure you've completed:

- [ ] Created CLI package structure (`src/mnist_classifier/cli/`)
- [ ] Implemented main CLI module (`predict.py`)
- [ ] Created argument parser with help messages
- [ ] Implemented image loading and preprocessing
- [ ] Handled different image formats and sizes
- [ ] Added color inversion detection
- [ ] Implemented single image prediction
- [ ] Added batch processing support
- [ ] Created verbose and quiet output modes
- [ ] Added JSON output option
- [ ] Proper error handling throughout
- [ ] Updated README.md with CLI usage
- [ ] Created comprehensive CLI documentation
- [ ] Created helper scripts (predict_digit.py, batch_predict.py)
- [ ] All verification tests pass

🎉 **Congratulations on completing Milestone 4!**

You've successfully:
- Built a professional command-line interface
- Made your model accessible to users
- Handled real-world image variations
- Created comprehensive documentation
- Learned CLI development best practices

### What You've Learned
- Building CLIs with argparse
- Image processing with Pillow
- Error handling and user feedback
- Documentation for developer tools
- Testing and validation strategies

### Your CLI Achievements
- **Flexible**: Handles various image formats and sizes
- **Robust**: Comprehensive error handling
- **User-friendly**: Clear help messages and feedback
- **Powerful**: Batch processing and multiple output formats
- **Professional**: Well-documented and tested

Ready to build a web interface? In Milestone 5, you'll create an interactive web application where users can draw digits and get instant predictions!
