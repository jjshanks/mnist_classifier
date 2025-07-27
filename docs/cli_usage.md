# MNIST Digit Classifier - CLI Usage Guide

This guide provides comprehensive documentation for using the MNIST digit classifier command-line tools.

## Table of Contents

1. [Installation](#installation)
2. [Single Image Prediction](#single-image-prediction)
3. [Batch Processing](#batch-processing)
4. [Creating Test Images](#creating-test-images)
5. [Image Requirements](#image-requirements)
6. [Troubleshooting](#troubleshooting)

## Installation

Before using the CLI tools, ensure you have:

1. Completed the project setup (Milestone 1)
2. Trained a model (Milestone 3)
3. Activated your virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

## Single Image Prediction

The `predict_digit.py` script classifies individual handwritten digit images.

### Basic Usage

```bash
python predict_digit.py path/to/image.png
```

### Options

- `--model PATH`: Specify a custom model path (default: `models/mnist_model`)
- `--output-format {verbose,quiet,json}`: Choose output format (default: verbose)
- `--confidence-threshold FLOAT`: Minimum confidence for prediction (0.0-1.0)

### Examples

**Verbose output (default):**
```bash
$ python predict_digit.py test_images/digit_7.png

Predicted digit: 7
Confidence: 99.82%

Probabilities for each digit:
  0: [░░░░░░░░░░░░░░░░░░░░] 0.00%
  1: [░░░░░░░░░░░░░░░░░░░░] 0.01%
  2: [░░░░░░░░░░░░░░░░░░░░] 0.03%
  3: [░░░░░░░░░░░░░░░░░░░░] 0.02%
  4: [░░░░░░░░░░░░░░░░░░░░] 0.00%
  5: [░░░░░░░░░░░░░░░░░░░░] 0.01%
  6: [░░░░░░░░░░░░░░░░░░░░] 0.00%
  7: [████████████████████] 99.82%
  8: [░░░░░░░░░░░░░░░░░░░░] 0.08%
  9: [░░░░░░░░░░░░░░░░░░░░] 0.03%
```

**Quiet output:**
```bash
$ python predict_digit.py test_images/digit_7.png --quiet
7
```

**JSON output:**
```bash
$ python predict_digit.py test_images/digit_7.png --output-format json
{
  "predicted_digit": 7,
  "confidence": 0.9982,
  "probabilities": {
    "0": 0.0000,
    "1": 0.0001,
    "2": 0.0003,
    "3": 0.0002,
    "4": 0.0000,
    "5": 0.0001,
    "6": 0.0000,
    "7": 0.9982,
    "8": 0.0008,
    "9": 0.0003
  }
}
```

**With confidence threshold:**
```bash
$ python predict_digit.py unclear_image.png --confidence-threshold 0.8
Error: Prediction confidence (65.43%) below threshold (80.00%)
```

## Batch Processing

The `batch_predict.py` script processes multiple images at once.

### Basic Usage

```bash
python batch_predict.py images/*.png
```

### Options

- `--model PATH`: Specify a custom model path
- `--output PATH`: Save results to file (CSV or JSON)
- `--format {table,csv,json}`: Output format (default: table)
- `--recursive`: Recursively search directories for images

### Examples

**Process all PNG files in a directory:**
```bash
$ python batch_predict.py test_images/*.png

Found 10 image(s) to process...

Filename                   | Digit | Confidence | Status
--------------------------------------------------------
digit_0.png                |     0 |     99.91% | success
digit_1.png                |     1 |     99.85% | success
digit_2.png                |     2 |     99.76% | success
digit_3.png                |     3 |     99.88% | success
digit_4.png                |     4 |     99.79% | success
digit_5.png                |     5 |     99.83% | success
digit_6.png                |     6 |     99.90% | success
digit_7.png                |     7 |     99.82% | success
digit_8.png                |     8 |     99.77% | success
digit_9.png                |     9 |     99.84% | success

Processed 10 images: 10 successful, 0 failed
```

**Save results to CSV:**
```bash
$ python batch_predict.py images/*.jpg --output results.csv
Found 25 image(s) to process...
Results saved to results.csv
```

**JSON output to stdout:**
```bash
$ python batch_predict.py images/*.png --format json
[
  {
    "filename": "images/digit_0.png",
    "status": "success",
    "predicted_digit": 0,
    "confidence": 0.9991,
    "error": null
  },
  {
    "filename": "images/digit_1.png",
    "status": "success",
    "predicted_digit": 1,
    "confidence": 0.9985,
    "error": null
  }
]
```

**Recursive directory search:**
```bash
$ python batch_predict.py test_data/ --recursive
Found 150 image(s) to process...
```

## Creating Test Images

The `create_test_images.py` script generates test images for the classifier.

### Basic Usage

```bash
python create_test_images.py output_directory/
```

### Options

- `--mnist`: Create images from MNIST dataset
- `--synthetic`: Create synthetic digit images (default)
- `--edge-cases`: Create edge case test images
- `--all`: Create all types of test images
- `--count N`: Number of images per digit (default: 3)

### Examples

**Create synthetic test images:**
```bash
$ python create_test_images.py test_images/

Creating synthetic test images...
Created test_images/synthetic_digit_0_normal.png and test_images/synthetic_digit_0_normal_large.png
Created test_images/synthetic_digit_0_inverted.png and test_images/synthetic_digit_0_inverted_large.png
...

Test images created in test_images/
You can now test the classifier with:
  python predict_digit.py test_images/<image_file>
  python batch_predict.py test_images/*.png
```

**Create images from MNIST:**
```bash
$ python create_test_images.py mnist_samples/ --mnist --count 5
Loading MNIST dataset...
Created mnist_samples/mnist_digit_0_1.png
Created mnist_samples/mnist_digit_0_2.png
...
```

**Create edge cases for testing:**
```bash
$ python create_test_images.py edge_cases/ --edge-cases
Creating edge case test images...
Created edge_case_blank.png
Created edge_case_faint.png
Created edge_case_noise.png
Created edge_case_multiple.png
```

## Image Requirements

### Supported Formats
- PNG (recommended)
- JPEG/JPG
- GIF
- BMP
- TIFF

### Image Specifications
- **Any size**: Images are automatically resized to 28x28 pixels
- **Any color mode**: Images are converted to grayscale
- **Background**: The classifier expects dark background with light digits (MNIST style)
  - Light backgrounds are automatically inverted
  - For best results, use black background with white digits

### Preprocessing Steps
1. Convert to grayscale (if color)
2. Resize to 28x28 pixels
3. Normalize pixel values to [0, 1]
4. Detect and invert if necessary (light background)
5. Reshape for model input

## Troubleshooting

### Common Issues

**Model not found:**
```
Error: Model not found at models/mnist_model
```
Solution: Train the model first with `./train_gpu.sh` or specify a different model path

**Image file not found:**
```
Error: Image file not found: path/to/image.png
```
Solution: Check the file path and ensure the image exists

**Invalid image format:**
```
Error: Failed to load or preprocess image: cannot identify image file
```
Solution: Ensure the file is a valid image in a supported format

**Low confidence prediction:**
```
Error: Prediction confidence (45.23%) below threshold (80.00%)
```
Solution: The image might be unclear, noisy, or contain multiple digits

### Tips for Best Results

1. **Clear Images**: Use high-contrast images with clear digit shapes
2. **Single Digits**: Each image should contain only one digit
3. **Centered**: Digits should be roughly centered in the image
4. **Dark Background**: Use black/dark background with white/light digits
5. **Appropriate Size**: While any size works, images close to 28x28 will have less distortion

### Performance Considerations

- **Model Loading**: The model is loaded once per batch for efficiency
- **Batch Processing**: Use `batch_predict.py` for multiple images
- **GPU Support**: Predictions use GPU if available (same as training)

## Advanced Usage

### Integration with Other Tools

**Process images and analyze results:**
```bash
# Generate predictions
python batch_predict.py images/*.png --output results.json

# Analyze with jq
cat results.json | jq '.[] | select(.confidence < 0.9)'
```

**Automated testing pipeline:**
```bash
#!/bin/bash
# test_pipeline.sh

# Create test images
python create_test_images.py test_batch/ --all --count 10

# Run predictions
python batch_predict.py test_batch/*.png --output test_results.csv

# Check results
python -c "
import csv
with open('test_results.csv') as f:
    reader = csv.DictReader(f)
    results = list(reader)
    success_rate = sum(1 for r in results if r['status'] == 'success') / len(results)
    print(f'Success rate: {success_rate:.1%}')
"
```

### Custom Model Paths

If you've trained multiple models:
```bash
# Use specific model version
python predict_digit.py image.png --model models/mnist_model_v2

# Use exported SavedModel format
python predict_digit.py image.png --model models/saved_model/

# Use checkpoint from training
python predict_digit.py image.png --model checkpoints/best_model.keras
```
