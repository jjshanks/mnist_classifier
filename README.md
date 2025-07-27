# MNIST Digit Classifier

A comprehensive machine learning application that classifies handwritten digits using deep learning, featuring both CLI and web interfaces.

## 🎯 Project Overview

This project implements a complete ML pipeline for digit recognition:
- Convolutional Neural Network (CNN) trained on MNIST dataset
- Command-line interface for single image predictions
- Interactive web application with real-time drawing
- Visualization of neural network activations

## 🛠️ Technologies Used

- **Python 3.13**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **FastAPI**: Modern web framework
- **uv**: Fast Python package manager
- **NumPy**: Numerical computations
- **Pillow**: Image processing

## 📋 Prerequisites

- Python 3.13 or higher
- Git
- 4GB RAM minimum
- 500MB free disk space

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mnist_classifier
   ```

2. **Set up environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv sync --all-extras
   ```

4. **Test data pipeline** (Milestone 2 complete!):
   ```bash
   python quick_setup_data.py
   ```

5. **Train the model** (Milestone 3 complete!):
   ```bash
   # Quick training (5 epochs)
   ./train_gpu.sh --quick

   # Full training (10 epochs recommended)
   ./train_gpu.sh --epochs 10
   ```

6. **Run the web app** (after implementing):
   ```bash
   uvicorn src.main:app --reload
   ```

## 📁 Project Structure

```
mnist_classifier/
├── src/
│   └── mnist_classifier/      # Main package
│       ├── __init__.py       # Package initialization
│       ├── data/             # Data loading and preprocessing
│       ├── models/           # Neural network models
│       ├── training/         # Training pipeline
│       └── cli/              # Command-line interface
├── tests/                    # Test suite
├── data/                     # Dataset storage (gitignored)
├── models/                   # Saved models
├── notebooks/                # Jupyter notebooks
├── docs/                     # Documentation
├── static/                   # Frontend assets (future)
├── templates/                # HTML templates (future)
├── predict_digit.py          # CLI tool for single predictions
├── batch_predict.py          # CLI tool for batch predictions
├── create_test_images.py     # Generate test images
└── README.md                 # This file
```

## 🎓 Learning Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## 🖥️ Command-Line Interface

### Single Image Prediction

```bash
# Basic usage
python predict_digit.py path/to/image.png

# With custom model
python predict_digit.py image.jpg --model models/my_model.keras

# JSON output
python predict_digit.py digit.png --output-format json

# Quiet mode (digit only)
python predict_digit.py digit.png --quiet
```

### Batch Processing

```bash
# Process multiple images
python batch_predict.py images/*.png

# Save results to CSV
python batch_predict.py images/*.jpg --output results.csv

# JSON output
python batch_predict.py images/*.png --format json

# Recursive directory search
python batch_predict.py images/ --recursive
```

### Create Test Images

```bash
# Create synthetic test images
python create_test_images.py test_images/

# Create from MNIST dataset
python create_test_images.py test_images/ --mnist

# Create all types
python create_test_images.py test_images/ --all
```

## 📝 Development Roadmap

- [x] Milestone 1: Project setup
- [x] Milestone 2: Data pipeline
- [x] Milestone 3: Model training (CNN with >98% accuracy)
- [x] Milestone 4: CLI interface
- [ ] Milestone 5: Web interface
- [ ] Milestone 6: Visualizations

### Current Features

- **CNN Model**: Two convolutional layers achieving >98% accuracy
- **GPU Support**: Optimized training with CUDA acceleration
- **CLI Tools**: Command-line interface for predictions
- **Batch Processing**: Process multiple images efficiently
- **Multiple Output Formats**: Human-readable, JSON, CSV
- **Image Preprocessing**: Automatic normalization and inversion detection
- **Test Image Generation**: Create synthetic and MNIST-based test images

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

## 📄 License

This project is for educational purposes.
