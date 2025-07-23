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

5. **Train the model** (after implementing):
   ```bash
   python src/train.py
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
│       ├── load_data.py      # MNIST data loading module
│       ├── preprocess.py     # Data preprocessing functions
│       ├── data_pipeline.py  # Integrated data pipeline
│       ├── example_usage.py  # Pipeline usage examples
│       └── py.typed          # Type checking marker
├── tests/                    # Test suite
├── data/                     # Dataset storage (gitignored)
│   ├── processed/            # Cached preprocessed data
│   └── samples/              # Sample images
├── notebooks/                # Jupyter notebooks
│   └── 01_data_exploration.ipynb  # Data analysis notebook
├── docs/                     # Documentation
│   ├── ROADMAP.md           # Project roadmap
│   └── MILESTONE_*.md       # Detailed milestone guides
├── static/                   # Frontend assets (future)
├── templates/                # HTML templates (future)
├── pyproject.toml            # Project configuration
├── uv.lock                   # Locked dependencies
├── CLAUDE.md                 # AI assistant instructions
├── quick_setup_data.py       # Quick data setup script
├── verify_milestone2.py      # Milestone verification
└── README.md                 # This file
```

## 🎓 Learning Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## 📝 Development Roadmap

- [x] Milestone 1: Project setup
- [x] Milestone 2: Data pipeline (Complete! ✨)
- [ ] Milestone 3: Model training
- [ ] Milestone 4: CLI interface
- [ ] Milestone 5: Web interface
- [ ] Milestone 6: Visualizations

### Current Features (Milestone 2)

- **Data Loading**: Automatic MNIST dataset download and caching
- **Preprocessing**: Normalization, reshaping, one-hot encoding
- **Data Pipeline**: Integrated pipeline with caching for efficiency
- **Data Exploration**: Jupyter notebook with visualizations
- **Validation Split**: Automatic train/validation splitting
- **Batch Generator**: Memory-efficient data loading for training

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

## 📄 License

This project is for educational purposes.
