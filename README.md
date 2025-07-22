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
   uv sync
   ```

4. **Train the model** (after implementing):
   ```bash
   python src/train.py
   ```

5. **Run the web app** (after implementing):
   ```bash
   uvicorn src.main:app --reload
   ```

## 📁 Project Structure

```
mnist_classifier/
├── src/              # Source code
├── data/             # Dataset storage
├── notebooks/        # Jupyter notebooks
├── docs/             # Documentation
├── static/           # Frontend assets
├── templates/        # HTML templates
├── pyproject.toml    # Project configuration and dependencies
└── README.md         # This file
```

## 🎓 Learning Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## 📝 Development Roadmap

- [x] Milestone 1: Project setup
- [ ] Milestone 2: Data pipeline
- [ ] Milestone 3: Model training
- [ ] Milestone 4: CLI interface
- [ ] Milestone 5: Web interface
- [ ] Milestone 6: Visualizations

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

## 📄 License

This project is for educational purposes.
