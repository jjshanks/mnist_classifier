# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MNIST digit classifier project using TensorFlow and FastAPI. **ALL 6 MILESTONES ARE NOW COMPLETE!** 🎉 The project is production-ready with comprehensive neural network visualizations, interactive web interface, and professional documentation. It's configured with modern Python 3.13 tooling and follows best practices for ML projects.

## Essential Commands

### Development Environment
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync --all-extras
```

### Code Quality
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Format code
uv run ruff format

# Lint and auto-fix
uv run ruff check --fix

# Type checking
uv run mypy src tests

# Security scanning
uv run bandit -c pyproject.toml -r src
```

### Testing
```bash
# Run tests with coverage
uv run pytest
```

### Running the Application
```bash
# Test data pipeline (Milestone 2)
uv run python quick_setup_data.py

# Run data pipeline verification
uv run python verify_milestone2.py

# Train model with GPU support (Milestone 3)
./train_gpu.sh --epochs 10  # Recommended: handles GPU setup automatically
# Or: uv run python train_model.py --epochs 10

# Quick training test (5 epochs)
./train_gpu.sh --quick

# Run the complete web application (COMPLETED!)
python run_web_app.py

# Or start web server directly
uv run uvicorn src.mnist_classifier.api.app:app --reload
```

## Architecture and Structure

### Core Components
1. **Data Pipeline** (✅ Implemented in Milestone 2):
   - `load_data.py`: MNIST dataset downloading and loading
   - `preprocess.py`: Data normalization, reshaping, and encoding
   - `data_pipeline.py`: Integrated pipeline with caching
2. **Model** (`src/mnist_classifier/models/`): TensorFlow neural network implementation (✅ Implemented in Milestone 3)
   - `cnn_model.py`: CNN architecture with two convolutional layers
   - Achieves >98% accuracy on MNIST
3. **Training** (`src/mnist_classifier/training/`): Model training, validation, and checkpoint management (✅ Implemented in Milestone 3)
   - `train.py`: Comprehensive training pipeline with callbacks
   - `quick_train.py`: Quick training script for testing
4. **CLI** (`src/mnist_classifier/cli/`): Command-line interface for training and prediction (✅ Implemented in Milestone 4)
   - `predict.py`: CLI prediction tool with batch processing
   - Professional argument parsing and error handling
5. **Web API** (`src/mnist_classifier/api/`): FastAPI endpoints for predictions (✅ Implemented in Milestone 5)
   - `app.py`: Complete FastAPI application with neural network visualizations
   - Real-time predictions and model introspection endpoints
6. **Frontend** (`templates/` and `static/`): Interactive web interface (✅ Implemented in Milestones 5 & 6)
   - Complete drawing canvas with neural network visualizations
   - Educational content and interactive layer exploration

### Key Technical Decisions
- **Package Manager**: Uses `uv` for fast, reliable dependency management
- **Config**: All tool configurations centralized in `pyproject.toml`
- **Type Safety**: Strict mypy configuration with `py.typed` marker
- **Testing**: pytest with coverage reporting configured
- **Code Quality**: ruff for linting/formatting, pre-commit hooks configured
- **Path Handling**: Uses `pathlib.Path` instead of `os.path` throughout the codebase
- **Pre-commit Hooks**: Automated checks for code quality, type safety, and security

### Implementation Roadmap
The project follows a structured 6-milestone plan - **ALL COMPLETED!** ✅
1. ✅ Project Foundation and Setup (Complete)
2. ✅ Data Acquisition and Preparation (Complete)
3. ✅ Neural Network for Digit Classification (Complete)
4. ✅ Command-Line Interface (CLI) (Complete)
5. ✅ Interactive Web Interface (Complete)
6. ✅ Neural Network Visualization (Complete)

Detailed milestone documentation is in `docs/ROADMAP.md` and individual milestone files.

### Development Notes
- All source code goes in `src/mnist_classifier/`
- Use type hints throughout the codebase
- Follow existing code structure patterns when adding new modules
- The `data/` directory is gitignored for dataset storage
- Run tests before committing changes
- Pre-commit hooks automatically check code quality - run `uv run pre-commit run --all-files` before committing
- Use `pathlib.Path` for all file operations, not `os.path`
- Break complex assertions into multiple simple assertions for better error messages

## GPU Support and Troubleshooting

### GPU Setup for WSL2
The project includes full GPU support for training. Use `./train_gpu.sh` which automatically:
- Detects WSL2 environment
- Configures CUDA library paths
- Sets appropriate environment variables

### Important GPU Dependencies
The project uses `tensorflow[and-cuda]` which includes ALL necessary NVIDIA libraries:
- CUDA runtime, cuDNN, cuBLAS, cuFFT, etc.
- Do NOT install just `tensorflow` - it lacks GPU support libraries

### Common GPU Issues and Solutions

1. **"Error loading CUDA libraries. GPU will not be used"**
   - **Cause**: Missing CUDA runtime libraries
   - **Solution**: Ensure `tensorflow[and-cuda]` is installed (not just `tensorflow`)
   - **Check**: Run `uv pip list | grep nvidia` - should show 10+ nvidia packages

2. **"Cannot dlopen some GPU libraries"**
   - **Cause**: LD_LIBRARY_PATH not set correctly
   - **Solution**: Use `./train_gpu.sh` which sets paths automatically
   - **Manual fix**: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$(uv run python -c "import site; print(site.getsitepackages()[0])")/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH`

3. **Model saving error: "Invalid filepath extension"**
   - **Cause**: TensorFlow 2.19+ requires explicit extensions
   - **Solution**: Use `model.export()` for SavedModel format, not `model.save()`

4. **Missing sklearn for confusion matrix**
   - **Solution**: `scikit-learn` is now included in dependencies

### Verifying GPU Setup
```bash
# Quick GPU diagnostic
uv run python diagnose_gpu.py

# Check if GPU is detected
uv run python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Performance Notes
- GPU training is 10-50x faster than CPU
- Expect ~4-5ms per step on RTX 3080 Ti
- Memory growth is enabled to prevent TensorFlow from allocating all GPU memory
