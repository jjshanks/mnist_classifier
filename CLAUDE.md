# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MNIST digit classifier project using TensorFlow and FastAPI. The project is currently at Milestone 1 (Project Foundation and Setup) of a 6-milestone roadmap. It's configured with modern Python 3.13 tooling and follows best practices for ML projects.

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
# Format code
ruff format

# Lint and auto-fix
ruff check --fix

# Type checking
mypy src tests

# Security scanning
bandit -c pyproject.toml -r src
```

### Testing
```bash
# Run tests with coverage
pytest
```

### Running the Application
```bash
# Train model (once implemented)
python src/train.py

# Start web server
uvicorn src.main:app --reload
```

## Architecture and Structure

### Core Components (To Be Implemented)
1. **Data Pipeline** (`src/mnist_classifier/data/`): MNIST dataset downloading, preprocessing, and augmentation
2. **Model** (`src/mnist_classifier/models/`): TensorFlow neural network implementation
3. **Training** (`src/mnist_classifier/training/`): Model training, validation, and checkpoint management
4. **CLI** (`src/mnist_classifier/cli/`): Command-line interface for training and prediction
5. **Web API** (`src/mnist_classifier/api/`): FastAPI endpoints for predictions
6. **Frontend** (`templates/` and `static/`): Interactive web interface

### Key Technical Decisions
- **Package Manager**: Uses `uv` for fast, reliable dependency management
- **Config**: All tool configurations centralized in `pyproject.toml`
- **Type Safety**: Strict mypy configuration with `py.typed` marker
- **Testing**: pytest with coverage reporting configured
- **Code Quality**: ruff for linting/formatting, pre-commit hooks ready

### Implementation Roadmap
The project follows a structured 6-milestone plan:
1. ✓ Project Foundation and Setup (Current)
2. Data Acquisition and Preparation
3. Neural Network for Digit Classification
4. Command-Line Interface (CLI)
5. Interactive Web Interface
6. Visualizing the Neural Network State

Detailed milestone documentation is in `docs/ROADMAP.md` and individual milestone files.

### Development Notes
- All source code goes in `src/mnist_classifier/`
- Use type hints throughout the codebase
- Follow existing code structure patterns when adding new modules
- The `data/` directory is gitignored for dataset storage
- Run tests before committing changes
