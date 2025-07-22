# MNIST Digit Classifier - Implementation Roadmap

## Project Overview
**Goal:** Build a comprehensive machine learning application that classifies handwritten digits from the MNIST dataset with both CLI and web interfaces.

**Core Technologies:**
- Python 3.13
- Virtual Environment: uv
- Web Framework: FastAPI
- ML Library: TensorFlow (with Keras API)
- Dataset: MNIST

## Implementation Progress
- [ ] Milestone 1: Project Foundation and Setup
- [ ] Milestone 2: Data Acquisition and Preparation
- [ ] Milestone 3: Neural Network for Digit Classification
- [ ] Milestone 4: Command-Line Interface (CLI)
- [ ] Milestone 5: Interactive Web Interface
- [ ] Milestone 6: Visualizing the Neural Network State

---

## Milestone 1: Project Foundation and Setup

### Objective
Establish a clean, organized, and reproducible project structure from an empty directory.

### Implementation Steps

#### 1.1 Create Project Directory Structure
- [ ] Create `src/` directory for Python source code
- [ ] Create `data/` directory for MNIST dataset storage
- [ ] Create `notebooks/` directory for Jupyter notebooks
- [ ] Create `docs/` directory for documentation
- [ ] Create `static/` directory for CSS and JavaScript files
- [ ] Create `templates/` directory for HTML templates

#### 1.2 Initialize Version Control
- [ ] Run `git init` in project root
- [ ] Create `.gitignore` file with content:
  ```
  __pycache__/
  *.pyc
  *.pyo
  *.pyd
  .Python
  .venv/
  venv/
  ENV/
  env/
  .env
  *.egg-info/
  dist/
  build/
  *.egg
  .DS_Store
  .idea/
  .vscode/
  *.swp
  *.swo
  data/mnist/
  *.h5
  *.keras
  .ipynb_checkpoints/
  ```

#### 1.3 Setup Virtual Environment
- [ ] Run `uv venv` to create virtual environment
- [ ] Activate environment: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)

#### 1.4 Install Dependencies
- [ ] Run `uv pip install fastapi uvicorn[standard] numpy tensorflow pillow jupyter matplotlib seaborn`
- [ ] Generate requirements.txt: `uv pip freeze > requirements.txt`

#### 1.5 Create Initial Documentation
- [ ] Create `README.md` with basic project description
- [ ] Commit initial project structure to git

### Acceptance Criteria
- ✓ All directories exist in correct structure
- ✓ Git repository initialized with proper .gitignore
- ✓ Virtual environment created and activated
- ✓ All dependencies installed successfully
- ✓ requirements.txt file generated
- ✓ README.md created with project overview
- ✓ Initial commit made to git

---

## Milestone 2: Data Acquisition and Preparation

### Prerequisites
- Milestone 1 completed

### Implementation Steps

#### 2.1 Create Data Loading Script
Create `src/load_data.py`:
```python
import tensorflow as tf
import numpy as np
import os

def load_mnist(data_dir='data/'):
    """Load MNIST dataset and return train/test splits"""
    # Implementation here
    pass
```

#### 2.2 Create Data Exploration Notebook
Create `notebooks/01_data_exploration.ipynb`:
- [ ] Import necessary libraries
- [ ] Load MNIST data
- [ ] Visualize sample images (5x5 grid)
- [ ] Plot label distribution
- [ ] Display dataset statistics (shapes, data types)

#### 2.3 Create Preprocessing Module
Create `src/preprocess.py`:
```python
def normalize_pixels(images):
    """Normalize pixel values to [0, 1]"""
    pass

def reshape_images(images):
    """Add channel dimension (28, 28) -> (28, 28, 1)"""
    pass

def one_hot_encode_labels(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    pass
```

#### 2.4 Update Data Loading with Preprocessing
- [ ] Integrate preprocessing functions into load_data.py
- [ ] Add train/validation split functionality

### Acceptance Criteria
- ✓ `python -c "from src.load_data import load_mnist; load_mnist()"` runs without errors
- ✓ Data saved to `data/` directory
- ✓ Jupyter notebook displays sample digits and statistics
- ✓ Preprocessing functions properly documented
- ✓ Data returned in correct format: (x_train, y_train), (x_test, y_test)
- ✓ Preprocessed data has shape (n, 28, 28, 1) and values in [0, 1]

---

## Milestone 3: Neural Network for Digit Classification

### Prerequisites
- Milestone 2 completed

### Implementation Steps

#### 3.1 Create Model Architecture
Create `src/model.py`:
```python
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    """Create CNN model for MNIST classification"""
    model = keras.Sequential([
        # Conv2D layer 1
        # MaxPooling2D
        # Conv2D layer 2
        # MaxPooling2D
        # Flatten
        # Dense layer
        # Output Dense layer with softmax
    ])
    return model
```

#### 3.2 Create Training Script
Create `src/train.py`:
- [ ] Import data loading and preprocessing
- [ ] Build model from model.py
- [ ] Compile with Adam optimizer, categorical_crossentropy loss
- [ ] Train with validation split
- [ ] Save model as 'mnist_model.h5'

#### 3.3 Document Model Architecture
Create `docs/model_architecture.md`:
- [ ] Explain each layer and its purpose
- [ ] Include model summary output
- [ ] Justify architectural choices

### Acceptance Criteria
- ✓ `src/model.py` contains create_model() function
- ✓ `python src/train.py` runs complete training pipeline
- ✓ Training displays loss/accuracy per epoch
- ✓ Achieves >98% validation accuracy
- ✓ Model saved as `mnist_model.h5`
- ✓ Architecture documentation complete and clear

---

## Milestone 4: Command-Line Interface (CLI)

### Prerequisites
- Milestone 3 completed

### Implementation Steps

#### 4.1 Create CLI Script
Create `src/cli.py`:
```python
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Classify a digit image')
    parser.add_argument('--image_path', required=True, help='Path to digit image')
    # Implementation here
    pass

if __name__ == '__main__':
    main()
```

#### 4.2 Implement Image Processing
- [ ] Load image with Pillow
- [ ] Convert to grayscale
- [ ] Resize to 28x28
- [ ] Apply same preprocessing as training data

#### 4.3 Update Documentation
- [ ] Add CLI usage section to README.md
- [ ] Include example command and expected output

### Acceptance Criteria
- ✓ `python src/cli.py --image_path test_digit.png` outputs prediction
- ✓ Handles file not found errors gracefully
- ✓ Handles invalid image formats
- ✓ Preprocessing matches training preprocessing
- ✓ README includes clear CLI usage instructions

---

## Milestone 5: Interactive Web Interface

### Prerequisites
- Milestone 4 completed

### Implementation Steps

#### 5.1 Create FastAPI Application
Create `src/main.py`:
```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI()

@app.get("/")
async def home():
    # Serve index.html
    pass

@app.post("/predict")
async def predict(request: Request):
    # Handle prediction
    pass
```

#### 5.2 Create HTML Frontend
Create `templates/index.html`:
- [ ] Canvas element for drawing (id="canvas")
- [ ] Predict button (id="predict-btn")
- [ ] Clear button (id="clear-btn")
- [ ] Result display area (id="result")

#### 5.3 Create JavaScript Logic
Create `static/js/main.js`:
- [ ] Canvas drawing functionality
- [ ] Clear button handler
- [ ] Predict button handler with fetch API
- [ ] Result display logic

#### 5.4 Mount Static Files
- [ ] Configure FastAPI to serve static files
- [ ] Test all endpoints

### Acceptance Criteria
- ✓ `uvicorn src.main:app --reload` starts server
- ✓ http://127.0.0.1:8000 displays drawing interface
- ✓ Drawing on canvas works smoothly
- ✓ Clear button erases canvas
- ✓ Predict button sends image and displays result
- ✓ Backend processes canvas data correctly

---

## Milestone 6: Visualizing the Neural Network State

### Prerequisites
- Milestone 5 completed

### Implementation Steps

#### 6.1 Create Visualization Model
Update `src/model.py`:
```python
def create_visualization_model():
    """Create model that outputs intermediate activations"""
    # Load base model
    # Create new model with multiple outputs
    pass
```

#### 6.2 Update Backend for Visualizations
Update `/predict` endpoint in `src/main.py`:
- [ ] Use visualization model
- [ ] Extract layer activations
- [ ] Format activation data for frontend
- [ ] Include prediction probabilities

#### 6.3 Enhance Frontend Visualizations
Update `templates/index.html` and `static/js/main.js`:
- [ ] Add activation visualization containers
- [ ] Add probability bar chart container
- [ ] Implement activation rendering logic
- [ ] Implement probability chart with labels

#### 6.4 Add Explanatory Text
- [ ] Add user-friendly explanations for visualizations
- [ ] Style visualization sections appropriately

### Acceptance Criteria
- ✓ Prediction displays feature maps from conv layers
- ✓ Bar chart shows probability for each digit (0-9)
- ✓ Visualizations update with each prediction
- ✓ Explanatory text helps users understand visualizations
- ✓ UI remains responsive and user-friendly

---

## Testing Guidelines

### Unit Tests
- Test preprocessing functions
- Test model creation
- Test CLI argument parsing

### Integration Tests
- Test data pipeline end-to-end
- Test web API endpoints
- Test model prediction accuracy

### Manual Testing
- Draw various digits in web interface
- Test with different image formats in CLI
- Verify visualization accuracy

## Deployment Considerations

1. **Model Optimization**
   - Consider model quantization for faster inference
   - Implement model caching in web server

2. **Web Security**
   - Add input validation for image uploads
   - Implement rate limiting

3. **Documentation**
   - API documentation with FastAPI's built-in docs
   - User guide for both CLI and web interfaces

## Success Metrics

- Model accuracy: >98% on test set
- Web interface response time: <500ms
- CLI processing time: <1s per image
- Clean, maintainable code structure
- Comprehensive documentation