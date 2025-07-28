# Milestone 5: Interactive Web Interface - Detailed Implementation Guide

## Overview

Welcome to Milestone 5! You've built a powerful digit classifier and made it accessible via command-line. Now it's time to create an interactive web interface that allows users to draw digits directly in their browser and get instant predictions. This milestone bridges machine learning with web development, creating a complete end-to-end application.

### What You'll Learn
- **Web Development with FastAPI**: Building modern, fast web APIs with Python
- **Frontend Development**: Creating interactive interfaces with HTML, CSS, and JavaScript
- **Canvas API**: Implementing drawing functionality in the browser
- **AJAX/Fetch**: Asynchronous communication between frontend and backend
- **Base64 Encoding**: Transmitting image data between client and server
- **Web Design**: Creating user-friendly, responsive interfaces

### Why Web Interfaces Matter
Web interfaces make your ML models accessible to everyone:
- No installation required - just open a browser
- Cross-platform compatibility
- Easy sharing and deployment
- Rich, interactive user experiences
- Foundation for production deployment

### Prerequisites
Before starting, ensure you have:
- ✅ Completed Milestone 4 (working CLI)
- ✅ Trained model with >98% accuracy
- ✅ Basic HTML/CSS knowledge
- ✅ Basic JavaScript understanding
- ✅ About 60-90 minutes to complete all tasks

### Success Criteria
By the end of this milestone, you will have:
- ✅ A FastAPI web application serving your model
- ✅ An interactive drawing canvas for digit input
- ✅ Real-time predictions displayed to users
- ✅ Clear, intuitive user interface
- ✅ Proper error handling and user feedback
- ✅ Mobile-responsive design

---

## Task 5.1: Create FastAPI Application

### Understanding FastAPI

FastAPI is a modern web framework for building APIs with Python. It offers:
- **Fast**: Very high performance, on par with NodeJS and Go
- **Easy**: Designed to be easy to use and learn
- **Standard-based**: Based on OpenAPI and JSON Schema
- **Automatic docs**: Interactive API documentation out of the box
- **Type hints**: Leverages Python type hints for validation

### Web Application Architecture

Our application will follow this architecture:
```
User Browser
    ↓ (draws digit)
JavaScript
    ↓ (sends image data)
FastAPI Backend
    ↓ (preprocesses image)
TensorFlow Model
    ↓ (makes prediction)
FastAPI Backend
    ↓ (formats response)
JavaScript
    ↓ (displays result)
User Browser
```

### Step-by-Step Instructions

1. **Create the API directory structure**:
   ```bash
   mkdir -p src/mnist_classifier/api
   touch src/mnist_classifier/api/__init__.py
   ```

2. **Create the main FastAPI application**:
   ```bash
   touch src/mnist_classifier/api/app.py
   ```

3. **Implement the FastAPI application**:

   Add this code to `src/mnist_classifier/api/app.py`:

   ```python
   """
   FastAPI Web Application for MNIST Digit Classification

   This module provides a web API and interface for classifying
   handwritten digits drawn by users in their browser.
   """

   import os
   import sys
   from pathlib import Path
   from typing import Optional, Dict, Any
   import base64
   import io
   from datetime import datetime
   import json

   from fastapi import FastAPI, Request, HTTPException, status
   from fastapi.responses import HTMLResponse, JSONResponse
   from fastapi.staticfiles import StaticFiles
   from fastapi.templating import Jinja2Templates
   from fastapi.middleware.cors import CORSMiddleware
   from pydantic import BaseModel, Field
   import numpy as np
   from PIL import Image
   import tensorflow as tf
   from tensorflow import keras

   # Add project root to path
   project_root = Path(__file__).parent.parent.parent.parent
   sys.path.insert(0, str(project_root))

   from src.mnist_classifier.preprocess import normalize_pixels, reshape_images
   from src.mnist_classifier.cli.predict import DigitClassifier


   # Create FastAPI app
   app = FastAPI(
       title="MNIST Digit Classifier",
       description="Interactive web interface for classifying handwritten digits",
       version="1.0.0"
   )

   # Add CORS middleware for development
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # In production, specify exact origins
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

   # Setup templates and static files
   templates_dir = project_root / "templates"
   static_dir = project_root / "static"

   # Create directories if they don't exist
   templates_dir.mkdir(exist_ok=True)
   static_dir.mkdir(exist_ok=True)
   (static_dir / "css").mkdir(exist_ok=True)
   (static_dir / "js").mkdir(exist_ok=True)

   # Configure templates
   templates = Jinja2Templates(directory=str(templates_dir))

   # Mount static files
   app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


   # Global model instance
   model_instance = None


   class PredictionRequest(BaseModel):
       """Request model for digit prediction."""
       image: str = Field(..., description="Base64 encoded image data")

       class Config:
           json_schema_extra = {
               "example": {
                   "image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
               }
           }


   class PredictionResponse(BaseModel):
       """Response model for digit prediction."""
       predicted_digit: int = Field(..., ge=0, le=9, description="Predicted digit (0-9)")
       confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
       probabilities: Dict[str, float] = Field(..., description="Probability for each digit")
       processing_time: float = Field(..., description="Processing time in milliseconds")

       class Config:
           json_schema_extra = {
               "example": {
                   "predicted_digit": 7,
                   "confidence": 0.998,
                   "probabilities": {
                       "0": 0.001, "1": 0.001, "2": 0.001, "3": 0.001,
                       "4": 0.001, "5": 0.001, "6": 0.001, "7": 0.998,
                       "8": 0.001, "9": 0.001
                   },
                   "processing_time": 23.5
               }
           }


   def load_model() -> keras.Model:
       """Load the trained model (singleton pattern)."""
       global model_instance

       if model_instance is None:
           print("Loading model...")
           classifier = DigitClassifier(verbose=False)
           model_instance = classifier.model
           print("Model loaded successfully!")

       return model_instance


   def preprocess_canvas_image(image_data: str) -> np.ndarray:
       """
       Preprocess image data from canvas for model input.

       Args:
           image_data: Base64 encoded image data from canvas

       Returns:
           Preprocessed image array ready for model input
       """
       # Remove data URL prefix if present
       if 'base64,' in image_data:
           image_data = image_data.split('base64,')[1]

       # Decode base64 to bytes
       image_bytes = base64.b64decode(image_data)

       # Open image with PIL
       image = Image.open(io.BytesIO(image_bytes))

       # Convert RGBA to RGB with white background
       if image.mode == 'RGBA':
           # Create white background
           background = Image.new('RGB', image.size, (255, 255, 255))
           # Paste image using alpha channel as mask
           background.paste(image, mask=image.split()[3])
           image = background

       # Convert to grayscale
       image = image.convert('L')

       # Resize to 28x28 (MNIST size)
       image = image.resize((28, 28), Image.Resampling.LANCZOS)

       # Convert to numpy array
       img_array = np.array(image, dtype=np.uint8)

       # Invert colors (canvas has black on white, MNIST has white on black)
       img_array = 255 - img_array

       # Add batch dimension
       img_array = img_array.reshape(1, 28, 28)

       # Normalize and reshape for model
       img_normalized = normalize_pixels(img_array)
       img_input = reshape_images(img_normalized, add_channel=True)

       return img_input


   @app.on_event("startup")
   async def startup_event():
       """Load model on startup."""
       try:
           load_model()
       except Exception as e:
           print(f"Warning: Could not preload model: {e}")
           print("Model will be loaded on first request.")


   @app.get("/", response_class=HTMLResponse)
   async def home(request: Request):
       """
       Serve the main page with drawing canvas.
       """
       return templates.TemplateResponse(
           "index.html",
           {
               "request": request,
               "title": "MNIST Digit Classifier"
           }
       )


   @app.post("/predict", response_model=PredictionResponse)
   async def predict(prediction_request: PredictionRequest) -> PredictionResponse:
       """
       Predict digit from canvas image.

       Args:
           prediction_request: Request containing base64 encoded image

       Returns:
           Prediction results with digit, confidence, and probabilities

       Raises:
           HTTPException: If prediction fails
       """
       import time
       start_time = time.time()

       try:
           # Load model if not already loaded
           model = load_model()

           # Preprocess image
           img_input = preprocess_canvas_image(prediction_request.image)

           # Make prediction
           predictions = model.predict(img_input, verbose=0)

           # Get results
           predicted_digit = int(np.argmax(predictions[0]))
           confidence = float(predictions[0][predicted_digit])

           # Create probability dictionary
           probabilities = {
               str(i): float(predictions[0][i])
               for i in range(10)
           }

           # Calculate processing time
           processing_time = (time.time() - start_time) * 1000  # Convert to ms

           return PredictionResponse(
               predicted_digit=predicted_digit,
               confidence=confidence,
               probabilities=probabilities,
               processing_time=processing_time
           )

       except Exception as e:
           raise HTTPException(
               status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
               detail=f"Prediction failed: {str(e)}"
           )


   @app.get("/health")
   async def health_check() -> Dict[str, Any]:
       """
       Health check endpoint.

       Returns:
           Status information about the service
       """
       try:
           model = load_model()
           model_loaded = model is not None
       except:
           model_loaded = False

       return {
           "status": "healthy" if model_loaded else "degraded",
           "timestamp": datetime.now().isoformat(),
           "model_loaded": model_loaded,
           "version": "1.0.0"
       }


   @app.get("/api/info")
   async def api_info() -> Dict[str, Any]:
       """
       Get API information and capabilities.

       Returns:
           Information about the API and model
       """
       try:
           model = load_model()
           model_info = {
               "loaded": True,
               "input_shape": model.input_shape,
               "output_shape": model.output_shape,
               "total_params": model.count_params()
           }
       except:
           model_info = {"loaded": False}

       return {
           "name": "MNIST Digit Classifier API",
           "version": "1.0.0",
           "endpoints": [
               {"path": "/", "method": "GET", "description": "Main web interface"},
               {"path": "/predict", "method": "POST", "description": "Predict digit from image"},
               {"path": "/health", "method": "GET", "description": "Health check"},
               {"path": "/api/info", "method": "GET", "description": "API information"},
               {"path": "/docs", "method": "GET", "description": "Interactive API documentation"}
           ],
           "model": model_info
       }


   # Error handlers
   @app.exception_handler(404)
   async def not_found(request: Request, exc: HTTPException):
       """Handle 404 errors."""
       if request.url.path.startswith("/api/"):
           return JSONResponse(
               status_code=404,
               content={"detail": "Endpoint not found"}
           )
       return templates.TemplateResponse(
           "404.html",
           {"request": request},
           status_code=404
       )


   @app.exception_handler(500)
   async def server_error(request: Request, exc: HTTPException):
       """Handle 500 errors."""
       return JSONResponse(
           status_code=500,
           content={"detail": "Internal server error"}
       )


   if __name__ == "__main__":
       """Run the application directly for development."""
       import uvicorn

       print("Starting MNIST Digit Classifier Web App...")
       print("Open http://localhost:8000 in your browser")

       uvicorn.run(
           "app:app",
           host="0.0.0.0",
           port=8000,
           reload=True,
           log_level="info"
       )
   ```

### Understanding the Code

Let's break down the key components:

**1. FastAPI Setup**:
```python
app = FastAPI(
    title="MNIST Digit Classifier",
    description="Interactive web interface",
    version="1.0.0"
)
```
- Creates the FastAPI application
- Provides metadata for automatic documentation

**2. CORS Middleware**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    ...
)
```
- Enables cross-origin requests
- Important for development and API access

**3. Pydantic Models**:
```python
class PredictionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
```
- Type validation for requests
- Automatic documentation generation
- Clear API contracts

**4. Image Preprocessing**:
```python
# Canvas has black on white, MNIST has white on black
img_array = 255 - img_array
```
- Handles the inversion needed for canvas drawings
- Consistent with MNIST training data

**5. Model Loading**:
```python
@app.on_event("startup")
async def startup_event():
    load_model()  # Preload for faster first prediction
```
- Singleton pattern for model instance
- Loads once, reuses for all requests

### API Design Best Practices

1. **RESTful Endpoints**:
   - GET `/` - Main interface
   - POST `/predict` - Prediction endpoint
   - GET `/health` - Health monitoring

2. **Error Handling**:
   - Proper HTTP status codes
   - Descriptive error messages
   - Graceful degradation

3. **Documentation**:
   - Automatic with FastAPI
   - Access at `/docs` (Swagger UI)
   - Access at `/redoc` (ReDoc)

---

## Task 5.2: Create HTML Frontend

### Understanding the Frontend Architecture

Our frontend will consist of:
1. **HTML**: Structure and content
2. **CSS**: Styling and layout
3. **JavaScript**: Interactivity and API communication

The main components:
- Drawing canvas for digit input
- Prediction display area
- Control buttons (predict, clear)
- Probability visualization

### Step-by-Step Instructions

1. **Create the main HTML template**:
   ```bash
   touch templates/index.html
   ```

2. **Implement the HTML structure**:

   Add this code to `templates/index.html`:

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>{{ title }} - Draw a Digit</title>

       <!-- CSS -->
       <link rel="stylesheet" href="{{ url_for('static', path='/css/style.css') }}">

       <!-- Favicon -->
       <link rel="icon" type="image/png" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==">
   </head>
   <body>
       <div class="container">
           <!-- Header -->
           <header>
               <h1>MNIST Digit Classifier</h1>
               <p class="subtitle">Draw a digit (0-9) and let AI recognize it!</p>
           </header>

           <!-- Main Content -->
           <main>
               <div class="app-grid">
                   <!-- Drawing Area -->
                   <div class="drawing-section">
                       <div class="canvas-container">
                           <canvas id="drawingCanvas" width="280" height="280"></canvas>
                           <div class="canvas-label">Draw here</div>
                       </div>

                       <div class="controls">
                           <button id="predictBtn" class="btn btn-primary">
                               <span class="btn-icon">🔮</span>
                               Predict
                           </button>
                           <button id="clearBtn" class="btn btn-secondary">
                               <span class="btn-icon">🗑️</span>
                               Clear
                           </button>
                       </div>
                   </div>

                   <!-- Results Area -->
                   <div class="results-section">
                       <div id="resultContainer" class="result-container">
                           <div class="placeholder">
                               <div class="placeholder-icon">✏️</div>
                               <p>Draw a digit and click "Predict"</p>
                           </div>
                       </div>

                       <div id="probabilityChart" class="probability-chart" style="display: none;">
                           <h3>Confidence Levels</h3>
                           <div class="chart-container">
                               <!-- Probability bars will be inserted here -->
                           </div>
                       </div>
                   </div>
               </div>

               <!-- Additional Info -->
               <div class="info-section">
                   <div class="info-card">
                       <h3>How it works</h3>
                       <ol>
                           <li>Draw a single digit (0-9) in the canvas</li>
                           <li>Click "Predict" to analyze your drawing</li>
                           <li>The AI will recognize the digit with confidence scores</li>
                           <li>Click "Clear" to try another digit</li>
                       </ol>
                   </div>

                   <div class="info-card">
                       <h3>Tips for best results</h3>
                       <ul>
                           <li>Draw the digit large and centered</li>
                           <li>Use smooth, continuous strokes</li>
                           <li>Make the digit fill most of the canvas</li>
                           <li>Avoid very thin or very thick lines</li>
                       </ul>
                   </div>

                   <div class="info-card">
                       <h3>About the model</h3>
                       <p>This classifier uses a Convolutional Neural Network (CNN) trained on the MNIST dataset with over 98% accuracy.</p>
                       <p><a href="/docs" target="_blank">View API Documentation →</a></p>
                   </div>
               </div>
           </main>

           <!-- Footer -->
           <footer>
               <p>Built with TensorFlow, FastAPI, and ❤️</p>
               <p class="footer-links">
                   <a href="https://github.com/yourusername/mnist-classifier" target="_blank">GitHub</a>
                   <span class="separator">•</span>
                   <a href="/docs" target="_blank">API Docs</a>
                   <span class="separator">•</span>
                   <a href="/health" target="_blank">Status</a>
               </p>
           </footer>
       </div>

       <!-- Loading Overlay -->
       <div id="loadingOverlay" class="loading-overlay" style="display: none;">
           <div class="spinner"></div>
           <p>Analyzing your digit...</p>
       </div>

       <!-- Error Modal -->
       <div id="errorModal" class="modal" style="display: none;">
           <div class="modal-content">
               <span class="close">&times;</span>
               <h2>Oops!</h2>
               <p id="errorMessage">Something went wrong. Please try again.</p>
               <button class="btn btn-primary" onclick="closeErrorModal()">OK</button>
           </div>
       </div>

       <!-- JavaScript -->
       <script src="{{ url_for('static', path='/js/main.js') }}"></script>
   </body>
   </html>
   ```

3. **Create the 404 error page**:
   ```bash
   touch templates/404.html
   ```

   Add this simple 404 page:

   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>404 - Page Not Found</title>
       <link rel="stylesheet" href="{{ url_for('static', path='/css/style.css') }}">
   </head>
   <body>
       <div class="container">
           <div class="error-page">
               <h1>404</h1>
               <p>Page not found</p>
               <a href="/" class="btn btn-primary">Go Home</a>
           </div>
       </div>
   </body>
   </html>
   ```

### Understanding the HTML Structure

**1. Responsive Design**:
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```
- Ensures proper scaling on mobile devices
- Essential for touch-based drawing

**2. Canvas Element**:
```html
<canvas id="drawingCanvas" width="280" height="280"></canvas>
```
- 280x280 provides good drawing space
- 10x the MNIST size for better user experience

**3. Semantic Structure**:
- `<header>`, `<main>`, `<footer>` for accessibility
- Clear sections for different functionalities
- Proper heading hierarchy

**4. Interactive Elements**:
- Buttons with clear labels and icons
- Loading overlay for feedback
- Error modal for handling failures

---

## Task 5.3: Create JavaScript Logic

### Understanding the JavaScript Requirements

Our JavaScript needs to:
1. Handle canvas drawing with mouse and touch
2. Capture canvas data as image
3. Send predictions to the API
4. Display results dynamically
5. Handle errors gracefully

### Step-by-Step Instructions

1. **Create the CSS file first**:
   ```bash
   touch static/css/style.css
   ```

   Add this CSS to `static/css/style.css`:

   ```css
   /* Reset and Base Styles */
   * {
       margin: 0;
       padding: 0;
       box-sizing: border-box;
   }

   body {
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
       background-color: #f5f5f5;
       color: #333;
       line-height: 1.6;
   }

   /* Container */
   .container {
       max-width: 1200px;
       margin: 0 auto;
       padding: 20px;
   }

   /* Header */
   header {
       text-align: center;
       margin-bottom: 40px;
   }

   h1 {
       font-size: 2.5em;
       color: #2c3e50;
       margin-bottom: 10px;
   }

   .subtitle {
       font-size: 1.2em;
       color: #7f8c8d;
   }

   /* Main Grid */
   .app-grid {
       display: grid;
       grid-template-columns: 1fr 1fr;
       gap: 40px;
       margin-bottom: 40px;
   }

   @media (max-width: 768px) {
       .app-grid {
           grid-template-columns: 1fr;
           gap: 30px;
       }
   }

   /* Drawing Section */
   .drawing-section {
       background: white;
       padding: 30px;
       border-radius: 10px;
       box-shadow: 0 2px 10px rgba(0,0,0,0.1);
   }

   .canvas-container {
       position: relative;
       display: inline-block;
       margin-bottom: 20px;
   }

   #drawingCanvas {
       border: 2px solid #ddd;
       border-radius: 8px;
       cursor: crosshair;
       background: white;
       display: block;
       margin: 0 auto;
       touch-action: none;
   }

   .canvas-label {
       position: absolute;
       top: 10px;
       left: 10px;
       color: #bbb;
       font-size: 14px;
       pointer-events: none;
       transition: opacity 0.3s;
   }

   /* Controls */
   .controls {
       display: flex;
       gap: 10px;
       justify-content: center;
   }

   .btn {
       padding: 12px 24px;
       font-size: 16px;
       border: none;
       border-radius: 5px;
       cursor: pointer;
       display: inline-flex;
       align-items: center;
       gap: 8px;
       transition: all 0.3s ease;
       font-weight: 500;
   }

   .btn:hover {
       transform: translateY(-2px);
       box-shadow: 0 4px 12px rgba(0,0,0,0.15);
   }

   .btn:active {
       transform: translateY(0);
   }

   .btn-primary {
       background-color: #3498db;
       color: white;
   }

   .btn-primary:hover {
       background-color: #2980b9;
   }

   .btn-secondary {
       background-color: #95a5a6;
       color: white;
   }

   .btn-secondary:hover {
       background-color: #7f8c8d;
   }

   .btn-icon {
       font-size: 20px;
   }

   /* Results Section */
   .results-section {
       background: white;
       padding: 30px;
       border-radius: 10px;
       box-shadow: 0 2px 10px rgba(0,0,0,0.1);
   }

   .result-container {
       min-height: 200px;
       display: flex;
       align-items: center;
       justify-content: center;
       margin-bottom: 30px;
   }

   .placeholder {
       text-align: center;
       color: #bbb;
   }

   .placeholder-icon {
       font-size: 48px;
       margin-bottom: 10px;
   }

   .prediction-result {
       text-align: center;
       animation: fadeIn 0.5s ease;
   }

   @keyframes fadeIn {
       from { opacity: 0; transform: translateY(10px); }
       to { opacity: 1; transform: translateY(0); }
   }

   .predicted-digit {
       font-size: 120px;
       font-weight: bold;
       color: #3498db;
       line-height: 1;
       margin-bottom: 10px;
   }

   .confidence-score {
       font-size: 24px;
       color: #27ae60;
       margin-bottom: 5px;
   }

   .processing-time {
       font-size: 14px;
       color: #95a5a6;
   }

   /* Probability Chart */
   .probability-chart h3 {
       margin-bottom: 15px;
       color: #2c3e50;
   }

   .chart-container {
       display: grid;
       gap: 8px;
   }

   .prob-bar {
       display: grid;
       grid-template-columns: 30px 1fr 60px;
       align-items: center;
       gap: 10px;
   }

   .prob-label {
       font-weight: 500;
       text-align: center;
   }

   .prob-bar-container {
       background: #ecf0f1;
       height: 20px;
       border-radius: 10px;
       overflow: hidden;
       position: relative;
   }

   .prob-bar-fill {
       height: 100%;
       background: #3498db;
       transition: width 0.5s ease;
       border-radius: 10px;
   }

   .prob-bar-fill.high {
       background: #27ae60;
   }

   .prob-value {
       text-align: right;
       font-size: 14px;
       color: #7f8c8d;
   }

   /* Info Section */
   .info-section {
       display: grid;
       grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
       gap: 20px;
       margin-bottom: 40px;
   }

   .info-card {
       background: white;
       padding: 25px;
       border-radius: 10px;
       box-shadow: 0 2px 10px rgba(0,0,0,0.1);
   }

   .info-card h3 {
       color: #2c3e50;
       margin-bottom: 15px;
   }

   .info-card ol,
   .info-card ul {
       padding-left: 20px;
   }

   .info-card li {
       margin-bottom: 8px;
   }

   .info-card a {
       color: #3498db;
       text-decoration: none;
   }

   .info-card a:hover {
       text-decoration: underline;
   }

   /* Footer */
   footer {
       text-align: center;
       padding: 30px 0;
       color: #7f8c8d;
   }

   .footer-links {
       margin-top: 10px;
   }

   .footer-links a {
       color: #3498db;
       text-decoration: none;
   }

   .footer-links a:hover {
       text-decoration: underline;
   }

   .separator {
       margin: 0 10px;
       color: #bdc3c7;
   }

   /* Loading Overlay */
   .loading-overlay {
       position: fixed;
       top: 0;
       left: 0;
       right: 0;
       bottom: 0;
       background: rgba(0, 0, 0, 0.7);
       display: flex;
       flex-direction: column;
       align-items: center;
       justify-content: center;
       z-index: 1000;
   }

   .spinner {
       width: 50px;
       height: 50px;
       border: 5px solid #f3f3f3;
       border-top: 5px solid #3498db;
       border-radius: 50%;
       animation: spin 1s linear infinite;
       margin-bottom: 20px;
   }

   @keyframes spin {
       0% { transform: rotate(0deg); }
       100% { transform: rotate(360deg); }
   }

   .loading-overlay p {
       color: white;
       font-size: 18px;
   }

   /* Modal */
   .modal {
       position: fixed;
       z-index: 1001;
       left: 0;
       top: 0;
       width: 100%;
       height: 100%;
       background-color: rgba(0,0,0,0.5);
       display: flex;
       align-items: center;
       justify-content: center;
   }

   .modal-content {
       background-color: white;
       padding: 30px;
       border-radius: 10px;
       max-width: 400px;
       width: 90%;
       text-align: center;
       position: relative;
   }

   .close {
       position: absolute;
       right: 15px;
       top: 15px;
       font-size: 28px;
       font-weight: bold;
       color: #aaa;
       cursor: pointer;
   }

   .close:hover {
       color: #000;
   }

   .modal h2 {
       margin-bottom: 15px;
       color: #e74c3c;
   }

   .modal p {
       margin-bottom: 20px;
       color: #555;
   }

   /* Error Page */
   .error-page {
       text-align: center;
       padding: 100px 20px;
   }

   .error-page h1 {
       font-size: 120px;
       color: #e74c3c;
       margin-bottom: 20px;
   }

   .error-page p {
       font-size: 24px;
       color: #7f8c8d;
       margin-bottom: 30px;
   }
   ```

2. **Create the JavaScript file**:
   ```bash
   touch static/js/main.js
   ```

   Add this JavaScript to `static/js/main.js`:

   ```javascript
   /**
    * MNIST Digit Classifier - Main JavaScript
    *
    * This script handles:
    * - Canvas drawing functionality
    * - API communication
    * - Result display
    * - User interactions
    */

   // Global variables
   let canvas;
   let ctx;
   let isDrawing = false;
   let lastX = 0;
   let lastY = 0;

   // Initialize when DOM is loaded
   document.addEventListener('DOMContentLoaded', function() {
       initializeCanvas();
       setupEventListeners();
   });

   /**
    * Initialize the drawing canvas
    */
   function initializeCanvas() {
       canvas = document.getElementById('drawingCanvas');
       ctx = canvas.getContext('2d');

       // Set drawing style
       ctx.strokeStyle = '#000000';
       ctx.lineWidth = 15;
       ctx.lineCap = 'round';
       ctx.lineJoin = 'round';

       // Fill with white background
       ctx.fillStyle = '#FFFFFF';
       ctx.fillRect(0, 0, canvas.width, canvas.height);
   }

   /**
    * Setup all event listeners
    */
   function setupEventListeners() {
       // Canvas events
       canvas.addEventListener('mousedown', startDrawing);
       canvas.addEventListener('mousemove', draw);
       canvas.addEventListener('mouseup', stopDrawing);
       canvas.addEventListener('mouseout', stopDrawing);

       // Touch events for mobile
       canvas.addEventListener('touchstart', handleTouch);
       canvas.addEventListener('touchmove', handleTouch);
       canvas.addEventListener('touchend', stopDrawing);

       // Button events
       document.getElementById('predictBtn').addEventListener('click', predict);
       document.getElementById('clearBtn').addEventListener('click', clearCanvas);

       // Prevent scrolling when touching canvas
       document.body.addEventListener('touchstart', function(e) {
           if (e.target === canvas) {
               e.preventDefault();
           }
       }, { passive: false });

       document.body.addEventListener('touchend', function(e) {
           if (e.target === canvas) {
               e.preventDefault();
           }
       }, { passive: false });

       document.body.addEventListener('touchmove', function(e) {
           if (e.target === canvas) {
               e.preventDefault();
           }
       }, { passive: false });
   }

   /**
    * Start drawing on canvas
    */
   function startDrawing(e) {
       isDrawing = true;
       [lastX, lastY] = getMousePos(e);

       // Hide the "Draw here" label
       document.querySelector('.canvas-label').style.opacity = '0';
   }

   /**
    * Draw on canvas
    */
   function draw(e) {
       if (!isDrawing) return;

       const [currentX, currentY] = getMousePos(e);

       ctx.beginPath();
       ctx.moveTo(lastX, lastY);
       ctx.lineTo(currentX, currentY);
       ctx.stroke();

       [lastX, lastY] = [currentX, currentY];
   }

   /**
    * Stop drawing
    */
   function stopDrawing() {
       isDrawing = false;
   }

   /**
    * Handle touch events
    */
   function handleTouch(e) {
       e.preventDefault();
       const touch = e.touches[0];
       const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' :
                                        e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
           clientX: touch.clientX,
           clientY: touch.clientY
       });
       canvas.dispatchEvent(mouseEvent);
   }

   /**
    * Get mouse position relative to canvas
    */
   function getMousePos(e) {
       const rect = canvas.getBoundingClientRect();
       return [
           e.clientX - rect.left,
           e.clientY - rect.top
       ];
   }

   /**
    * Clear the canvas
    */
   function clearCanvas() {
       ctx.clearRect(0, 0, canvas.width, canvas.height);
       ctx.fillStyle = '#FFFFFF';
       ctx.fillRect(0, 0, canvas.width, canvas.height);

       // Show the "Draw here" label again
       document.querySelector('.canvas-label').style.opacity = '1';

       // Clear results
       clearResults();
   }

   /**
    * Clear prediction results
    */
   function clearResults() {
       document.getElementById('resultContainer').innerHTML = `
           <div class="placeholder">
               <div class="placeholder-icon">✏️</div>
               <p>Draw a digit and click "Predict"</p>
           </div>
       `;
       document.getElementById('probabilityChart').style.display = 'none';
   }

   /**
    * Make prediction
    */
   async function predict() {
       // Check if canvas is empty
       if (isCanvasEmpty()) {
           showError('Please draw a digit first!');
           return;
       }

       // Show loading overlay
       showLoading();

       try {
           // Get canvas data as base64
           const imageData = canvas.toDataURL('image/png');

           // Send to API
           const response = await fetch('/predict', {
               method: 'POST',
               headers: {
                   'Content-Type': 'application/json',
               },
               body: JSON.stringify({
                   image: imageData
               })
           });

           if (!response.ok) {
               throw new Error(`HTTP error! status: ${response.status}`);
           }

           const result = await response.json();

           // Hide loading
           hideLoading();

           // Display results
           displayResults(result);

       } catch (error) {
           hideLoading();
           showError('Failed to make prediction. Please try again.');
           console.error('Prediction error:', error);
       }
   }

   /**
    * Check if canvas is empty
    */
   function isCanvasEmpty() {
       const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
       const data = imageData.data;

       // Check if all pixels are white
       for (let i = 0; i < data.length; i += 4) {
           // Check RGB values (ignore alpha)
           if (data[i] !== 255 || data[i + 1] !== 255 || data[i + 2] !== 255) {
               return false;
           }
       }
       return true;
   }

   /**
    * Display prediction results
    */
   function displayResults(result) {
       // Display main prediction
       document.getElementById('resultContainer').innerHTML = `
           <div class="prediction-result">
               <div class="predicted-digit">${result.predicted_digit}</div>
               <div class="confidence-score">${(result.confidence * 100).toFixed(1)}% confident</div>
               <div class="processing-time">Processed in ${result.processing_time.toFixed(1)}ms</div>
           </div>
       `;

       // Display probability chart
       displayProbabilityChart(result.probabilities);
   }

   /**
    * Display probability chart
    */
   function displayProbabilityChart(probabilities) {
       const chartContainer = document.querySelector('.chart-container');
       chartContainer.innerHTML = '';

       // Sort probabilities by value
       const sortedProbs = Object.entries(probabilities)
           .sort((a, b) => b[1] - a[1]);

       // Create bars for each digit
       sortedProbs.forEach(([digit, probability]) => {
           const percentage = (probability * 100).toFixed(1);
           const isHigh = probability > 0.5;

           const probBar = document.createElement('div');
           probBar.className = 'prob-bar';
           probBar.innerHTML = `
               <div class="prob-label">${digit}</div>
               <div class="prob-bar-container">
                   <div class="prob-bar-fill ${isHigh ? 'high' : ''}" style="width: ${percentage}%"></div>
               </div>
               <div class="prob-value">${percentage}%</div>
           `;

           chartContainer.appendChild(probBar);
       });

       // Show the chart
       document.getElementById('probabilityChart').style.display = 'block';
   }

   /**
    * Show loading overlay
    */
   function showLoading() {
       document.getElementById('loadingOverlay').style.display = 'flex';
   }

   /**
    * Hide loading overlay
    */
   function hideLoading() {
       document.getElementById('loadingOverlay').style.display = 'none';
   }

   /**
    * Show error message
    */
   function showError(message) {
       document.getElementById('errorMessage').textContent = message;
       document.getElementById('errorModal').style.display = 'flex';
   }

   /**
    * Close error modal
    */
   function closeErrorModal() {
       document.getElementById('errorModal').style.display = 'none';
   }

   // Close modal when clicking outside
   window.onclick = function(event) {
       const modal = document.getElementById('errorModal');
       if (event.target === modal) {
           closeErrorModal();
       }
   }

   // Close modal with close button
   document.addEventListener('DOMContentLoaded', function() {
       const closeBtn = document.querySelector('.close');
       if (closeBtn) {
           closeBtn.onclick = closeErrorModal;
       }
   });
   ```

### Understanding the JavaScript

Key components explained:

**1. Canvas Drawing**:
```javascript
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
```
- Thick lines simulate pen/marker
- Round caps and joins for smooth strokes

**2. Touch Support**:
```javascript
canvas.addEventListener('touchstart', handleTouch);
// Convert touch to mouse events
```
- Essential for mobile devices
- Prevents scrolling while drawing

**3. Canvas to Base64**:
```javascript
const imageData = canvas.toDataURL('image/png');
```
- Converts canvas content to base64 string
- Can be sent as JSON to API

**4. Async/Await for API**:
```javascript
const response = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({image: imageData})
});
```
- Modern approach to handling promises
- Clean error handling with try/catch

---

## Task 5.4: Mount Static Files

This task is already completed in the FastAPI application code, but let's understand how it works:

```python
# In app.py
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
```

This tells FastAPI to:
1. Serve files from the `static/` directory
2. Make them available at `/static/*` URLs
3. Use the name "static" for URL generation

### Directory Structure

Ensure your directory structure looks like this:
```
mnist_classifier/
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   ├── index.html
│   └── 404.html
└── src/
    └── mnist_classifier/
        └── api/
            └── app.py
```

---

## Helper Scripts

### run_web_app.py - Web App Launcher

Create this convenience script in the project root:

```python
#!/usr/bin/env python3
"""
Launch the MNIST Digit Classifier web application.

This script starts the FastAPI server with appropriate settings
for development or production.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for web app launcher."""
    parser = argparse.ArgumentParser(
        description="Launch MNIST Digit Classifier Web App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run in development mode
  %(prog)s --port 8080        # Run on different port
  %(prog)s --production       # Run in production mode
  %(prog)s --host 0.0.0.0     # Allow external connections

After starting, open http://localhost:8000 in your browser.
        """
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )

    parser.add_argument(
        '--production',
        action='store_true',
        help='Run in production mode (no auto-reload)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (production only)'
    )

    args = parser.parse_args()

    # Import uvicorn
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed")
        print("Install with: pip install uvicorn[standard]")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════╗
║     MNIST Digit Classifier Web App           ║
╠══════════════════════════════════════════════╣
║  Host: {args.host:<37} ║
║  Port: {args.port:<37} ║
║  Mode: {'Production' if args.production else 'Development':<37} ║
╚══════════════════════════════════════════════╝
    """)

    # Configure uvicorn
    config = {
        "app": "src.mnist_classifier.api.app:app",
        "host": args.host,
        "port": args.port,
        "log_level": "info",
    }

    if args.production:
        config.update({
            "workers": args.workers,
            "reload": False,
        })
        print("\n🚀 Starting in PRODUCTION mode...")
    else:
        config.update({
            "reload": True,
            "reload_dirs": ["src", "templates", "static"],
        })
        print("\n🔧 Starting in DEVELOPMENT mode (auto-reload enabled)...")

    print(f"\n📍 Access the app at: http://{args.host}:{args.port}")
    print("📍 API documentation at: http://{args.host}:{args.port}/docs")
    print("\nPress CTRL+C to stop the server\n")

    # Run the server
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### test_web_api.py - API Testing Script

Create this script to test the API:

```python
#!/usr/bin/env python3
"""
Test script for the MNIST Web API.

This script tests various API endpoints and functionality.
"""

import requests
import base64
import json
import time
from PIL import Image, ImageDraw
import io
import sys


def create_test_digit_image(digit=7):
    """
    Create a test image with a digit.

    Returns:
        Base64 encoded image string
    """
    # Create image
    img = Image.new('RGBA', (280, 280), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw a simple digit (7)
    if digit == 7:
        draw.line([(70, 70), (210, 70)], fill=(0, 0, 0), width=20)
        draw.line([(210, 70), (140, 210)], fill=(0, 0, 0), width=20)
    elif digit == 1:
        draw.line([(140, 70), (140, 210)], fill=(0, 0, 0), width=20)

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def test_health_endpoint(base_url):
    """Test the health check endpoint."""
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_api_info_endpoint(base_url):
    """Test the API info endpoint."""
    print("\n2. Testing /api/info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/info")
        print(f"   Status Code: {response.status_code}")
        data = response.json()
        print(f"   API Name: {data.get('name')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Model Loaded: {data.get('model', {}).get('loaded')}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_prediction_endpoint(base_url):
    """Test the prediction endpoint."""
    print("\n3. Testing /predict endpoint...")

    # Create test image
    print("   Creating test image...")
    test_image = create_test_digit_image(7)

    # Send prediction request
    print("   Sending prediction request...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json={"image": test_image},
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = (time.time() - start_time) * 1000

        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed_time:.1f}ms")

        if response.status_code == 200:
            data = response.json()
            print(f"   Predicted Digit: {data['predicted_digit']}")
            print(f"   Confidence: {data['confidence']:.1%}")
            print(f"   Processing Time: {data['processing_time']:.1f}ms")

            # Show top 3 probabilities
            probs = data['probabilities']
            sorted_probs = sorted(probs.items(), key=lambda x: float(x[1]), reverse=True)[:3]
            print("   Top 3 predictions:")
            for digit, prob in sorted_probs:
                print(f"     Digit {digit}: {float(prob):.1%}")

            return True
        else:
            print(f"   ❌ Error Response: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_main_page(base_url):
    """Test the main web page."""
    print("\n4. Testing main page...")
    try:
        response = requests.get(base_url)
        print(f"   Status Code: {response.status_code}")
        print(f"   Content Type: {response.headers.get('content-type')}")
        print(f"   Page Title Found: {'MNIST Digit Classifier' in response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_static_files(base_url):
    """Test static file serving."""
    print("\n5. Testing static files...")

    static_files = [
        "/static/css/style.css",
        "/static/js/main.js"
    ]

    all_good = True
    for file_path in static_files:
        try:
            response = requests.get(f"{base_url}{file_path}")
            status = "✅" if response.status_code == 200 else "❌"
            print(f"   {status} {file_path} - Status: {response.status_code}")
            if response.status_code != 200:
                all_good = False
        except Exception as e:
            print(f"   ❌ {file_path} - Error: {e}")
            all_good = False

    return all_good


def main():
    """Run all API tests."""
    # Default API URL
    base_url = "http://localhost:8000"

    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')

    print(f"🧪 Testing MNIST Web API")
    print(f"📍 Base URL: {base_url}")
    print("=" * 50)

    # Check if server is running
    try:
        requests.get(base_url, timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to server")
        print(f"   Make sure the server is running at {base_url}")
        print("   Start with: python run_web_app.py")
        sys.exit(1)

    # Run tests
    tests = [
        test_health_endpoint(base_url),
        test_api_info_endpoint(base_url),
        test_prediction_endpoint(base_url),
        test_main_page(base_url),
        test_static_files(base_url)
    ]

    # Summary
    passed = sum(tests)
    total = len(tests)

    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed! The web API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### verify_milestone5.py - Comprehensive Verification

```python
#!/usr/bin/env python3
"""
Comprehensive verification script for Milestone 5.
Ensures the web interface is properly implemented and functional.
"""

import os
import sys
import subprocess
import importlib.util
import time
import requests
import json

def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"

def check_api_module():
    """Check if the API module exists and can be imported."""
    print("\n📦 Checking API module:")

    # Check if file exists
    api_path = os.path.join("src", "mnist_classifier", "api", "app.py")
    file_exists = os.path.exists(api_path)
    print(f"  {check_mark(file_exists)} app.py exists at {api_path}")

    # Try to import the module
    can_import = False
    has_app = False
    if file_exists:
        try:
            # Add src to Python path
            sys.path.insert(0, "src")
            from mnist_classifier.api import app as api_module
            can_import = True

            # Check if FastAPI app exists
            has_app = hasattr(api_module, 'app')
        except Exception as e:
            print(f"  ❌ Import error: {e}")

    print(f"  {check_mark(can_import)} API module can be imported")
    print(f"  {check_mark(has_app)} FastAPI app instance exists")

    return file_exists and can_import and has_app

def check_templates():
    """Check if HTML templates exist."""
    print("\n📄 Checking templates:")

    templates = [
        ("templates/index.html", "Main page template"),
    ]

    all_exist = True
    for path, description in templates:
        exists = os.path.exists(path)
        print(f"  {check_mark(exists)} {description}: {path}")
        if not exists:
            all_exist = False

    return all_exist

def check_static_files():
    """Check if static files exist."""
    print("\n🎨 Checking static files:")

    static_files = [
        ("static/css/style.css", "CSS styles"),
        ("static/js/main.js", "JavaScript code"),
    ]

    all_exist = True
    for path, description in static_files:
        exists = os.path.exists(path)
        print(f"  {check_mark(exists)} {description}: {path}")
        if not exists:
            all_exist = False

    return all_exist

def check_model_exists():
    """Check if a trained model exists."""
    print("\n🧠 Checking for trained model:")

    model_paths = [
        "models/mnist_model",
        "models/mnist_cnn_model",
    ]

    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"  ✅ Found model at: {path}")
            model_found = True
            break

    if not model_found:
        print("  ❌ No trained model found")
        print("  💡 Run training first: python train_model.py")

    return model_found

def test_server_startup():
    """Test if the server can start."""
    print("\n🚀 Testing server startup:")

    try:
        # Start the server in a subprocess
        cmd = [sys.executable, "-m", "uvicorn", "src.mnist_classifier.api.app:app",
               "--host", "0.0.0.0", "--port", "8000"]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give the server time to start
        print("  ⏳ Starting server...")
        time.sleep(3)

        # Check if process is still running
        if process.poll() is None:
            print("  ✅ Server started successfully")

            # Try to access the root endpoint
            try:
                response = requests.get("http://localhost:8000/", timeout=5)
                print(f"  ✅ Root endpoint accessible (status: {response.status_code})")

                # Check API docs
                docs_response = requests.get("http://localhost:8000/docs", timeout=5)
                print(f"  ✅ API docs accessible (status: {docs_response.status_code})")

                server_ok = True
            except requests.RequestException as e:
                print(f"  ❌ Could not access endpoints: {e}")
                server_ok = False

            # Stop the server
            process.terminate()
            process.wait(timeout=5)
            print("  ✅ Server stopped cleanly")

            return server_ok
        else:
            # Process died
            stdout, stderr = process.communicate()
            print("  ❌ Server failed to start")
            if stderr:
                print(f"  Error: {stderr.strip()}")
            return False

    except Exception as e:
        print(f"  ❌ Error testing server: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints if server is running."""
    print("\n🔌 Testing API endpoints:")

    # Check if server is already running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        server_running = response.status_code == 200
    except:
        server_running = False

    if not server_running:
        print("  ℹ️  Server not running. Start with: uvicorn src.mnist_classifier.api.app:app --reload")
        return False

    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health")
        health_ok = response.status_code == 200
        print(f"  {check_mark(health_ok)} Health check endpoint")
    except:
        health_ok = False
        print("  ❌ Health check endpoint failed")

    # Test predict endpoint with dummy data
    try:
        # Create a simple test image (28x28 white square)
        import base64
        import numpy as np
        from PIL import Image
        import io

        # Create white image
        img_array = np.ones((28, 28), dtype=np.uint8) * 255
        img = Image.fromarray(img_array)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Send prediction request
        response = requests.post(
            "http://localhost:8000/predict",
            json={"image": img_base64}
        )

        predict_ok = response.status_code == 200
        print(f"  {check_mark(predict_ok)} Prediction endpoint")

        if predict_ok:
            result = response.json()
            print(f"    Predicted digit: {result.get('digit', 'N/A')}")
            print(f"    Confidence: {result.get('confidence', 0):.2%}")
    except Exception as e:
        predict_ok = False
        print(f"  ❌ Prediction endpoint failed: {e}")

    return health_ok and predict_ok

def check_frontend_functionality():
    """Check if frontend files have required functionality."""
    print("\n🎯 Checking frontend functionality:")

    # Check JavaScript for required functions
    js_path = "static/js/main.js"
    if os.path.exists(js_path):
        with open(js_path, 'r') as f:
            js_content = f.read()

        required_functions = [
            ("clearCanvas", "Canvas clearing"),
            ("predictDigit", "Prediction submission"),
            ("drawing logic", "Mouse/touch drawing"),
        ]

        for func, desc in required_functions:
            has_func = func in js_content
            print(f"  {check_mark(has_func)} {desc}")
    else:
        print("  ❌ JavaScript file not found")
        return False

    # Check HTML for required elements
    html_path = "templates/index.html"
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html_content = f.read()

        required_elements = [
            ("<canvas", "Drawing canvas"),
            ("Clear", "Clear button"),
            ("Predict", "Predict button"),
        ]

        all_found = True
        for element, desc in required_elements:
            has_element = element in html_content
            print(f"  {check_mark(has_element)} {desc}")
            if not has_element:
                all_found = False

        return all_found
    else:
        print("  ❌ HTML template not found")
        return False

def main():
    """Run all verification checks."""
    print("🔍 MNIST Classifier - Milestone 5 Verification")
    print("=" * 50)

    # Check if in correct directory
    if not os.path.exists("src/mnist_classifier"):
        print("❌ Error: Not in project root directory")
        print("Please run from the mnist_classifier directory")
        sys.exit(1)

    # Run all checks
    checks = [
        ("API Module", check_api_module()),
        ("Templates", check_templates()),
        ("Static Files", check_static_files()),
        ("Trained Model", check_model_exists()),
        ("Frontend Functionality", check_frontend_functionality()),
        ("Server Startup", test_server_startup()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 5 is complete!")
        print("\n🚀 To run your web application:")
        print("  1. Start the server: uvicorn src.mnist_classifier.api.app:app --reload")
        print("  2. Open browser: http://localhost:8000")
        print("  3. Draw a digit and click Predict!")
        print("\nYou're ready for Milestone 6: Visualizing Neural Network State")
    else:
        print("\n⚠️  Some checks failed. Please complete all tasks before proceeding.")
        print("\n💡 Tips:")
        if not checks[0][1]:  # API module
            print("  - Ensure app.py is in src/mnist_classifier/api/")
            print("  - Check for import errors in your code")
        if not checks[3][1]:  # Model
            print("  - Train a model first: python train_model.py")
        if not checks[5][1]:  # Server
            print("  - Check for syntax errors in app.py")
            print("  - Ensure all dependencies are installed")

if __name__ == "__main__":
    main()
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### FastAPI Issues

**Problem**: "ModuleNotFoundError: No module named 'fastapi'"
- **Solution**: Install FastAPI: `uv pip install fastapi uvicorn[standard]`
- **Verify**: `uv pip list | grep fastapi`

**Problem**: "ImportError: cannot import name 'Jinja2Templates'"
- **Solution**: Install Jinja2: `uv pip install jinja2`
- **Note**: Usually included with FastAPI installation

**Problem**: Server won't start - "Address already in use"
- **Solution**: Another process is using port 8000
- **Fix**: Kill the process or use a different port:
  ```bash
  # Find process using port 8000
  lsof -i :8000  # Mac/Linux
  netstat -ano | findstr :8000  # Windows

  # Use different port
  uvicorn src.mnist_classifier.api.app:app --port 8001
  ```

#### Model Loading Issues

**Problem**: "No model found at path"
- **Solution**: Ensure you've trained a model (Milestone 3)
- **Check**: Look for model in `models/` directory
- **Fix**: Run training: `python train_model.py`

**Problem**: "SavedModel format not supported"
- **Solution**: Model might be in wrong format
- **Fix**: Retrain with proper export:
  ```python
  model.export("models/mnist_model")
  ```

#### Frontend Issues

**Problem**: Canvas doesn't appear
- **Check**: Browser console for JavaScript errors (F12)
- **Common causes**:
  - Incorrect canvas ID in JavaScript
  - Missing canvas element in HTML
  - CSS making canvas invisible

**Problem**: Drawing doesn't work
- **Check**: Mouse event listeners are attached
- **Fix**: Ensure JavaScript loads after DOM:
  ```html
  <script src="/static/js/main.js" defer></script>
  ```

**Problem**: Predictions always return same digit
- **Causes**:
  - Model not loaded properly
  - Image preprocessing incorrect
  - Canvas data not being sent correctly
- **Debug**: Add logging to see image data:
  ```python
  print(f"Image shape: {image_array.shape}")
  print(f"Image range: [{image_array.min()}, {image_array.max()}]")
  ```

#### CORS Issues

**Problem**: "CORS policy" errors in browser
- **Solution**: Ensure CORS middleware is configured
- **Fix in app.py**:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

#### Static Files Issues

**Problem**: CSS/JS files return 404
- **Check**: Static files mount in app.py
- **Fix**: Ensure correct path:
  ```python
  app.mount("/static", StaticFiles(directory="static"), name="static")
  ```
- **Verify**: File paths are relative to project root

### Performance Optimization

1. **Model Loading**: Load model once at startup, not per request
2. **Image Processing**: Use NumPy operations instead of loops
3. **Caching**: Cache static files with proper headers
4. **Async Operations**: Use async/await for I/O operations

### Security Considerations

1. **Input Validation**: Always validate image data
2. **File Size Limits**: Limit uploaded image size
3. **Rate Limiting**: Add rate limiting for production
4. **CORS**: Restrict origins in production

---

## Learning Resources

### FastAPI Resources
- [FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/best-practices/)
- [Building ML APIs with FastAPI](https://testdriven.io/blog/fastapi-machine-learning/)
- [FastAPI WebSocket Guide](https://fastapi.tiangolo.com/advanced/websockets/)

### Frontend Development
- [MDN Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [JavaScript Async/Await](https://javascript.info/async-await)
- [CSS Grid Layout](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [Touch Events](https://developer.mozilla.org/en-US/docs/Web/API/Touch_events)

### ML Model Serving
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Model Deployment Best Practices](https://neptune.ai/blog/model-deployment-best-practices)
- [MLOps Principles](https://ml-ops.org/content/mlops-principles)

### Web Security
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Web Security Basics](https://developer.mozilla.org/en-US/docs/Web/Security)
- [API Security Checklist](https://github.com/shieldfy/API-Security-Checklist)

---

## Milestone 5 Checklist

Before moving to Milestone 6, ensure you've completed:

### Backend Implementation
- [ ] Created FastAPI application in `src/mnist_classifier/api/app.py`
- [ ] Implemented model loading functionality
- [ ] Created `/predict` endpoint with base64 image handling
- [ ] Added `/health` endpoint for monitoring
- [ ] Configured CORS middleware
- [ ] Set up static file serving
- [ ] Configured Jinja2 templates

### Frontend Implementation
- [ ] Created HTML template with canvas element
- [ ] Implemented CSS styling for responsive design
- [ ] Built JavaScript drawing functionality
- [ ] Added touch support for mobile devices
- [ ] Implemented prediction API calls
- [ ] Created result display UI

### Integration & Testing
- [ ] Tested drawing on desktop browsers
- [ ] Tested drawing on mobile devices
- [ ] Verified predictions return correct format
- [ ] Tested error handling for invalid inputs
- [ ] Checked UI responsiveness
- [ ] Validated all API endpoints

### Documentation & Verification
- [ ] Added API documentation (automatic with FastAPI)
- [ ] Created helper scripts (run_web_app.py, test_web_api.py)
- [ ] Run verification script successfully
- [ ] Tested complete user flow

---

## Conclusion

🎉 **Outstanding work on completing Milestone 5!** You've successfully built a complete web application for your MNIST classifier. This is a significant achievement that combines multiple technologies:

### What You've Accomplished
1. **Built a RESTful API** using FastAPI with proper structure and error handling
2. **Created an interactive frontend** with real-time drawing capabilities
3. **Integrated ML model serving** with web technologies
4. **Implemented cross-platform compatibility** (desktop and mobile)
5. **Established client-server communication** with modern web standards

### Key Skills Developed
- **Full-stack development**: Backend API + Frontend UI
- **Asynchronous programming**: Modern Python async/await
- **Web security basics**: CORS, input validation
- **API design**: RESTful principles, JSON communication
- **Frontend interactivity**: Canvas API, event handling
- **Mobile-first design**: Responsive UI, touch support

### Real-World Applications
The patterns you've learned here apply to many production ML systems:
- **Medical imaging**: Drawing regions of interest
- **Document processing**: Signature verification
- **Quality control**: Defect marking interfaces
- **Educational tools**: Interactive learning applications

### What's Next?
In Milestone 6, you'll add the final piece: **visualizing what the neural network "sees"** when making predictions. This will provide insights into how your model works and create an even more engaging user experience.

Take a moment to appreciate your progress. You've built a complete, production-ready web application that serves machine learning predictions in real-time. Try showing it to friends or colleagues - draw some digits and watch their reactions when the model correctly predicts what they drew!

Ready for the final milestone? Let's add neural network visualization and complete your MNIST classifier!
