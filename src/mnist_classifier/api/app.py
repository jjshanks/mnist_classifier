"""
FastAPI Web Application for MNIST Digit Classification

This module provides a web API and interface for classifying
handwritten digits drawn by users in their browser.
"""

import base64
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from fastapi import (  # type: ignore[import-not-found]
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
from fastapi.responses import (  # type: ignore[import-not-found]
    HTMLResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles  # type: ignore[import-not-found]
from fastapi.templating import Jinja2Templates  # type: ignore[import-not-found]
from PIL import Image
from pydantic import BaseModel, Field  # type: ignore[import-not-found]
from tensorflow import keras

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mnist_classifier.cli.predict import (  # noqa: E402
    load_model as load_model_from_file,
)
from src.mnist_classifier.preprocess import (  # noqa: E402
    normalize_pixels,
    reshape_images,
)

# Create FastAPI app
app = FastAPI(
    title="MNIST Digit Classifier",
    description="Interactive web interface for classifying handwritten digits",
    version="1.0.0",
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
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {"image": "data:image/png;base64,iVBORw0KGgoAAAANS..."}
        }


class PredictionResponse(BaseModel):
    """Response model for digit prediction."""

    predicted_digit: int = Field(..., ge=0, le=9, description="Predicted digit (0-9)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each digit"
    )
    processing_time: float = Field(..., description="Processing time in milliseconds")

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "predicted_digit": 7,
                "confidence": 0.998,
                "probabilities": {
                    "0": 0.001,
                    "1": 0.001,
                    "2": 0.001,
                    "3": 0.001,
                    "4": 0.001,
                    "5": 0.001,
                    "6": 0.001,
                    "7": 0.998,
                    "8": 0.001,
                    "9": 0.001,
                },
                "processing_time": 23.5,
            }
        }


def load_model() -> keras.Model:
    """Load the trained model (singleton pattern)."""
    global model_instance  # noqa: PLW0603

    if model_instance is None:
        print("Loading model...")
        # Try different model paths
        model_paths = [
            Path("models/mnist_cnn_model"),
            Path("models/mnist_model"),
            Path("models/mnist_model.h5"),
        ]

        for model_path in model_paths:
            if model_path.exists():
                try:
                    model_instance = load_model_from_file(model_path)
                    print(f"Model loaded successfully from {model_path}!")
                    break
                except Exception as e:
                    print(f"Failed to load from {model_path}: {e}")

        if model_instance is None:
            raise RuntimeError("No trained model found. Please train a model first.")

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
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]

    # Decode base64 to bytes
    image_bytes = base64.b64decode(image_data)

    # Open image with PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Convert RGBA to RGB with white background
    if image.mode == "RGBA":
        # Create white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        # Paste image using alpha channel as mask
        background.paste(image, mask=image.split()[3])
        image = background

    # Convert to grayscale
    image = image.convert("L")

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
    return reshape_images(img_normalized, add_channel=True)


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not preload model: {e}")
        print("Model will be loaded on first request.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """
    Serve the main page with drawing canvas.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "MNIST Digit Classifier"}
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
        probabilities = {str(i): float(predictions[0][i]) for i in range(10)}

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return PredictionResponse(
            predicted_digit=predicted_digit,
            confidence=confidence,
            probabilities=probabilities,
            processing_time=processing_time,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e!s}",
        ) from e


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Status information about the service
    """
    try:
        model = load_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return {
        "status": "healthy" if model_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "version": "1.0.0",
    }


@app.get("/api/info")
async def api_info() -> dict[str, Any]:
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
            "total_params": model.count_params(),
        }
    except Exception:
        model_info = {"loaded": False}

    return {
        "name": "MNIST Digit Classifier API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Main web interface"},
            {
                "path": "/predict",
                "method": "POST",
                "description": "Predict digit from image",
            },
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/api/info", "method": "GET", "description": "API information"},
            {
                "path": "/docs",
                "method": "GET",
                "description": "Interactive API documentation",
            },
        ],
        "model": model_info,
    }


# Error handlers
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException) -> Any:  # noqa: ARG001
    """Handle 404 errors."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content={"detail": "Endpoint not found"})
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)


@app.exception_handler(500)
async def server_error(request: Request, exc: HTTPException) -> JSONResponse:  # noqa: ARG001
    """Handle 500 errors."""
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    """Run the application directly for development."""
    import uvicorn  # type: ignore[import-not-found]

    print("Starting MNIST Digit Classifier Web App...")
    print("Open http://localhost:8000 in your browser")

    uvicorn.run(
        "src.mnist_classifier.api.app:app",
        host="0.0.0.0",  # nosec B104
        port=8000,
        reload=True,
        log_level="info",
    )
