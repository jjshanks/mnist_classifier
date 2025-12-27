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
from src.mnist_classifier.models.cnn_model import (  # noqa: E402
    create_visualization_model,
    get_activation_model,
    process_activations,
)
from src.mnist_classifier.preprocess import (  # noqa: E402
    normalize_pixels,
    reshape_images,
)
from src.mnist_classifier.visualization.activation_viz import (  # noqa: E402
    create_activation_html,
    create_layer_summary_plot,
    create_probability_chart,
    create_uncertainty_chart,
    create_mc_samples_distribution,
    create_uncertainty_gauge,
    create_mc_dropout_summary_html,
)
from src.mnist_classifier.experiments.mc_dropout import (  # noqa: E402
    MCDropoutPredictor,
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


# Global model instances
model_instance = None
viz_model_instance = None


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


def load_models() -> tuple[keras.Model, keras.Model]:
    """Load both the regular model and visualization model."""
    global model_instance, viz_model_instance  # noqa: PLW0603

    if model_instance is None or viz_model_instance is None:
        print("Loading models...")
        # Find latest model - look for complete model files, not weights
        model_paths = [
            Path("models/experiments"),
            Path("models"),
        ]

        h5_files = []
        for base_path in model_paths:
            if base_path.exists():
                # Look for final_model.h5 or best_model.h5 files
                h5_files.extend(base_path.glob("**/final_model.h5"))
                h5_files.extend(base_path.glob("**/best_model.h5"))
                # Also check in models directory
                h5_files.extend(base_path.glob("mnist_model.h5"))
                h5_files.extend(base_path.glob("mnist_cnn_model.h5"))

        # Filter out weights files
        h5_files = [f for f in h5_files if "weights" not in f.name.lower()]

        if h5_files:
            # Sort by modification time, newest first
            latest_model = max(h5_files, key=lambda p: p.stat().st_mtime)
            print(f"Loading model from: {latest_model}")
            model_instance, viz_model_instance = get_activation_model(str(latest_model))
            print("Models loaded successfully")
        else:
            # Try the original load_model approach
            try:
                model_instance = load_model()
                viz_model_instance = create_visualization_model(model_instance)
                print("Models loaded using fallback method")
            except Exception:
                raise RuntimeError(
                    "No trained model found. Please train a model first."
                )

    return model_instance, viz_model_instance


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
    """Load models on startup."""
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Could not preload models: {e}")
        print("Models will be loaded on first request.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """
    Serve the main page with drawing canvas.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "MNIST Digit Classifier"}
    )


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """
    Predict digit from canvas data with activation visualizations.
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    start_time = time.time()

    try:
        # Load models if not already loaded
        model, viz_model = load_models()

        # Get JSON data
        data = await request.json()
        image_data = data.get("image")

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="No image data provided",
            )

        # Process image
        image_array = preprocess_canvas_image(image_data)

        # Get predictions and activations
        activations = viz_model.predict(image_array, verbose=0)

        # Process activations
        processed_activations = process_activations(activations)

        # Create HTML visualizations
        conv1_plot = create_activation_html(processed_activations, "conv1")
        conv2_plot = create_activation_html(processed_activations, "conv2")
        conv3_plot = create_activation_html(processed_activations, "conv3")
        prob_chart = create_probability_chart(
            processed_activations["predictions"]["probabilities"]
        )
        summary_plot = create_layer_summary_plot(processed_activations)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Prepare response
        response = {
            "prediction": processed_activations["predictions"]["predicted_class"],
            "confidence": processed_activations["predictions"]["confidence"],
            "probabilities": processed_activations["predictions"]["probabilities"],
            "processing_time": processing_time,
            "visualizations": {
                "conv1": conv1_plot,
                "conv2": conv2_plot,
                "conv3": conv3_plot,
                "probability_chart": prob_chart,
                "summary": summary_plot,
            },
            "layer_info": {
                "conv1": {
                    "shape": processed_activations.get("conv1", {}).get("shape", []),
                    "num_filters": processed_activations.get("conv1", {}).get(
                        "num_filters", 0
                    ),
                },
                "conv2": {
                    "shape": processed_activations.get("conv2", {}).get("shape", []),
                    "num_filters": processed_activations.get("conv2", {}).get(
                        "num_filters", 0
                    ),
                },
                "conv3": {
                    "shape": processed_activations.get("conv3", {}).get("shape", []),
                    "num_filters": processed_activations.get("conv3", {}).get(
                        "num_filters", 0
                    ),
                },
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Prediction error: {e!s}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e!s}",
        ) from e


@app.post("/predict/mc-dropout")
async def predict_mc_dropout(request: Request) -> JSONResponse:
    """
    Predict digit with Monte Carlo Dropout uncertainty estimation.

    This endpoint runs multiple forward passes with dropout enabled
    to estimate prediction uncertainty.
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    start_time = time.time()

    try:
        # Load model
        model, _ = load_models()

        # Get JSON data
        data = await request.json()
        image_data = data.get("image")
        n_samples = data.get("n_samples", 50)  # Default 50 MC samples

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="No image data provided",
            )

        # Process image
        image_array = preprocess_canvas_image(image_data)

        # Create MC Dropout predictor and run inference
        mc_predictor = MCDropoutPredictor(model, n_samples=n_samples)
        mc_result = mc_predictor.predict(image_array)

        # Create visualizations
        uncertainty_chart = create_uncertainty_chart(
            mean_probs=mc_result.mean_prediction.tolist(),
            variances=mc_result.prediction_variance.tolist(),
            confidence_interval=mc_result.confidence_interval,
            predicted_class=mc_result.predicted_class,
        )
        samples_dist = create_mc_samples_distribution(
            all_predictions=mc_result.all_predictions,
            predicted_class=mc_result.predicted_class,
        )
        uncertainty_gauge = create_uncertainty_gauge(
            predictive_entropy=mc_result.predictive_entropy,
            mutual_information=mc_result.mutual_information,
        )
        summary_html = create_mc_dropout_summary_html(mc_result.to_dict())

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms

        # Prepare response
        response = {
            "prediction": mc_result.predicted_class,
            "confidence": mc_result.confidence,
            "mean_probabilities": {
                str(i): float(p) for i, p in enumerate(mc_result.mean_prediction)
            },
            "uncertainty": {
                "predictive_entropy": mc_result.predictive_entropy,
                "mutual_information": mc_result.mutual_information,
                "confidence_interval_95": {
                    "lower": mc_result.confidence_interval[0],
                    "upper": mc_result.confidence_interval[1],
                },
            },
            "variance_per_class": {
                str(i): float(v) for i, v in enumerate(mc_result.prediction_variance)
            },
            "n_samples": n_samples,
            "processing_time": processing_time,
            "visualizations": {
                "uncertainty_chart": uncertainty_chart,
                "samples_distribution": samples_dist,
                "uncertainty_gauge": uncertainty_gauge,
                "summary_html": summary_html,
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"MC Dropout prediction error: {e!s}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"MC Dropout prediction failed: {e!s}",
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


@app.get("/model-info")
async def get_model_info() -> dict[str, Any]:
    """
    Get information about the loaded model architecture.
    """
    try:
        model, _ = load_models()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        ) from None

    layer_info = []
    for layer in model.layers:
        info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": layer.output_shape,
            "params": layer.count_params(),
        }

        # Add layer-specific info
        if hasattr(layer, "filters"):
            info["filters"] = layer.filters
        if hasattr(layer, "kernel_size"):
            info["kernel_size"] = layer.kernel_size
        if hasattr(layer, "units"):
            info["units"] = layer.units

        layer_info.append(info)

    return {
        "model_name": model.name,
        "total_params": model.count_params(),
        "layers": layer_info,
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
            {
                "path": "/predict/mc-dropout",
                "method": "POST",
                "description": "Predict with MC Dropout uncertainty estimation",
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
