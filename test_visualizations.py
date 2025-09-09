#!/usr/bin/env python3
"""
Test script for neural network visualizations.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageDraw

from src.mnist_classifier.models.cnn_model import (
    create_feature_map_grid,
    get_activation_model,
    process_activations,
)
from src.mnist_classifier.visualization.activation_viz import (
    create_activation_plot,
    create_layer_summary_plot,
    create_probability_chart,
)


def create_test_digit(digit: int = 7) -> np.ndarray:
    """Create a test digit image."""
    img = Image.new("L", (28, 28), 0)
    draw = ImageDraw.Draw(img)

    # Draw a simple 7
    if digit == 7:
        draw.line([(8, 8), (20, 8)], fill=255, width=2)
        draw.line([(20, 8), (14, 20)], fill=255, width=2)

    return np.array(img).reshape(1, 28, 28, 1) / 255.0


def test_visualization_model():
    """Test the visualization model creation and activation extraction."""
    print("🧪 Testing Visualization Components")
    print("=" * 50)

    # Find a trained model (look for final_model.h5 or best_model.h5)
    model_paths = []
    for pattern in ["**/final_model.h5", "**/best_model.h5"]:
        model_paths.extend(Path("models/experiments").glob(pattern))

    if not model_paths:
        print("❌ No trained model found. Train a model first!")
        return

    model_path = model_paths[0]
    print(f"\n1. Loading model from: {model_path}")

    try:
        # Load models
        original_model, viz_model = get_activation_model(str(model_path))
        print("   ✅ Models loaded successfully")

        # Create test input
        test_image = create_test_digit()
        print("\n2. Created test digit image")

        # Get activations
        print("\n3. Extracting activations...")
        activations = viz_model.predict(test_image, verbose=0)
        print(f"   ✅ Got {len(activations)} activation outputs")

        # Process activations
        print("\n4. Processing activations...")
        processed = process_activations(activations)

        for layer_name, data in processed.items():
            if layer_name == "predictions":
                print(
                    f"   - {layer_name}: predicted {data['predicted_class']} "
                    f"with {data['confidence']:.1%} confidence"
                )
            elif "shape" in data:
                print(
                    f"   - {layer_name}: shape {data['shape']}, "
                    f"{data.get('num_filters', 'N/A')} filters"
                )

        # Test visualization creation
        print("\n5. Creating visualizations...")

        # Probability chart
        prob_chart = create_probability_chart(processed["predictions"]["probabilities"])
        print("   ✅ Probability chart created")

        # Layer plots
        for layer in ["conv1", "conv2", "conv3"]:
            if layer in processed:
                plot = create_activation_plot(processed, layer)
                print(f"   ✅ {layer} activation plot created")

        # Summary plot
        summary = create_layer_summary_plot(processed)
        print("   ✅ Summary plot created")

        # Test feature map grid
        if "conv1" in processed:
            grid = create_feature_map_grid(processed["conv1"]["top_filters"])
            print(f"\n6. Feature map grid created: {grid.shape}")

        print("\n✅ All visualization tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_visualization_model()
