"""
CNN Model Architecture for MNIST Digit Classification

This module defines a Convolutional Neural Network optimized for
recognizing handwritten digits from the MNIST dataset.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    name: str = "mnist_cnn",
) -> keras.Model:
    """
    Create a CNN model for MNIST digit classification.

    This architecture is specifically designed for MNIST:
    - Small input size (28x28) requires careful feature extraction
    - Grayscale images need fewer initial filters than color images
    - Balanced between accuracy and training speed

    Architecture Overview:
    1. Feature Extraction: Two conv+pool blocks to detect patterns
    2. Feature Refinement: Additional conv layers for complex features
    3. Classification: Dense layers to map features to digit classes

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        name: Model name

    Returns:
        Compiled Keras model ready for training

    Example:
        >>> model = create_cnn_model()
        >>> model.summary()
    """

    # Input layer - explicitly define input shape
    inputs = keras.Input(shape=input_shape, name="digit_input")

    # First Convolutional Block
    # - 32 filters: Sufficient for initial edge/curve detection
    # - 3x3 kernel: Standard size, captures local patterns
    # - ReLU: Simple, effective for CNNs
    # - He normal: Good initialization for ReLU
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",  # Preserves spatial dimensions
        kernel_initializer="he_normal",
        name="conv1",
    )(inputs)

    # Batch Normalization: Stabilizes training
    x = layers.BatchNormalization(name="bn1")(x)

    # First Pooling: Reduces from 28x28 to 14x14
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Second Convolutional Block
    # - 64 filters: Increased capacity for complex patterns
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv2",
    )(x)

    x = layers.BatchNormalization(name="bn2")(x)

    # Second Pooling: Reduces from 14x14 to 7x7
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Third Convolutional Block (no pooling)
    # - 128 filters: Maximum feature extraction
    # - No pooling: Preserves remaining spatial information
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv3",
    )(x)

    x = layers.BatchNormalization(name="bn3")(x)

    # Global Average Pooling: Modern alternative to Flatten
    # - Reduces parameters
    # - Adds slight translational invariance
    # - Less prone to overfitting
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense Classification Head
    # - 128 units: Sufficient for MNIST complexity
    x = layers.Dense(
        units=128, activation="relu", kernel_initializer="he_normal", name="dense1"
    )(x)

    # Dropout: Prevent overfitting
    x = layers.Dropout(rate=0.5, name="dropout")(x)

    # Output layer
    # - Softmax: Converts to probabilities
    # - Units = num_classes: One per digit
    outputs = layers.Dense(
        units=num_classes, activation="softmax", name="digit_output"
    )(x)

    # Create model
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def create_simple_model(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    name: str = "simple_mnist",
) -> keras.Model:
    """
    Create a simpler CNN model for quick experimentation.

    This model has fewer parameters and trains faster, useful for:
    - Quick prototyping
    - Understanding CNN basics
    - Limited computational resources

    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        name: Model name

    Returns:
        Compiled Keras model
    """

    model: keras.Model = keras.Sequential(
        [
            # Input layer
            keras.Input(shape=input_shape),
            # Simple feature extraction
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Classification
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name=name,
    )
    return model


def compile_model(
    model: keras.Model, learning_rate: float = 0.001, optimizer: str = "adam"
) -> keras.Model:
    """
    Compile model with appropriate optimizer and loss function.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')

    Returns:
        Compiled model

    Details on choices:
    - Optimizer: Adam adapts learning rate per parameter
    - Loss: Categorical crossentropy for multi-class classification
    - Metrics: Accuracy is intuitive for balanced dataset
    """

    # Create optimizer
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Compile model
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",  # For one-hot encoded labels
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )

    return model


def get_model_summary_str(model: keras.Model) -> str:
    """
    Get model summary as a string.

    Args:
        model: Keras model

    Returns:
        Model summary string
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)


def count_parameters(model: keras.Model) -> dict[str, int]:
    """
    Count trainable and non-trainable parameters.

    Args:
        model: Keras model

    Returns:
        Dictionary with parameter counts
    """
    trainable_params = np.sum(
        [keras.backend.count_params(w) for w in model.trainable_weights]
    )
    non_trainable_params = np.sum(
        [keras.backend.count_params(w) for w in model.non_trainable_weights]
    )

    return {
        "trainable": int(trainable_params),
        "non_trainable": int(non_trainable_params),
        "total": int(trainable_params + non_trainable_params),
    }


def save_model_architecture(
    model: keras.Model, filepath: str = "models/architecture.json"
) -> None:
    """
    Save model architecture to JSON file.

    Args:
        model: Keras model
        filepath: Where to save architecture
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Get architecture as JSON
    architecture = json.loads(model.to_json())

    # Add custom metadata
    metadata = {
        "model_name": model.name,
        "input_shape": model.input_shape[1:],
        "output_shape": model.output_shape[1:],
        "parameters": count_parameters(model),
        "layers": len(model.layers),
    }

    # Combine
    full_architecture = {"metadata": metadata, "architecture": architecture}

    # Save
    with Path(filepath).open("w") as f:
        json.dump(full_architecture, f, indent=2)


def create_model_visualization(
    model: keras.Model, save_path: str = "models/model_plot.png"
) -> None:
    """
    Create a visualization of the model architecture.

    Args:
        model: Keras model
        save_path: Where to save the visualization
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        keras.utils.plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",  # Top to Bottom
            expand_nested=True,
            dpi=96,
        )
        print(f"Model visualization saved to {save_path}")
    except Exception as e:
        print(f"Could not create visualization: {e}")
        print("Install graphviz for model plotting: pip install pydot graphviz")


def create_visualization_model(base_model: keras.Model) -> keras.Model:
    """
    Create a model that outputs intermediate layer activations.

    This model has the same input as the base model but outputs
    activations from multiple layers for visualization.

    Args:
        base_model: Trained CNN model

    Returns:
        Model with multiple outputs for visualization
    """
    # Get the layers we want to visualize
    layer_names = ["conv1", "conv2", "conv3", "dense1"]

    # Extract the outputs of these layers
    layer_outputs = []
    for name in layer_names:
        try:
            layer = base_model.get_layer(name)
            layer_outputs.append(layer.output)
        except ValueError:
            print(f"Warning: Layer '{name}' not found in model")

    # Also include the final predictions
    layer_outputs.append(base_model.output)

    # Create a model with multiple outputs
    visualization_model = keras.Model(
        inputs=base_model.input, outputs=layer_outputs, name="visualization_model"
    )

    return visualization_model


def get_activation_model(model_path: str) -> tuple[keras.Model, keras.Model]:
    """
    Load a model and create its visualization version.

    Args:
        model_path: Path to saved model

    Returns:
        Tuple of (original_model, visualization_model)
    """
    # Load the trained model
    original_model = keras.models.load_model(model_path)

    # Create visualization model
    viz_model = create_visualization_model(original_model)

    return original_model, viz_model


def process_activations(activations: list[np.ndarray]) -> dict[str, Any]:
    """
    Process raw activations into a format suitable for visualization.

    Args:
        activations: List of activation arrays from different layers

    Returns:
        Dictionary with processed activations
    """
    processed = {}

    # Process convolutional layers
    conv_names = ["conv1", "conv2", "conv3"]
    for i, name in enumerate(conv_names):
        if i < len(activations) - 2:  # -2 for dense and predictions
            activation = activations[i][0]  # Remove batch dimension

            # For conv layers, we'll select the most active filters
            # Calculate the mean activation for each filter
            mean_activations = np.mean(activation, axis=(0, 1))

            # Get indices of top 16 most active filters
            top_indices = np.argsort(mean_activations)[-16:][::-1]

            # Extract top filters
            top_filters = activation[:, :, top_indices]

            processed[name] = {
                "shape": activation.shape,
                "num_filters": activation.shape[-1],
                "top_filters": top_filters,
                "top_indices": top_indices.tolist(),
                "mean_activations": mean_activations.tolist(),
            }

    # Process dense layer
    if len(activations) > 3:
        dense_activation = activations[-2][0]  # -2 is dense, -1 is predictions
        processed["dense"] = {
            "activations": dense_activation.tolist(),
            "shape": dense_activation.shape,
        }

    # Process final predictions
    predictions = activations[-1][0]
    processed["predictions"] = {
        "probabilities": predictions.tolist(),
        "predicted_class": int(np.argmax(predictions)),
        "confidence": float(np.max(predictions)),
    }

    return processed


def create_feature_map_grid(feature_maps: np.ndarray, max_maps: int = 16) -> np.ndarray:
    """
    Create a grid visualization of feature maps.

    Args:
        feature_maps: Array of shape (height, width, num_filters)
        max_maps: Maximum number of feature maps to include

    Returns:
        Single image containing grid of feature maps
    """
    n_maps = min(feature_maps.shape[-1], max_maps)
    grid_size = int(np.ceil(np.sqrt(n_maps)))

    # Create empty grid
    map_height, map_width = feature_maps.shape[:2]
    grid_height = grid_size * map_height + (grid_size - 1) * 2  # 2 pixel padding
    grid_width = grid_size * map_width + (grid_size - 1) * 2

    grid = np.zeros((grid_height, grid_width))

    # Fill grid with feature maps
    for i in range(n_maps):
        row = i // grid_size
        col = i % grid_size

        # Calculate position in grid
        y_start = row * (map_height + 2)
        y_end = y_start + map_height
        x_start = col * (map_width + 2)
        x_end = x_start + map_width

        # Normalize feature map to [0, 1]
        feature_map = feature_maps[:, :, i]
        if feature_map.max() > feature_map.min():
            feature_map = (feature_map - feature_map.min()) / (
                feature_map.max() - feature_map.min()
            )

        grid[y_start:y_end, x_start:x_end] = feature_map

    return grid


if __name__ == "__main__":
    """
    Test model creation and display information.
    """
    print("Testing CNN Model Creation")
    print("=" * 50)

    # Create models
    print("\n1. Creating standard CNN model...")
    model = create_cnn_model()
    model = compile_model(model)

    # Display summary
    print("\n2. Model Summary:")
    model.summary()

    # Count parameters
    params = count_parameters(model)
    print("\n3. Parameter Count:")
    print(f"   Trainable: {params['trainable']:,}")
    print(f"   Non-trainable: {params['non_trainable']:,}")
    print(f"   Total: {params['total']:,}")

    # Test with sample data
    print("\n4. Testing forward pass...")
    sample_input = np.random.randn(1, 28, 28, 1).astype(np.float32)
    output = model.predict(sample_input, verbose=0)
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sum: {output.sum():.3f} (should be ~1.0)")

    # Create simple model for comparison
    print("\n5. Creating simple model for comparison...")
    simple_model = create_simple_model()
    simple_model = compile_model(simple_model)
    simple_params = count_parameters(simple_model)
    print(f"   Simple model parameters: {simple_params['total']:,}")
    print(f"   Reduction: {params['total'] / simple_params['total']:.1f}x")

    # Save architecture
    print("\n6. Saving model architecture...")
    save_model_architecture(model, "models/cnn_architecture.json")
    create_model_visualization(model, "models/cnn_architecture.png")

    print("\n✅ Model creation test completed successfully!")
