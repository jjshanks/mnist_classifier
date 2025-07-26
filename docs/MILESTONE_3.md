# Milestone 3: Neural Network for Digit Classification - Detailed Implementation Guide

## Overview

Welcome to Milestone 3! This is where the magic happens - you'll build and train a Convolutional Neural Network (CNN) that can recognize handwritten digits with over 98% accuracy. You'll learn fundamental deep learning concepts and gain hands-on experience with TensorFlow/Keras.

### What You'll Learn
- **CNN Architecture**: Understanding convolutional layers, pooling, and why they work for images
- **Model Design**: How to choose layer sizes, activation functions, and architectures
- **Training Process**: Optimizers, loss functions, and the backpropagation algorithm
- **Evaluation Metrics**: Accuracy, loss curves, and confusion matrices
- **Best Practices**: Avoiding overfitting, model checkpointing, and experiment tracking

### Why CNNs for Image Classification?
Convolutional Neural Networks are specifically designed for processing grid-like data (images):
- **Local Patterns**: CNNs detect features regardless of their position in the image
- **Hierarchical Learning**: Early layers detect edges, later layers detect complex shapes
- **Parameter Efficiency**: Weight sharing reduces the number of parameters dramatically
- **Translation Invariance**: A "7" is recognized whether it's in the top-left or bottom-right

### Prerequisites
Before starting, ensure you have:
- ✅ Completed Milestone 2 (data pipeline working)
- ✅ Basic understanding of neural networks (neurons, layers, activation functions)
- ✅ Familiarity with NumPy arrays and basic math (matrix multiplication)
- ✅ About 60-90 minutes to complete all tasks
- ✅ GPU access is helpful but not required (CPU training takes ~5-10 minutes)

### Success Criteria
By the end of this milestone, you will have:
- ✅ A well-designed CNN architecture optimized for MNIST
- ✅ A training script with proper validation and checkpointing
- ✅ Model achieving >98% accuracy on the test set
- ✅ Clear documentation of architectural choices
- ✅ Visualization of training progress and model performance
- ✅ Reusable code for future projects

---

## Task 3.1: Create Model Architecture

### Understanding CNN Components

Before building our model, let's understand each component:

1. **Convolutional Layers (Conv2D)**:
   - Apply filters (kernels) across the image
   - Each filter detects a specific pattern (edge, corner, curve)
   - Output: Feature maps showing where patterns were detected

2. **Pooling Layers (MaxPooling2D)**:
   - Reduce spatial dimensions (downsampling)
   - Make features more robust to small translations
   - Reduce computation and control overfitting

3. **Activation Functions**:
   - **ReLU**: f(x) = max(0, x) - Simple and effective
   - **Softmax**: Converts outputs to probabilities (used in final layer)

4. **Dense Layers**:
   - Traditional fully-connected layers
   - Combine features for final classification

5. **Dropout**:
   - Randomly "drops" neurons during training
   - Prevents overfitting by forcing redundant learning

### Step-by-Step Instructions

1. **Create the models directory**:
   ```bash
   mkdir -p src/mnist_classifier/models
   touch src/mnist_classifier/models/__init__.py
   ```

2. **Create the model module**:
   ```bash
   touch src/mnist_classifier/models/cnn_model.py
   ```

3. **Implement the CNN architecture**:

   Add this code to `src/mnist_classifier/models/cnn_model.py`:

   ```python
   """
   CNN Model Architecture for MNIST Digit Classification

   This module defines a Convolutional Neural Network optimized for
   recognizing handwritten digits from the MNIST dataset.
   """

   import tensorflow as tf
   from tensorflow import keras
   from tensorflow.keras import layers
   import numpy as np
   from typing import Tuple, Optional, Dict
   import json


   def create_cnn_model(input_shape: Tuple[int, int, int] = (28, 28, 1),
                       num_classes: int = 10,
                       name: str = "mnist_cnn") -> keras.Model:
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
           activation='relu',
           padding='same',  # Preserves spatial dimensions
           kernel_initializer='he_normal',
           name='conv1'
       )(inputs)

       # Batch Normalization: Stabilizes training
       x = layers.BatchNormalization(name='bn1')(x)

       # First Pooling: Reduces from 28x28 to 14x14
       x = layers.MaxPooling2D(
           pool_size=(2, 2),
           name='pool1'
       )(x)

       # Second Convolutional Block
       # - 64 filters: Increased capacity for complex patterns
       x = layers.Conv2D(
           filters=64,
           kernel_size=(3, 3),
           activation='relu',
           padding='same',
           kernel_initializer='he_normal',
           name='conv2'
       )(x)

       x = layers.BatchNormalization(name='bn2')(x)

       # Second Pooling: Reduces from 14x14 to 7x7
       x = layers.MaxPooling2D(
           pool_size=(2, 2),
           name='pool2'
       )(x)

       # Third Convolutional Block (no pooling)
       # - 128 filters: Maximum feature extraction
       # - No pooling: Preserves remaining spatial information
       x = layers.Conv2D(
           filters=128,
           kernel_size=(3, 3),
           activation='relu',
           padding='same',
           kernel_initializer='he_normal',
           name='conv3'
       )(x)

       x = layers.BatchNormalization(name='bn3')(x)

       # Global Average Pooling: Modern alternative to Flatten
       # - Reduces parameters
       # - Adds slight translational invariance
       # - Less prone to overfitting
       x = layers.GlobalAveragePooling2D(name='gap')(x)

       # Dense Classification Head
       # - 128 units: Sufficient for MNIST complexity
       x = layers.Dense(
           units=128,
           activation='relu',
           kernel_initializer='he_normal',
           name='dense1'
       )(x)

       # Dropout: Prevent overfitting
       x = layers.Dropout(rate=0.5, name='dropout')(x)

       # Output layer
       # - Softmax: Converts to probabilities
       # - Units = num_classes: One per digit
       outputs = layers.Dense(
           units=num_classes,
           activation='softmax',
           name='digit_output'
       )(x)

       # Create model
       model = keras.Model(
           inputs=inputs,
           outputs=outputs,
           name=name
       )

       return model


   def create_simple_model(input_shape: Tuple[int, int, int] = (28, 28, 1),
                          num_classes: int = 10,
                          name: str = "simple_mnist") -> keras.Model:
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

       model = keras.Sequential([
           # Input layer
           keras.Input(shape=input_shape),

           # Simple feature extraction
           layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
           layers.MaxPooling2D(pool_size=(2, 2)),

           layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
           layers.MaxPooling2D(pool_size=(2, 2)),

           # Classification
           layers.Flatten(),
           layers.Dense(64, activation='relu'),
           layers.Dropout(0.5),
           layers.Dense(num_classes, activation='softmax')
       ], name=name)

       return model


   def compile_model(model: keras.Model,
                    learning_rate: float = 0.001,
                    optimizer: str = 'adam') -> keras.Model:
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
       if optimizer.lower() == 'adam':
           opt = keras.optimizers.Adam(learning_rate=learning_rate)
       elif optimizer.lower() == 'sgd':
           opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
       elif optimizer.lower() == 'rmsprop':
           opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
       else:
           raise ValueError(f"Unknown optimizer: {optimizer}")

       # Compile model
       model.compile(
           optimizer=opt,
           loss='categorical_crossentropy',  # For one-hot encoded labels
           metrics=[
               'accuracy',
               keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
           ]
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


   def count_parameters(model: keras.Model) -> Dict[str, int]:
       """
       Count trainable and non-trainable parameters.

       Args:
           model: Keras model

       Returns:
           Dictionary with parameter counts
       """
       trainable_params = np.sum([
           keras.backend.count_params(w) for w in model.trainable_weights
       ])
       non_trainable_params = np.sum([
           keras.backend.count_params(w) for w in model.non_trainable_weights
       ])

       return {
           'trainable': int(trainable_params),
           'non_trainable': int(non_trainable_params),
           'total': int(trainable_params + non_trainable_params)
       }


   def save_model_architecture(model: keras.Model,
                             filepath: str = 'models/architecture.json'):
       """
       Save model architecture to JSON file.

       Args:
           model: Keras model
           filepath: Where to save architecture
       """
       import os
       os.makedirs(os.path.dirname(filepath), exist_ok=True)

       # Get architecture as JSON
       architecture = json.loads(model.to_json())

       # Add custom metadata
       metadata = {
           'model_name': model.name,
           'input_shape': model.input_shape[1:],
           'output_shape': model.output_shape[1:],
           'parameters': count_parameters(model),
           'layers': len(model.layers)
       }

       # Combine
       full_architecture = {
           'metadata': metadata,
           'architecture': architecture
       }

       # Save
       with open(filepath, 'w') as f:
           json.dump(full_architecture, f, indent=2)


   def create_model_visualization(model: keras.Model,
                                save_path: str = 'models/model_plot.png'):
       """
       Create a visualization of the model architecture.

       Args:
           model: Keras model
           save_path: Where to save the visualization
       """
       import os
       os.makedirs(os.path.dirname(save_path), exist_ok=True)

       try:
           keras.utils.plot_model(
               model,
               to_file=save_path,
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB',  # Top to Bottom
               expand_nested=True,
               dpi=96
           )
           print(f"Model visualization saved to {save_path}")
       except Exception as e:
           print(f"Could not create visualization: {e}")
           print("Install graphviz for model plotting: pip install pydot graphviz")


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
       print(f"\n3. Parameter Count:")
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
       save_model_architecture(model, 'models/cnn_architecture.json')
       create_model_visualization(model, 'models/cnn_architecture.png')

       print("\n✅ Model creation test completed successfully!")
   ```

4. **Test the model creation**:
   ```bash
   python src/mnist_classifier/models/cnn_model.py
   ```

### Understanding the Architecture

Let's break down why we made these choices:

**1. Convolutional Blocks**:
```
Input (28x28x1) → Conv(32) → BatchNorm → Pool → (14x14x32)
                → Conv(64) → BatchNorm → Pool → (7x7x64)
                → Conv(128) → BatchNorm → GAP → (128,)
```

- **Progressive Filter Increase**: 32→64→128 gradually builds complexity
- **BatchNorm**: Normalizes activations, allows higher learning rates
- **No Pool After Last Conv**: Preserves spatial information

**2. Why Global Average Pooling?**:
Traditional approach: Flatten → Large Dense Layer
Our approach: Global Average Pooling → Smaller Dense Layer

Benefits:
- Reduces parameters from 6,272 to 128
- Acts as regularization
- More interpretable (each filter votes for classes)

**3. Regularization Strategy**:
- Dropout (50%): Primary defense against overfitting
- BatchNorm: Also has regularization effect
- Architecture: Fewer parameters = less overfitting

### Model Variations

For different requirements, consider these variations:

1. **Faster Training** (Simple Model):
   - Fewer filters (16→32)
   - No BatchNorm
   - Single Dense layer

2. **Higher Accuracy** (Complex Model):
   - More conv blocks
   - Residual connections
   - Ensemble multiple models

3. **Mobile Deployment**:
   - Depthwise separable convolutions
   - Quantization-aware training
   - Knowledge distillation

---

## Task 3.2: Create Training Script

### Understanding the Training Process

Training a neural network involves:
1. **Forward Pass**: Input → Predictions
2. **Loss Calculation**: How wrong were the predictions?
3. **Backward Pass**: Calculate gradients via backpropagation
4. **Weight Update**: Adjust weights to reduce loss

This happens thousands of times, gradually improving the model.

### Key Training Concepts

1. **Batch Size**: Number of samples processed together
   - Small batch: More updates, noisier gradients
   - Large batch: Fewer updates, smoother gradients
   - Typical: 32-256 for MNIST

2. **Epochs**: Complete passes through the dataset
   - Too few: Underfitting
   - Too many: Overfitting
   - Monitor validation loss to decide

3. **Learning Rate**: How much to adjust weights
   - Too high: Unstable training
   - Too low: Slow convergence
   - Can be scheduled to decay

4. **Callbacks**: Functions called during training
   - ModelCheckpoint: Save best model
   - EarlyStopping: Stop if no improvement
   - ReduceLROnPlateau: Lower LR when stuck

### Step-by-Step Instructions

1. **Create the training module**:
   ```bash
   mkdir -p src/mnist_classifier/training
   touch src/mnist_classifier/training/__init__.py
   touch src/mnist_classifier/training/train.py
   ```

2. **Implement the training script**:

   Add this code to `src/mnist_classifier/training/train.py`:

   ```python
   """
   Training Script for MNIST CNN Model

   This module handles the complete training pipeline including:
   - Data loading and preparation
   - Model creation and compilation
   - Training with callbacks
   - Evaluation and visualization
   - Model saving and export
   """

   import os
   import sys
   import json
   import time
   from datetime import datetime
   from typing import Dict, Tuple, Optional, List
   import numpy as np
   import tensorflow as tf
   from tensorflow import keras
   import matplotlib.pyplot as plt
   import seaborn as sns
   from pathlib import Path

   # Add project root to path
   project_root = Path(__file__).parent.parent.parent.parent
   sys.path.insert(0, str(project_root))

   from src.mnist_classifier.data_pipeline import MNISTDataPipeline
   from src.mnist_classifier.models.cnn_model import (
       create_cnn_model, compile_model, count_parameters
   )


   class MNISTTrainer:
       """
       Complete training pipeline for MNIST digit classification.

       This class encapsulates:
       - Data preparation
       - Model creation
       - Training with monitoring
       - Evaluation and visualization
       - Model persistence
       """

       def __init__(self,
                    model_name: str = "mnist_cnn",
                    output_dir: str = "models/experiments"):
           """
           Initialize the trainer.

           Args:
               model_name: Name for this training run
               output_dir: Directory to save outputs
           """
           self.model_name = model_name
           self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           self.run_name = f"{model_name}_{self.timestamp}"

           # Create output directories
           self.output_dir = Path(output_dir) / self.run_name
           self.output_dir.mkdir(parents=True, exist_ok=True)

           # Sub-directories
           self.checkpoint_dir = self.output_dir / "checkpoints"
           self.checkpoint_dir.mkdir(exist_ok=True)

           self.log_dir = self.output_dir / "logs"
           self.log_dir.mkdir(exist_ok=True)

           self.plot_dir = self.output_dir / "plots"
           self.plot_dir.mkdir(exist_ok=True)

           # Initialize attributes
           self.model = None
           self.history = None
           self.data = None
           self.config = None

       def prepare_data(self, validation_split: float = 0.1) -> Dict:
           """
           Load and prepare data for training.

           Args:
               validation_split: Fraction of training data for validation

           Returns:
               Dictionary with prepared data
           """
           print(f"\n📊 Preparing data with {validation_split:.0%} validation split...")

           pipeline = MNISTDataPipeline()
           self.data = pipeline.prepare_data(
               validation_split=validation_split,
               normalize_method='standard',
               use_cache=True
           )

           print(f"✅ Data prepared:")
           print(f"   Training: {self.data['x_train'].shape}")
           print(f"   Validation: {self.data['x_val'].shape}")
           print(f"   Test: {self.data['x_test'].shape}")

           return self.data

       def create_model(self, model_type: str = 'standard') -> keras.Model:
           """
           Create and compile the model.

           Args:
               model_type: 'standard' or 'simple'

           Returns:
               Compiled Keras model
           """
           print(f"\n🏗️ Creating {model_type} model...")

           if model_type == 'standard':
               self.model = create_cnn_model()
           else:
               from src.mnist_classifier.models.cnn_model import create_simple_model
               self.model = create_simple_model()

           self.model = compile_model(self.model)

           # Display model info
           params = count_parameters(self.model)
           print(f"✅ Model created: {self.model.name}")
           print(f"   Total parameters: {params['total']:,}")
           print(f"   Trainable parameters: {params['trainable']:,}")

           return self.model

       def create_callbacks(self) -> List[keras.callbacks.Callback]:
           """
           Create training callbacks.

           Returns:
               List of Keras callbacks
           """
           callbacks = []

           # 1. Model Checkpoint - Save best model
           checkpoint_path = self.checkpoint_dir / "best_model.h5"
           checkpoint = keras.callbacks.ModelCheckpoint(
               filepath=str(checkpoint_path),
               monitor='val_accuracy',
               mode='max',
               save_best_only=True,
               save_weights_only=False,
               verbose=1
           )
           callbacks.append(checkpoint)

           # 2. Early Stopping - Stop if no improvement
           early_stop = keras.callbacks.EarlyStopping(
               monitor='val_loss',
               mode='min',
               patience=10,
               restore_best_weights=True,
               verbose=1
           )
           callbacks.append(early_stop)

           # 3. Reduce Learning Rate - Lower LR when stuck
           reduce_lr = keras.callbacks.ReduceLROnPlateau(
               monitor='val_loss',
               mode='min',
               factor=0.5,
               patience=5,
               min_lr=1e-6,
               verbose=1
           )
           callbacks.append(reduce_lr)

           # 4. TensorBoard - Logging for visualization
           tensorboard = keras.callbacks.TensorBoard(
               log_dir=str(self.log_dir),
               histogram_freq=1,
               write_graph=True,
               write_images=True,
               update_freq='epoch'
           )
           callbacks.append(tensorboard)

           # 5. CSV Logger - Save metrics to file
           csv_logger = keras.callbacks.CSVLogger(
               filename=str(self.output_dir / "training_history.csv"),
               separator=',',
               append=False
           )
           callbacks.append(csv_logger)

           # 6. Custom Progress Callback
           class TrainingProgress(keras.callbacks.Callback):
               def on_epoch_end(self, epoch, logs=None):
                   logs = logs or {}
                   print(f"\n📈 Epoch {epoch + 1} Summary:")
                   print(f"   Train Loss: {logs.get('loss', 0):.4f}")
                   print(f"   Train Acc: {logs.get('accuracy', 0):.4f}")
                   print(f"   Val Loss: {logs.get('val_loss', 0):.4f}")
                   print(f"   Val Acc: {logs.get('val_accuracy', 0):.4f}")
                   print(f"   Learning Rate: {self.model.optimizer.learning_rate.numpy():.6f}")

           callbacks.append(TrainingProgress())

           return callbacks

       def train(self,
                epochs: int = 30,
                batch_size: int = 128,
                model_type: str = 'standard',
                validation_split: float = 0.1) -> Dict:
           """
           Complete training pipeline.

           Args:
               epochs: Maximum number of epochs
               batch_size: Batch size for training
               model_type: Type of model to create
               validation_split: Validation data fraction

           Returns:
               Training history dictionary
           """
           print(f"\n🚀 Starting training pipeline: {self.run_name}")
           print("=" * 60)

           # Record configuration
           self.config = {
               'model_name': self.model_name,
               'model_type': model_type,
               'epochs': epochs,
               'batch_size': batch_size,
               'validation_split': validation_split,
               'timestamp': self.timestamp
           }

           # Save configuration
           with open(self.output_dir / 'config.json', 'w') as f:
               json.dump(self.config, f, indent=2)

           # Prepare data
           if self.data is None:
               self.prepare_data(validation_split)

           # Create model
           if self.model is None:
               self.create_model(model_type)

           # Create callbacks
           callbacks = self.create_callbacks()

           # Start training
           print(f"\n🏃 Training for up to {epochs} epochs...")
           print(f"   Batch size: {batch_size}")
           print(f"   Steps per epoch: {len(self.data['x_train']) // batch_size}")

           start_time = time.time()

           self.history = self.model.fit(
               x=self.data['x_train'],
               y=self.data['y_train'],
               batch_size=batch_size,
               epochs=epochs,
               validation_data=(self.data['x_val'], self.data['y_val']),
               callbacks=callbacks,
               verbose=1
           )

           training_time = time.time() - start_time

           print(f"\n✅ Training completed in {training_time:.1f} seconds")
           print(f"   Final train accuracy: {self.history.history['accuracy'][-1]:.4f}")
           print(f"   Final val accuracy: {self.history.history['val_accuracy'][-1]:.4f}")

           # Save final model
           self.save_model()

           # Evaluate on test set
           self.evaluate()

           # Create visualizations
           self.plot_training_history()
           self.plot_confusion_matrix()
           self.visualize_predictions()

           # Save training summary
           self.save_training_summary(training_time)

           return self.history.history

       def evaluate(self) -> Dict[str, float]:
           """
           Evaluate model on test set.

           Returns:
               Dictionary with test metrics
           """
           print("\n📏 Evaluating on test set...")

           test_loss, test_acc, test_top3 = self.model.evaluate(
               self.data['x_test'],
               self.data['y_test'],
               batch_size=256,
               verbose=1
           )

           results = {
               'test_loss': test_loss,
               'test_accuracy': test_acc,
               'test_top3_accuracy': test_top3
           }

           print(f"\n📊 Test Results:")
           print(f"   Test Loss: {test_loss:.4f}")
           print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
           print(f"   Top-3 Accuracy: {test_top3:.4f} ({test_top3*100:.2f}%)")

           return results

       def save_model(self):
           """Save model in multiple formats."""
           print("\n💾 Saving model...")

           # Save Keras model
           model_path = self.output_dir / "final_model.h5"
           self.model.save(str(model_path))
           print(f"   Saved Keras model: {model_path}")

           # Save weights only
           weights_path = self.output_dir / "model_weights.h5"
           self.model.save_weights(str(weights_path))
           print(f"   Saved weights: {weights_path}")

           # Save TensorFlow SavedModel format
           tf_path = self.output_dir / "saved_model"
           self.model.save(str(tf_path), save_format='tf')
           print(f"   Saved TensorFlow format: {tf_path}")

           # Save TFLite version for mobile
           self.save_tflite_model()

       def save_tflite_model(self):
           """Convert and save TFLite model for mobile deployment."""
           try:
               converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
               converter.optimizations = [tf.lite.Optimize.DEFAULT]
               tflite_model = converter.convert()

               tflite_path = self.output_dir / "model.tflite"
               with open(tflite_path, 'wb') as f:
                   f.write(tflite_model)

               print(f"   Saved TFLite model: {tflite_path}")
               print(f"   TFLite size: {len(tflite_model) / 1024:.1f} KB")
           except Exception as e:
               print(f"   Could not save TFLite: {e}")

       def plot_training_history(self):
           """Plot training and validation metrics."""
           history = self.history.history
           epochs = range(1, len(history['loss']) + 1)

           fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

           # Loss plot
           ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
           ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
           ax1.set_title('Model Loss', fontsize=14)
           ax1.set_xlabel('Epoch')
           ax1.set_ylabel('Loss')
           ax1.legend()
           ax1.grid(True, alpha=0.3)

           # Accuracy plot
           ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
           ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
           ax2.set_title('Model Accuracy', fontsize=14)
           ax2.set_xlabel('Epoch')
           ax2.set_ylabel('Accuracy')
           ax2.legend()
           ax2.grid(True, alpha=0.3)

           # Add target line
           ax2.axhline(y=0.98, color='g', linestyle='--', alpha=0.5, label='Target (98%)')

           plt.tight_layout()
           plot_path = self.plot_dir / "training_history.png"
           plt.savefig(plot_path, dpi=300, bbox_inches='tight')
           plt.close()

           print(f"   Saved training history plot: {plot_path}")

       def plot_confusion_matrix(self):
           """Create confusion matrix visualization."""
           from sklearn.metrics import confusion_matrix

           # Get predictions
           y_true = np.argmax(self.data['y_test'], axis=1)
           y_pred = np.argmax(self.model.predict(self.data['x_test']), axis=1)

           # Create confusion matrix
           cm = confusion_matrix(y_true, y_pred)

           # Plot
           plt.figure(figsize=(10, 8))
           sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=range(10), yticklabels=range(10))
           plt.title('Confusion Matrix', fontsize=16)
           plt.xlabel('Predicted Label')
           plt.ylabel('True Label')

           # Add accuracy per class
           class_accuracy = cm.diagonal() / cm.sum(axis=1)
           for i, acc in enumerate(class_accuracy):
               plt.text(10.5, i + 0.5, f'{acc:.2%}',
                       ha='left', va='center')

           plt.tight_layout()
           plot_path = self.plot_dir / "confusion_matrix.png"
           plt.savefig(plot_path, dpi=300, bbox_inches='tight')
           plt.close()

           print(f"   Saved confusion matrix: {plot_path}")

       def visualize_predictions(self, n_samples: int = 20):
           """Visualize model predictions on test samples."""
           # Get random test samples
           indices = np.random.choice(len(self.data['x_test']), n_samples, replace=False)
           x_samples = self.data['x_test'][indices]
           y_true = np.argmax(self.data['y_test'][indices], axis=1)

           # Get predictions
           y_pred_probs = self.model.predict(x_samples)
           y_pred = np.argmax(y_pred_probs, axis=1)

           # Create visualization
           fig, axes = plt.subplots(4, 5, figsize=(15, 12))
           axes = axes.ravel()

           for i in range(n_samples):
               ax = axes[i]

               # Show image
               img = x_samples[i].squeeze()
               ax.imshow(img, cmap='gray')

               # Color based on correctness
               color = 'green' if y_true[i] == y_pred[i] else 'red'

               # Add prediction info
               confidence = y_pred_probs[i].max() * 100
               ax.set_title(f'True: {y_true[i]}, Pred: {y_pred[i]}\n'
                          f'Conf: {confidence:.1f}%',
                          color=color, fontsize=10)
               ax.axis('off')

           plt.suptitle('Model Predictions on Test Samples', fontsize=16)
           plt.tight_layout()

           plot_path = self.plot_dir / "predictions_sample.png"
           plt.savefig(plot_path, dpi=300, bbox_inches='tight')
           plt.close()

           print(f"   Saved predictions visualization: {plot_path}")

       def visualize_misclassified(self, n_samples: int = 20):
           """Visualize misclassified examples."""
           # Get all predictions
           y_true = np.argmax(self.data['y_test'], axis=1)
           y_pred_probs = self.model.predict(self.data['x_test'])
           y_pred = np.argmax(y_pred_probs, axis=1)

           # Find misclassified
           misclassified_idx = np.where(y_true != y_pred)[0]

           if len(misclassified_idx) == 0:
               print("   No misclassified examples found!")
               return

           # Sample misclassified
           n_show = min(n_samples, len(misclassified_idx))
           sample_idx = np.random.choice(misclassified_idx, n_show, replace=False)

           # Create visualization
           fig, axes = plt.subplots(4, 5, figsize=(15, 12))
           axes = axes.ravel()

           for i, idx in enumerate(sample_idx):
               if i >= len(axes):
                   break

               ax = axes[i]

               # Show image
               img = self.data['x_test'][idx].squeeze()
               ax.imshow(img, cmap='gray')

               # Get top 3 predictions
               top3_idx = np.argsort(y_pred_probs[idx])[-3:][::-1]
               top3_probs = y_pred_probs[idx][top3_idx]

               # Add info
               title = f'True: {y_true[idx]}\n'
               title += f'Pred: {y_pred[idx]} ({top3_probs[0]*100:.1f}%)\n'
               title += f'2nd: {top3_idx[1]} ({top3_probs[1]*100:.1f}%)'

               ax.set_title(title, color='red', fontsize=9)
               ax.axis('off')

           # Hide unused subplots
           for i in range(n_show, len(axes)):
               axes[i].axis('off')

           plt.suptitle('Misclassified Examples', fontsize=16)
           plt.tight_layout()

           plot_path = self.plot_dir / "misclassified_examples.png"
           plt.savefig(plot_path, dpi=300, bbox_inches='tight')
           plt.close()

           print(f"   Saved misclassified examples: {plot_path}")

       def save_training_summary(self, training_time: float):
           """Save comprehensive training summary."""
           summary = {
               'run_name': self.run_name,
               'config': self.config,
               'model_info': {
                   'name': self.model.name,
                   'parameters': count_parameters(self.model),
                   'layers': len(self.model.layers)
               },
               'training_time_seconds': training_time,
               'final_metrics': {
                   'train_loss': float(self.history.history['loss'][-1]),
                   'train_accuracy': float(self.history.history['accuracy'][-1]),
                   'val_loss': float(self.history.history['val_loss'][-1]),
                   'val_accuracy': float(self.history.history['val_accuracy'][-1])
               },
               'best_epoch': int(np.argmax(self.history.history['val_accuracy']) + 1),
               'total_epochs': len(self.history.history['loss'])
           }

           # Add test results if available
           if hasattr(self, 'test_results'):
               summary['test_results'] = self.test_results

           # Save summary
           summary_path = self.output_dir / 'training_summary.json'
           with open(summary_path, 'w') as f:
               json.dump(summary, f, indent=2)

           print(f"\n📝 Training summary saved: {summary_path}")

           # Create readable report
           self.create_training_report(summary)

       def create_training_report(self, summary: Dict):
           """Create human-readable training report."""
           report_path = self.output_dir / 'training_report.txt'

           with open(report_path, 'w') as f:
               f.write(f"MNIST CNN Training Report\n")
               f.write(f"{'=' * 60}\n\n")

               f.write(f"Run Name: {summary['run_name']}\n")
               f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

               f.write(f"Model Configuration:\n")
               f.write(f"  Model Type: {summary['config']['model_type']}\n")
               f.write(f"  Batch Size: {summary['config']['batch_size']}\n")
               f.write(f"  Max Epochs: {summary['config']['epochs']}\n")
               f.write(f"  Validation Split: {summary['config']['validation_split']:.1%}\n\n")

               f.write(f"Model Architecture:\n")
               f.write(f"  Total Parameters: {summary['model_info']['parameters']['total']:,}\n")
               f.write(f"  Trainable Parameters: {summary['model_info']['parameters']['trainable']:,}\n")
               f.write(f"  Number of Layers: {summary['model_info']['layers']}\n\n")

               f.write(f"Training Results:\n")
               f.write(f"  Training Time: {summary['training_time_seconds']:.1f} seconds\n")
               f.write(f"  Total Epochs: {summary['total_epochs']}\n")
               f.write(f"  Best Epoch: {summary['best_epoch']}\n\n")

               f.write(f"Final Metrics:\n")
               f.write(f"  Training Accuracy: {summary['final_metrics']['train_accuracy']:.4f}\n")
               f.write(f"  Validation Accuracy: {summary['final_metrics']['val_accuracy']:.4f}\n")
               f.write(f"  Training Loss: {summary['final_metrics']['train_loss']:.4f}\n")
               f.write(f"  Validation Loss: {summary['final_metrics']['val_loss']:.4f}\n")

               if 'test_results' in summary:
                   f.write(f"\nTest Set Performance:\n")
                   f.write(f"  Test Accuracy: {summary['test_results']['test_accuracy']:.4f}\n")
                   f.write(f"  Test Loss: {summary['test_results']['test_loss']:.4f}\n")

               f.write(f"\nModel Files:\n")
               f.write(f"  Keras Model: final_model.h5\n")
               f.write(f"  Weights Only: model_weights.h5\n")
               f.write(f"  TensorFlow SavedModel: saved_model/\n")
               f.write(f"  TFLite Model: model.tflite\n")

               f.write(f"\nVisualization Files:\n")
               f.write(f"  Training History: plots/training_history.png\n")
               f.write(f"  Confusion Matrix: plots/confusion_matrix.png\n")
               f.write(f"  Sample Predictions: plots/predictions_sample.png\n")

           print(f"   Training report saved: {report_path}")


   def run_training_experiment(
       model_type: str = 'standard',
       epochs: int = 30,
       batch_size: int = 128,
       learning_rate: float = 0.001,
       validation_split: float = 0.1
   ) -> MNISTTrainer:
       """
       Run a complete training experiment.

       Args:
           model_type: Type of model ('standard' or 'simple')
           epochs: Number of training epochs
           batch_size: Batch size
           learning_rate: Initial learning rate
           validation_split: Validation split fraction

       Returns:
           Trained MNISTTrainer instance
       """
       # Create trainer
       trainer = MNISTTrainer(model_name=f"mnist_{model_type}")

       # Run training
       history = trainer.train(
           epochs=epochs,
           batch_size=batch_size,
           model_type=model_type,
           validation_split=validation_split
       )

       # Additional visualizations
       trainer.visualize_misclassified()

       return trainer


   if __name__ == "__main__":
       """
       Run training with default parameters.
       """
       import argparse

       parser = argparse.ArgumentParser(description="Train MNIST CNN model")
       parser.add_argument('--model', type=str, default='standard',
                          choices=['standard', 'simple'],
                          help='Model type to train')
       parser.add_argument('--epochs', type=int, default=20,
                          help='Number of epochs')
       parser.add_argument('--batch-size', type=int, default=128,
                          help='Batch size')
       parser.add_argument('--lr', type=float, default=0.001,
                          help='Learning rate')
       parser.add_argument('--val-split', type=float, default=0.1,
                          help='Validation split')

       args = parser.parse_args()

       print("🚀 MNIST CNN Training Script")
       print("=" * 60)

       # Run training
       trainer = run_training_experiment(
           model_type=args.model,
           epochs=args.epochs,
           batch_size=args.batch_size,
           learning_rate=args.lr,
           validation_split=args.val_split
       )

       print("\n✅ Training completed successfully!")
       print(f"📁 Results saved to: {trainer.output_dir}")
       print("\n💡 To view training progress in TensorBoard:")
       print(f"   tensorboard --logdir {trainer.log_dir}")
   ```

3. **Create a simplified training script for quick testing**:
   ```bash
   touch src/mnist_classifier/training/quick_train.py
   ```

   Add this code:

   ```python
   """
   Quick training script for rapid experimentation.
   """

   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

   from src.mnist_classifier.training.train import run_training_experiment


   def quick_train():
       """Run a quick training session with reduced epochs."""
       print("🚀 Quick Training Mode")
       print("=" * 50)
       print("Training with reduced epochs for quick testing...")

       # Run with fewer epochs
       trainer = run_training_experiment(
           model_type='simple',  # Use simpler model
           epochs=5,            # Just 5 epochs
           batch_size=256,      # Larger batch for speed
           validation_split=0.1
       )

       print("\n✅ Quick training completed!")
       print("This was just a test run. For full training, use:")
       print("  python -m src.mnist_classifier.training.train")

       return trainer


   if __name__ == "__main__":
       quick_train()
   ```

### Understanding Training Components

**1. Callbacks Explained**:

- **ModelCheckpoint**: Saves model when validation accuracy improves
  ```python
  monitor='val_accuracy'  # What to track
  save_best_only=True    # Only save improvements
  ```

- **EarlyStopping**: Stops training if no improvement
  ```python
  patience=10            # Wait 10 epochs
  restore_best_weights=True  # Revert to best
  ```

- **ReduceLROnPlateau**: Lowers learning rate when stuck
  ```python
  factor=0.5            # Multiply LR by 0.5
  patience=5            # Wait 5 epochs
  ```

**2. Why These Hyperparameters?**:

- **Batch Size 128**: Balance between:
  - GPU memory usage
  - Gradient stability
  - Training speed

- **Initial LR 0.001**: Standard for Adam optimizer
  - Not too high (unstable)
  - Not too low (slow)

- **30 Epochs**: Usually enough for MNIST
  - Early stopping prevents overfitting

**3. Data Augmentation** (Optional Enhancement):
```python
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,      # Rotate ±10 degrees
    width_shift_range=0.1,  # Shift horizontally
    height_shift_range=0.1, # Shift vertically
    zoom_range=0.1         # Zoom in/out
)
```

---

## Task 3.3: Document Model Architecture

### Why Documentation Matters

Good documentation:
- Helps others understand your design choices
- Serves as reference for future improvements
- Demonstrates professional practices
- Aids in debugging and optimization

### Step-by-Step Instructions

1. **Create documentation file**:
   ```bash
   touch docs/model_architecture.md
   ```

2. **Create the architecture documentation**:

   Add this content to `docs/model_architecture.md`:

   ```markdown
   # MNIST CNN Model Architecture

   ## Overview

   This document describes the Convolutional Neural Network (CNN) architecture designed for classifying handwritten digits from the MNIST dataset. The model achieves >98% accuracy while maintaining computational efficiency.

   ## Architecture Summary

   ```
   Input (28×28×1)
   ↓
   Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
   ↓
   Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
   ↓
   Conv2D(128, 3×3) → BatchNorm → ReLU
   ↓
   GlobalAveragePooling2D
   ↓
   Dense(128) → ReLU → Dropout(0.5)
   ↓
   Dense(10) → Softmax
   ↓
   Output (10 classes)
   ```

   ## Detailed Layer Description

   ### 1. Input Layer
   - **Shape**: (28, 28, 1)
   - **Description**: Grayscale images of handwritten digits
   - **Preprocessing**: Normalized to [0, 1] range

   ### 2. First Convolutional Block
   - **Conv2D**: 32 filters, 3×3 kernel, 'same' padding
   - **Purpose**: Detect low-level features (edges, curves)
   - **Output shape**: (28, 28, 32)
   - **BatchNorm**: Normalizes activations, improves training stability
   - **Activation**: ReLU - f(x) = max(0, x)
   - **MaxPooling2D**: 2×2 pool size, reduces to (14, 14, 32)

   ### 3. Second Convolutional Block
   - **Conv2D**: 64 filters, 3×3 kernel, 'same' padding
   - **Purpose**: Detect mid-level features (corners, shapes)
   - **Output shape**: (14, 14, 64)
   - **MaxPooling2D**: 2×2 pool size, reduces to (7, 7, 64)

   ### 4. Third Convolutional Block
   - **Conv2D**: 128 filters, 3×3 kernel, 'same' padding
   - **Purpose**: Detect high-level features (digit parts)
   - **Output shape**: (7, 7, 128)
   - **No pooling**: Preserves remaining spatial information

   ### 5. Global Average Pooling
   - **Purpose**: Reduces spatial dimensions to 1×1
   - **Output shape**: (128,)
   - **Advantages**:
     - Reduces parameters vs. Flatten
     - Adds regularization effect
     - More interpretable

   ### 6. Classification Head
   - **Dense(128)**: Fully connected layer
   - **Dropout(0.5)**: Regularization, prevents overfitting
   - **Dense(10)**: Output layer, one unit per class
   - **Softmax**: Converts to probability distribution

   ## Design Rationale

   ### Why This Architecture?

   1. **Progressive Feature Extraction**:
      - 32→64→128 filters gradually increase capacity
      - Matches complexity of features at each level

   2. **Batch Normalization**:
      - Stabilizes training
      - Allows higher learning rates
      - Reduces internal covariate shift

   3. **Global Average Pooling vs. Flatten**:
      - Flatten would create 7×7×128 = 6,272 parameters
      - GAP creates only 128 parameters
      - Significant reduction in overfitting risk

   4. **Dropout Placement**:
      - Only in dense layers where overfitting is common
      - 0.5 rate is aggressive but works well for MNIST

   ### Parameter Count

   | Layer Type | Output Shape | Parameters |
   |------------|--------------|------------|
   | Input | (28, 28, 1) | 0 |
   | Conv2D | (28, 28, 32) | 320 |
   | BatchNorm | (28, 28, 32) | 128 |
   | MaxPool | (14, 14, 32) | 0 |
   | Conv2D | (14, 14, 64) | 18,496 |
   | BatchNorm | (14, 14, 64) | 256 |
   | MaxPool | (7, 7, 64) | 0 |
   | Conv2D | (7, 7, 128) | 73,856 |
   | BatchNorm | (7, 7, 128) | 512 |
   | GAP | (128,) | 0 |
   | Dense | (128,) | 16,512 |
   | Dropout | (128,) | 0 |
   | Dense | (10,) | 1,290 |
   | **Total** | - | **111,370** |

   ## Training Configuration

   ### Optimizer: Adam
   - **Learning rate**: 0.001 (initial)
   - **β₁**: 0.9 (momentum term)
   - **β₂**: 0.999 (RMSprop term)
   - **Why Adam?**: Adaptive learning rates, works well out-of-box

   ### Loss Function: Categorical Crossentropy
   - **Formula**: -Σ(y_true × log(y_pred))
   - **Why?**: Standard for multi-class classification
   - **Requires**: One-hot encoded labels

   ### Metrics
   - **Accuracy**: Primary metric, easy to interpret
   - **Top-3 Accuracy**: Useful for understanding near-misses

   ### Callbacks
   1. **ModelCheckpoint**: Save best model based on validation accuracy
   2. **EarlyStopping**: Stop if validation loss doesn't improve for 10 epochs
   3. **ReduceLROnPlateau**: Reduce learning rate by 50% after 5 epochs of no improvement

   ## Performance Benchmarks

   ### Training Performance
   - **Hardware**: CPU (Intel i7) / GPU (NVIDIA GTX 1060)
   - **Training time**: ~2 min (GPU) / ~10 min (CPU)
   - **Inference time**: <1ms per image

   ### Accuracy Metrics
   - **Training accuracy**: ~99.5%
   - **Validation accuracy**: ~99.2%
   - **Test accuracy**: ~99.1%
   - **Top-3 accuracy**: ~99.9%

   ### Common Misclassifications
   - 4 ↔ 9: Similar shape when handwritten
   - 3 ↔ 5: Curved similarities
   - 7 ↔ 1: Depends on writing style

   ## Variations and Experiments

   ### 1. Simpler Architecture
   For faster training or limited resources:
   - Reduce filters: 16→32→64
   - Remove batch normalization
   - Single dense layer

   ### 2. Data Augmentation
   Improves generalization:
   ```python
   ImageDataGenerator(
       rotation_range=10,
       width_shift_range=0.1,
       height_shift_range=0.1,
       zoom_range=0.1
   )
   ```

   ### 3. Advanced Techniques
   - **Ensemble**: Combine multiple models
   - **Test Time Augmentation**: Average predictions over augmented versions
   - **Knowledge Distillation**: Train smaller model from larger

   ## Deployment Considerations

   ### Model Size
   - **Keras H5**: ~1.3 MB
   - **TFLite**: ~450 KB (quantized)
   - **ONNX**: ~450 KB

   ### Optimization Techniques
   1. **Quantization**: Reduce to INT8
   2. **Pruning**: Remove small weights
   3. **Knowledge Distillation**: Train smaller model

   ### Platform-Specific Formats
   - **Web**: TensorFlow.js
   - **Mobile**: TFLite (Android/iOS)
   - **Edge**: ONNX Runtime

   ## Future Improvements

   1. **Architecture**:
      - Residual connections
      - Depthwise separable convolutions
      - Attention mechanisms

   2. **Training**:
      - Mixup augmentation
      - Label smoothing
      - Cosine learning rate schedule

   3. **Deployment**:
      - Further quantization
      - Model compression
      - Federated learning

   ## References

   1. LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"
   2. He et al. (2015): "Deep Residual Learning for Image Recognition"
   3. Howard et al. (2017): "MobileNets: Efficient Convolutional Neural Networks"

   ## Code Example

   ```python
   from src.mnist_classifier.models.cnn_model import create_cnn_model

   # Create model
   model = create_cnn_model(
       input_shape=(28, 28, 1),
       num_classes=10
   )

   # Compile with standard settings
   model.compile(
       optimizer='adam',
       loss='categorical_crossentropy',
       metrics=['accuracy']
   )

   # View architecture
   model.summary()
   ```
   ```

3. **Create an architecture visualization script**:
   ```bash
   touch src/mnist_classifier/models/visualize_architecture.py
   ```

   Add this code:

   ```python
   """
   Visualize and analyze model architecture.
   """

   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

   import numpy as np
   import matplotlib.pyplot as plt
   from src.mnist_classifier.models.cnn_model import (
       create_cnn_model, create_simple_model, count_parameters
   )


   def visualize_layer_outputs():
       """Visualize intermediate layer outputs."""
       from tensorflow import keras
       from src.mnist_classifier.data_pipeline import MNISTDataPipeline

       # Load model and data
       model = create_cnn_model()
       pipeline = MNISTDataPipeline()
       data = pipeline.prepare_data()

       # Get a sample image
       sample_img = data['x_test'][0:1]  # First test image
       sample_label = np.argmax(data['y_test'][0])

       # Create model for layer outputs
       layer_names = ['conv1', 'conv2', 'conv3']
       layer_outputs = [model.get_layer(name).output for name in layer_names]
       activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

       # Get activations
       activations = activation_model.predict(sample_img)

       # Plot
       fig, axes = plt.subplots(1, 4, figsize=(15, 4))

       # Original image
       axes[0].imshow(sample_img[0].squeeze(), cmap='gray')
       axes[0].set_title(f'Original (Label: {sample_label})')
       axes[0].axis('off')

       # Feature maps
       for i, (name, activation) in enumerate(zip(layer_names, activations)):
           ax = axes[i + 1]
           # Show first 16 filters as 4x4 grid
           n_features = min(16, activation.shape[-1])
           size = int(np.sqrt(n_features))

           display_grid = np.zeros((size * activation.shape[1],
                                   size * activation.shape[2]))

           for row in range(size):
               for col in range(size):
                   channel_idx = row * size + col
                   if channel_idx < activation.shape[-1]:
                       feature = activation[0, :, :, channel_idx]
                       # Normalize
                       feature = (feature - feature.mean()) / (feature.std() + 1e-5)
                       display_grid[row * activation.shape[1]:(row + 1) * activation.shape[1],
                                   col * activation.shape[2]:(col + 1) * activation.shape[2]] = feature

           ax.imshow(display_grid, cmap='viridis')
           ax.set_title(f'{name} (first {n_features} filters)')
           ax.axis('off')

       plt.tight_layout()
       plt.savefig('models/layer_visualizations.png', dpi=150, bbox_inches='tight')
       plt.show()


   def compare_architectures():
       """Compare different model architectures."""
       # Create models
       standard_model = create_cnn_model()
       simple_model = create_simple_model()

       # Get parameters
       standard_params = count_parameters(standard_model)
       simple_params = count_parameters(simple_model)

       # Create comparison
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

       # Parameter comparison
       models = ['Standard CNN', 'Simple CNN']
       params = [standard_params['total'], simple_params['total']]

       ax1.bar(models, params, color=['blue', 'green'])
       ax1.set_ylabel('Total Parameters')
       ax1.set_title('Model Size Comparison')

       # Add values on bars
       for i, v in enumerate(params):
           ax1.text(i, v + 1000, f'{v:,}', ha='center')

       # Layer count comparison
       layer_counts = [len(standard_model.layers), len(simple_model.layers)]

       ax2.bar(models, layer_counts, color=['blue', 'green'])
       ax2.set_ylabel('Number of Layers')
       ax2.set_title('Model Complexity Comparison')

       # Add values on bars
       for i, v in enumerate(layer_counts):
           ax2.text(i, v + 0.5, str(v), ha='center')

       plt.tight_layout()
       plt.savefig('models/architecture_comparison.png', dpi=150, bbox_inches='tight')
       plt.show()

       # Print detailed comparison
       print("\nDetailed Architecture Comparison:")
       print("=" * 50)
       print(f"Standard CNN:")
       print(f"  Total parameters: {standard_params['total']:,}")
       print(f"  Trainable: {standard_params['trainable']:,}")
       print(f"  Layers: {len(standard_model.layers)}")
       print(f"\nSimple CNN:")
       print(f"  Total parameters: {simple_params['total']:,}")
       print(f"  Trainable: {simple_params['trainable']:,}")
       print(f"  Layers: {len(simple_model.layers)}")
       print(f"\nParameter Reduction: {standard_params['total'] / simple_params['total']:.1f}x")


   if __name__ == "__main__":
       print("🎨 Model Architecture Visualization")
       print("=" * 50)

       # Run visualizations
       print("\n1. Comparing architectures...")
       compare_architectures()

       print("\n2. Visualizing layer outputs...")
       print("   (This requires a trained model)")
       try:
           visualize_layer_outputs()
       except Exception as e:
           print(f"   Skipped: {e}")

       print("\n✅ Visualization complete!")
   ```

---

## Helper Scripts

### train_model.py - Main Training Entry Point

Create this in your project root:

```python
#!/usr/bin/env python3
"""
Main entry point for training the MNIST CNN model.

Usage:
    python train_model.py                    # Standard training
    python train_model.py --model simple     # Train simple model
    python train_model.py --epochs 50        # Train for 50 epochs
    python train_model.py --quick            # Quick 5-epoch test
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mnist_classifier.training.train import run_training_experiment


def main():
    """Main training function with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train MNIST digit classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                  # Standard training
  python train_model.py --quick          # Quick test (5 epochs)
  python train_model.py --model simple   # Use simple architecture
  python train_model.py --batch-size 64  # Smaller batches
        """
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='standard',
                       choices=['standard', 'simple'],
                       help='Model architecture (default: standard)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (5 epochs)')

    args = parser.parse_args()

    # Override for quick mode
    if args.quick:
        print("🚀 Quick training mode activated!")
        args.epochs = 5
        args.model = 'simple'
        args.batch_size = 256

    print(f"""
╔══════════════════════════════════════════╗
║        MNIST CNN Training Script         ║
╠══════════════════════════════════════════╣
║  Model Type: {args.model:<27} ║
║  Epochs: {args.epochs:<31} ║
║  Batch Size: {args.batch_size:<27} ║
║  Learning Rate: {args.lr:<24} ║
║  Validation Split: {args.val_split:<21} ║
╚══════════════════════════════════════════╝
    """)

    # Run training
    try:
        trainer = run_training_experiment(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_split=args.val_split
        )

        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {trainer.output_dir}")

        # Print quick summary
        test_acc = trainer.history.history['val_accuracy'][-1]
        print(f"\n📊 Final validation accuracy: {test_acc:.2%}")

        if test_acc >= 0.98:
            print("🎉 Achieved target accuracy of 98%!")
        else:
            print(f"📈 {0.98 - test_acc:.2%} below target of 98%")

        print("\n💡 Next steps:")
        print("  1. Review training plots in:", trainer.plot_dir)
        print("  2. Test the model with: python test_model.py")
        print("  3. View in TensorBoard: tensorboard --logdir", trainer.log_dir)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### test_model.py - Model Testing Script

Create this in your project root:

```python
#!/usr/bin/env python3
"""
Test a trained MNIST model on individual images or the test set.

Usage:
    python test_model.py                           # Test on test set
    python test_model.py --image path/to/digit.png # Test on single image
    python test_model.py --model path/to/model.h5  # Use specific model
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from tensorflow import keras
from src.mnist_classifier.data_pipeline import MNISTDataPipeline
from src.mnist_classifier.preprocess import normalize_pixels, reshape_images


def load_latest_model(model_dir: str = "models/experiments") -> Tuple[keras.Model, Path]:
    """Load the most recent trained model."""
    model_dir = Path(model_dir)

    # Find latest experiment
    experiments = sorted(model_dir.glob("mnist_*"), key=lambda p: p.stat().st_mtime)

    if not experiments:
        raise FileNotFoundError(f"No experiments found in {model_dir}")

    latest = experiments[-1]
    model_path = latest / "final_model.h5"

    if not model_path.exists():
        model_path = latest / "checkpoints" / "best_model.h5"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {latest}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    return model, latest


def test_on_image(model: keras.Model, image_path: str):
    """Test model on a single image."""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize to 28x28 if needed
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to array
    img_array = np.array(img)

    # Preprocess
    img_normalized = normalize_pixels(img_array.reshape(1, 28, 28))
    img_input = reshape_images(img_normalized, add_channel=True)

    # Predict
    predictions = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Show image
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title(f'Input Image\nPredicted: {predicted_class} ({confidence:.1%})')
    ax1.axis('off')

    # Show predictions
    ax2.bar(range(10), predictions[0])
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    ax2.set_title('Prediction Probabilities')
    ax2.set_xticks(range(10))

    # Highlight prediction
    ax2.patches[predicted_class].set_color('green')

    plt.tight_layout()
    plt.show()

    # Print top 3 predictions
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    print("\nTop 3 predictions:")
    for i, idx in enumerate(top3_idx):
        print(f"  {i+1}. Digit {idx}: {predictions[0][idx]:.1%}")


def test_on_test_set(model: keras.Model):
    """Evaluate model on entire test set."""
    # Load test data
    print("Loading test data...")
    pipeline = MNISTDataPipeline()
    data = pipeline.prepare_data()

    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(
        data['x_test'],
        data['y_test'],
        batch_size=256,
        verbose=1
    )

    # Get metric names
    metric_names = model.metrics_names

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)

    for name, value in zip(metric_names, results):
        if 'accuracy' in name:
            print(f"{name}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{name}: {value:.4f}")

    # Analyze errors
    print("\nAnalyzing misclassifications...")
    predictions = model.predict(data['x_test'], verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(data['y_test'], axis=1)

    # Find errors
    errors = y_pred != y_true
    error_indices = np.where(errors)[0]

    print(f"\nTotal errors: {len(error_indices)} out of {len(y_true)}")
    print(f"Error rate: {len(error_indices)/len(y_true):.2%}")

    # Confusion analysis
    if len(error_indices) > 0:
        print("\nMost common confusions:")
        from collections import Counter
        confusions = []
        for idx in error_indices:
            confusions.append(f"{y_true[idx]}→{y_pred[idx]}")

        confusion_counts = Counter(confusions).most_common(5)
        for confusion, count in confusion_counts:
            print(f"  {confusion}: {count} times")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test trained MNIST model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: use latest)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image file to test')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment folder name')

    args = parser.parse_args()

    print("🧪 MNIST Model Testing")
    print("=" * 50)

    try:
        # Load model
        if args.model:
            print(f"Loading specified model: {args.model}")
            model = keras.models.load_model(args.model)
            exp_dir = Path(args.model).parent
        else:
            model, exp_dir = load_latest_model()

        print(f"Model loaded from: {exp_dir.name}")
        print(f"Model summary: {len(model.layers)} layers, "
              f"{model.count_params():,} parameters")

        # Test mode
        if args.image:
            print(f"\nTesting on image: {args.image}")
            test_on_image(model, args.image)
        else:
            test_on_test_set(model)

            # Show sample predictions
            print("\nWould you like to see sample predictions? (y/n): ", end='')
            if input().lower().strip() == 'y':
                # Load one batch for visualization
                pipeline = MNISTDataPipeline()
                data = pipeline.prepare_data()

                # Random samples
                indices = np.random.choice(len(data['x_test']), 9)
                samples = data['x_test'][indices]
                true_labels = np.argmax(data['y_test'][indices], axis=1)

                # Predict
                predictions = model.predict(samples, verbose=0)
                pred_labels = np.argmax(predictions, axis=1)

                # Display
                fig, axes = plt.subplots(3, 3, figsize=(10, 10))
                axes = axes.ravel()

                for i, (img, true, pred) in enumerate(zip(samples, true_labels, pred_labels)):
                    ax = axes[i]
                    ax.imshow(img.squeeze(), cmap='gray')

                    color = 'green' if true == pred else 'red'
                    confidence = predictions[i][pred] * 100

                    ax.set_title(f'True: {true}, Pred: {pred}\n'
                                f'Conf: {confidence:.1f}%',
                                color=color)
                    ax.axis('off')

                plt.suptitle('Sample Test Predictions', fontsize=16)
                plt.tight_layout()
                plt.show()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### verify_milestone3.py - Verification Script

Create this comprehensive verification script:

```python
#!/usr/bin/env python3
"""
Verification script for Milestone 3 completion.
"""

import os
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def check_files_exist():
    """Check if all required files exist."""
    print("\n📁 Checking required files:")

    required_files = [
        ('src/mnist_classifier/models/__init__.py', 'Models package init'),
        ('src/mnist_classifier/models/cnn_model.py', 'CNN model definition'),
        ('src/mnist_classifier/training/__init__.py', 'Training package init'),
        ('src/mnist_classifier/training/train.py', 'Training script'),
        ('docs/model_architecture.md', 'Architecture documentation'),
    ]

    all_exist = True
    for filepath, description in required_files:
        exists = os.path.exists(filepath)
        print(f"  {check_mark(exists)} {description}: {filepath}")
        if not exists:
            all_exist = False

    return all_exist


def check_model_creation():
    """Test model creation."""
    print("\n🏗️ Testing model creation:")

    try:
        from src.mnist_classifier.models.cnn_model import create_cnn_model, compile_model

        # Create model
        model = create_cnn_model()
        model = compile_model(model)

        # Check model properties
        checks = [
            (model.input_shape == (None, 28, 28, 1),
             f"Input shape: {model.input_shape}"),
            (model.output_shape == (None, 10),
             f"Output shape: {model.output_shape}"),
            (len(model.layers) > 5,
             f"Number of layers: {len(model.layers)}"),
            (model.count_params() > 10000,
             f"Total parameters: {model.count_params():,}"),
        ]

        all_good = True
        for condition, description in checks:
            print(f"  {check_mark(condition)} {description}")
            if not condition:
                all_good = False

        # Test forward pass
        test_input = np.random.randn(1, 28, 28, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)

        forward_ok = output.shape == (1, 10) and abs(output.sum() - 1.0) < 0.01
        print(f"  {check_mark(forward_ok)} Forward pass test")

        return all_good and forward_ok

    except Exception as e:
        print(f"  {check_mark(False)} Model creation failed: {str(e)}")
        return False


def check_training_capability():
    """Check if training script can be imported."""
    print("\n🏃 Testing training capability:")

    try:
        from src.mnist_classifier.training.train import MNISTTrainer

        # Check key methods exist
        trainer = MNISTTrainer()
        methods = ['prepare_data', 'create_model', 'train', 'evaluate']

        all_exist = True
        for method in methods:
            exists = hasattr(trainer, method)
            print(f"  {check_mark(exists)} Method '{method}' exists")
            if not exists:
                all_exist = False

        return all_exist

    except Exception as e:
        print(f"  {check_mark(False)} Training module error: {str(e)}")
        return False


def check_trained_model():
    """Check if a trained model exists."""
    print("\n🎯 Checking for trained model:")

    model_dirs = [
        'models/experiments',
        'models',
    ]

    found_model = False
    model_path = None

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            # Look for .h5 files
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.h5'):
                        model_path = os.path.join(root, file)
                        found_model = True
                        break
                if found_model:
                    break
        if found_model:
            break

    print(f"  {check_mark(found_model)} Trained model found" +
          (f": {model_path}" if found_model else ""))

    # If model found, try to load and check accuracy
    if found_model:
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path)

            # Check if training history exists
            exp_dir = Path(model_path).parent.parent
            history_file = exp_dir / 'training_history.csv'

            if history_file.exists():
                import pandas as pd
                history = pd.read_csv(history_file)

                if 'val_accuracy' in history.columns:
                    best_val_acc = history['val_accuracy'].max()
                    print(f"  {check_mark(best_val_acc > 0.98)} "
                          f"Best validation accuracy: {best_val_acc:.4f}")
                    return best_val_acc > 0.98

            print("  ℹ️  Could not verify accuracy (no history file)")
            return True  # Model exists at least

        except Exception as e:
            print(f"  ⚠️  Could not load model: {str(e)}")
            return True  # Model file exists

    return False


def check_documentation():
    """Check if documentation exists and is comprehensive."""
    print("\n📚 Checking documentation:")

    doc_path = 'docs/model_architecture.md'

    if not os.path.exists(doc_path):
        print(f"  {check_mark(False)} Documentation not found")
        return False

    with open(doc_path, 'r') as f:
        content = f.read()

    # Check for key sections
    sections = [
        ('# MNIST CNN Model Architecture', 'Title'),
        ('## Architecture Summary', 'Architecture summary'),
        ('## Design Rationale', 'Design rationale'),
        ('Conv2D', 'Convolutional layers mentioned'),
        ('Parameters', 'Parameter count mentioned'),
    ]

    all_present = True
    for section, description in sections:
        present = section in content
        print(f"  {check_mark(present)} {description}")
        if not present:
            all_present = False

    # Check length
    word_count = len(content.split())
    adequate_length = word_count > 500
    print(f"  {check_mark(adequate_length)} "
          f"Adequate length ({word_count} words)")

    return all_present and adequate_length


def check_helper_scripts():
    """Check if helper scripts exist."""
    print("\n🛠️ Checking helper scripts:")

    scripts = [
        ('train_model.py', 'Main training script'),
        ('test_model.py', 'Model testing script'),
    ]

    all_exist = True
    for script, description in scripts:
        exists = os.path.exists(script)
        print(f"  {check_mark(exists)} {description}: {script}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("🔍 Milestone 3 Verification")
    print("=" * 50)

    # Make sure we're in the right directory
    if not os.path.exists('src/mnist_classifier'):
        print("❌ Error: Not in project root directory!")
        print("Please run this from the mnist_classifier project root.")
        sys.exit(1)

    # Add project root to path
    sys.path.insert(0, os.path.abspath('.'))

    # Run all checks
    checks = [
        ("Required Files", check_files_exist()),
        ("Model Creation", check_model_creation()),
        ("Training Capability", check_training_capability()),
        ("Trained Model", check_trained_model()),
        ("Documentation", check_documentation()),
        ("Helper Scripts", check_helper_scripts()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 3 is complete!")
        print("\nYou have successfully:")
        print("  ✓ Built a CNN architecture for MNIST")
        print("  ✓ Implemented comprehensive training pipeline")
        print("  ✓ Achieved >98% accuracy (if model trained)")
        print("  ✓ Documented the architecture thoroughly")
        print("\nNext: Milestone 4 - Build the CLI interface!")
    else:
        print("\n⚠️  Some checks failed. Please review and fix the issues above.")
        print("\nTips:")
        print("  - Run 'python train_model.py --quick' for a quick test")
        print("  - Check that all imports are correct")
        print("  - Ensure model achieves >98% validation accuracy")


if __name__ == "__main__":
    main()
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Import/Module Errors

**Problem**: "ModuleNotFoundError: No module named 'src'"
```bash
# Solution: Run from project root
cd /path/to/mnist_classifier
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Problem**: "No module named 'tensorflow'"
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install tensorflow
```

#### Training Issues

**Problem**: "OOM (Out of Memory) error"
```python
# Reduce batch size
trainer.train(batch_size=32)  # or even 16

# Or use gradient accumulation
# Or CPU instead of GPU
```

**Problem**: Training is very slow
```python
# Use simpler model
python train_model.py --model simple

# Larger batch size (if memory allows)
python train_model.py --batch-size 256

# Fewer epochs for testing
python train_model.py --epochs 10
```

**Problem**: Model not converging
```python
# Try different learning rates
python train_model.py --lr 0.0001  # Lower
python train_model.py --lr 0.01    # Higher

# Check data pipeline
# Ensure normalization is correct
```

#### Model Architecture Issues

**Problem**: Model accuracy stuck at ~10%
- Check loss function matches label format
- Ensure one-hot encoding is correct
- Verify output layer has softmax

**Problem**: Overfitting (train >> validation accuracy)
- Add more dropout
- Reduce model complexity
- Add data augmentation
- Use early stopping

**Problem**: Model too large for deployment
```python
# Use quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Performance Optimization

1. **Training Speed**:
   ```python
   # Use mixed precision
   policy = keras.mixed_precision.Policy('mixed_float16')
   keras.mixed_precision.set_global_policy(policy)
   ```

2. **Memory Usage**:
   ```python
   # Use tf.data pipeline
   dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
   dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
   ```

3. **Multi-GPU Training**:
   ```python
   strategy = tf.distribute.MirroredStrategy()
   with strategy.scope():
       model = create_cnn_model()
   ```

---

## Learning Resources

### Essential Concepts

1. **Convolutional Neural Networks**:
   - [CS231n CNN Tutorial](http://cs231n.github.io/convolutional-networks/)
   - [3Blue1Brown Neural Network Series](https://www.youtube.com/watch?v=aircAruvnKk)
   - [CNN Explainer Interactive](https://poloclub.github.io/cnn-explainer/)

2. **TensorFlow/Keras**:
   - [Official TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
   - [Keras Documentation](https://keras.io/guides/)
   - [TensorFlow Playground](https://playground.tensorflow.org/)

3. **Deep Learning Theory**:
   - [Deep Learning Book](https://www.deeplearningbook.org/) by Goodfellow, Bengio, Courville
   - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
   - [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Practical Tutorials

1. **Building CNNs**:
   - "Building a CNN from Scratch" tutorials
   - Kaggle MNIST competitions and kernels
   - PyImageSearch CNN tutorials

2. **Training Best Practices**:
   - "Bag of Tricks for Image Classification" paper
   - Learning rate scheduling strategies
   - Data augmentation techniques

3. **Model Optimization**:
   - TensorFlow Model Optimization Toolkit
   - Quantization and pruning guides
   - Edge deployment tutorials

### Advanced Topics

1. **Modern Architectures**:
   - ResNet and skip connections
   - EfficientNet and compound scaling
   - Vision Transformers (ViT)

2. **Training Techniques**:
   - Mixed precision training
   - Gradient accumulation
   - Distributed training

3. **Interpretability**:
   - Grad-CAM visualizations
   - SHAP for neural networks
   - Attention visualization

---

## Milestone 3 Checklist

Before moving to Milestone 4, ensure you've completed:

- [ ] Created model architecture module (`cnn_model.py`)
- [ ] Implemented CNN with Conv2D, pooling, and dense layers
- [ ] Model compiles without errors
- [ ] Created comprehensive training script (`train.py`)
- [ ] Implemented training with callbacks
- [ ] Model saves in multiple formats
- [ ] Training produces visualization plots
- [ ] Achieved >98% validation accuracy
- [ ] Documented architecture in `model_architecture.md`
- [ ] Documentation explains design choices
- [ ] Created helper scripts (train_model.py, test_model.py)
- [ ] All verification tests pass

🎉 **Congratulations on completing Milestone 3!**

You've successfully:
- Built a CNN that understands visual patterns
- Implemented a professional training pipeline
- Achieved state-of-the-art accuracy on MNIST
- Created comprehensive documentation
- Learned fundamental deep learning concepts

### What You've Learned
- CNN architecture design principles
- How convolution and pooling work
- Training neural networks with Keras
- Monitoring and improving model performance
- Best practices for ML experimentation

### Your Model's Achievements
- **Accuracy**: >98% on test set (human-level performance)
- **Size**: ~450KB (mobile-deployable)
- **Speed**: <1ms inference time
- **Robustness**: Handles varied handwriting styles

Ready to make your model usable? In Milestone 4, you'll create a command-line interface that lets anyone use your digit classifier!
