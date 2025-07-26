"""
Training Script for MNIST CNN Model

This module handles the complete training pipeline including:
- Data loading and preparation
- Model creation and compilation
- Training with callbacks
- Evaluation and visualization
- Model saving and export
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.mnist_classifier.data_pipeline import MNISTDataPipeline
from src.mnist_classifier.models.cnn_model import (
    compile_model,
    count_parameters,
    create_cnn_model,
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

    def __init__(
        self, model_name: str = "mnist_cnn", output_dir: str = "models/experiments"
    ):
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

    def prepare_data(self, validation_split: float = 0.1) -> dict:
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
            normalize_method="standard",
            use_cache=False,
        )

        print("✅ Data prepared:")
        print(f"   Training: {self.data['x_train'].shape}")
        print(f"   Validation: {self.data['x_val'].shape}")
        print(f"   Test: {self.data['x_test'].shape}")

        return self.data

    def create_model(self, model_type: str = "standard") -> keras.Model:
        """
        Create and compile the model.

        Args:
            model_type: 'standard' or 'simple'

        Returns:
            Compiled Keras model
        """
        print(f"\n🏗️ Creating {model_type} model...")

        if model_type == "standard":
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

    def create_callbacks(self) -> list[keras.callbacks.Callback]:
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
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        callbacks.append(checkpoint)

        # 2. Early Stopping - Stop if no improvement
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop)

        # 3. Reduce Learning Rate - Lower LR when stuck
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        )
        callbacks.append(reduce_lr)

        # 4. TensorBoard - Logging for visualization
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(self.log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        )
        callbacks.append(tensorboard)

        # 5. CSV Logger - Save metrics to file
        csv_logger = keras.callbacks.CSVLogger(
            filename=str(self.output_dir / "training_history.csv"),
            separator=",",
            append=False,
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
                print(
                    f"   Learning Rate: {self.model.optimizer.learning_rate.numpy():.6f}"
                )

        callbacks.append(TrainingProgress())

        return callbacks

    def train(
        self,
        epochs: int = 30,
        batch_size: int = 128,
        model_type: str = "standard",
        validation_split: float = 0.1,
    ) -> dict:
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
            "model_name": self.model_name,
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "timestamp": self.timestamp,
        }

        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
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
            x=self.data["x_train"],
            y=self.data["y_train"],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.data["x_val"], self.data["y_val"]),
            callbacks=callbacks,
            verbose=1,
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

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with test metrics
        """
        print("\n📏 Evaluating on test set...")

        test_loss, test_acc, test_top3 = self.model.evaluate(
            self.data["x_test"], self.data["y_test"], batch_size=256, verbose=1
        )

        results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_top3_accuracy": test_top3,
        }

        print("\n📊 Test Results:")
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
        weights_path = self.output_dir / "model_weights.weights.h5"
        self.model.save_weights(str(weights_path))
        print(f"   Saved weights: {weights_path}")

        # Export TensorFlow SavedModel format for serving
        tf_path = self.output_dir / "saved_model"
        self.model.export(str(tf_path))
        print(f"   Exported TensorFlow SavedModel: {tf_path}")

        # Save TFLite version for mobile
        self.save_tflite_model()

    def save_tflite_model(self):
        """Convert and save TFLite model for mobile deployment."""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            tflite_path = self.output_dir / "model.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

            print(f"   Saved TFLite model: {tflite_path}")
            print(f"   TFLite size: {len(tflite_model) / 1024:.1f} KB")
        except Exception as e:
            print(f"   Could not save TFLite: {e}")

    def plot_training_history(self):
        """Plot training and validation metrics."""
        history = self.history.history
        epochs = range(1, len(history["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(epochs, history["loss"], "b-", label="Training Loss")
        ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
        ax1.set_title("Model Loss", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, history["accuracy"], "b-", label="Training Accuracy")
        ax2.plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy")
        ax2.set_title("Model Accuracy", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add target line
        ax2.axhline(y=0.98, color="g", linestyle="--", alpha=0.5, label="Target (98%)")

        plt.tight_layout()
        plot_path = self.plot_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   Saved training history plot: {plot_path}")

    def plot_confusion_matrix(self):
        """Create confusion matrix visualization."""
        from sklearn.metrics import confusion_matrix

        # Get predictions
        y_true = np.argmax(self.data["y_test"], axis=1)
        y_pred = np.argmax(self.model.predict(self.data["x_test"]), axis=1)

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(10),
            yticklabels=range(10),
        )
        plt.title("Confusion Matrix", fontsize=16)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Add accuracy per class
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracy):
            plt.text(10.5, i + 0.5, f"{acc:.2%}", ha="left", va="center")

        plt.tight_layout()
        plot_path = self.plot_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   Saved confusion matrix: {plot_path}")

    def visualize_predictions(self, n_samples: int = 20):
        """Visualize model predictions on test samples."""
        # Get random test samples
        indices = np.random.choice(len(self.data["x_test"]), n_samples, replace=False)
        x_samples = self.data["x_test"][indices]
        y_true = np.argmax(self.data["y_test"][indices], axis=1)

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
            ax.imshow(img, cmap="gray")

            # Color based on correctness
            color = "green" if y_true[i] == y_pred[i] else "red"

            # Add prediction info
            confidence = y_pred_probs[i].max() * 100
            ax.set_title(
                f"True: {y_true[i]}, Pred: {y_pred[i]}\n" f"Conf: {confidence:.1f}%",
                color=color,
                fontsize=10,
            )
            ax.axis("off")

        plt.suptitle("Model Predictions on Test Samples", fontsize=16)
        plt.tight_layout()

        plot_path = self.plot_dir / "predictions_sample.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   Saved predictions visualization: {plot_path}")

    def visualize_misclassified(self, n_samples: int = 20):
        """Visualize misclassified examples."""
        # Get all predictions
        y_true = np.argmax(self.data["y_test"], axis=1)
        y_pred_probs = self.model.predict(self.data["x_test"])
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
            img = self.data["x_test"][idx].squeeze()
            ax.imshow(img, cmap="gray")

            # Get top 3 predictions
            top3_idx = np.argsort(y_pred_probs[idx])[-3:][::-1]
            top3_probs = y_pred_probs[idx][top3_idx]

            # Add info
            title = f"True: {y_true[idx]}\n"
            title += f"Pred: {y_pred[idx]} ({top3_probs[0]*100:.1f}%)\n"
            title += f"2nd: {top3_idx[1]} ({top3_probs[1]*100:.1f}%)"

            ax.set_title(title, color="red", fontsize=9)
            ax.axis("off")

        # Hide unused subplots
        for i in range(n_show, len(axes)):
            axes[i].axis("off")

        plt.suptitle("Misclassified Examples", fontsize=16)
        plt.tight_layout()

        plot_path = self.plot_dir / "misclassified_examples.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   Saved misclassified examples: {plot_path}")

    def save_training_summary(self, training_time: float):
        """Save comprehensive training summary."""
        summary = {
            "run_name": self.run_name,
            "config": self.config,
            "model_info": {
                "name": self.model.name,
                "parameters": count_parameters(self.model),
                "layers": len(self.model.layers),
            },
            "training_time_seconds": training_time,
            "final_metrics": {
                "train_loss": float(self.history.history["loss"][-1]),
                "train_accuracy": float(self.history.history["accuracy"][-1]),
                "val_loss": float(self.history.history["val_loss"][-1]),
                "val_accuracy": float(self.history.history["val_accuracy"][-1]),
            },
            "best_epoch": int(np.argmax(self.history.history["val_accuracy"]) + 1),
            "total_epochs": len(self.history.history["loss"]),
        }

        # Add test results if available
        if hasattr(self, "test_results"):
            summary["test_results"] = self.test_results

        # Save summary
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📝 Training summary saved: {summary_path}")

        # Create readable report
        self.create_training_report(summary)

    def create_training_report(self, summary: dict):
        """Create human-readable training report."""
        report_path = self.output_dir / "training_report.txt"

        with open(report_path, "w") as f:
            f.write("MNIST CNN Training Report\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"Run Name: {summary['run_name']}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Model Configuration:\n")
            f.write(f"  Model Type: {summary['config']['model_type']}\n")
            f.write(f"  Batch Size: {summary['config']['batch_size']}\n")
            f.write(f"  Max Epochs: {summary['config']['epochs']}\n")
            f.write(
                f"  Validation Split: {summary['config']['validation_split']:.1%}\n\n"
            )

            f.write("Model Architecture:\n")
            f.write(
                f"  Total Parameters: {summary['model_info']['parameters']['total']:,}\n"
            )
            f.write(
                f"  Trainable Parameters: {summary['model_info']['parameters']['trainable']:,}\n"
            )
            f.write(f"  Number of Layers: {summary['model_info']['layers']}\n\n")

            f.write("Training Results:\n")
            f.write(
                f"  Training Time: {summary['training_time_seconds']:.1f} seconds\n"
            )
            f.write(f"  Total Epochs: {summary['total_epochs']}\n")
            f.write(f"  Best Epoch: {summary['best_epoch']}\n\n")

            f.write("Final Metrics:\n")
            f.write(
                f"  Training Accuracy: {summary['final_metrics']['train_accuracy']:.4f}\n"
            )
            f.write(
                f"  Validation Accuracy: {summary['final_metrics']['val_accuracy']:.4f}\n"
            )
            f.write(f"  Training Loss: {summary['final_metrics']['train_loss']:.4f}\n")
            f.write(f"  Validation Loss: {summary['final_metrics']['val_loss']:.4f}\n")

            if "test_results" in summary:
                f.write("\nTest Set Performance:\n")
                f.write(
                    f"  Test Accuracy: {summary['test_results']['test_accuracy']:.4f}\n"
                )
                f.write(f"  Test Loss: {summary['test_results']['test_loss']:.4f}\n")

            f.write("\nModel Files:\n")
            f.write("  Keras Model: final_model.h5\n")
            f.write("  Weights Only: model_weights.h5\n")
            f.write("  TensorFlow SavedModel: saved_model/\n")
            f.write("  TFLite Model: model.tflite\n")

            f.write("\nVisualization Files:\n")
            f.write("  Training History: plots/training_history.png\n")
            f.write("  Confusion Matrix: plots/confusion_matrix.png\n")
            f.write("  Sample Predictions: plots/predictions_sample.png\n")

        print(f"   Training report saved: {report_path}")


def run_training_experiment(
    model_type: str = "standard",
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
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
        validation_split=validation_split,
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
    parser.add_argument(
        "--model",
        type=str,
        default="standard",
        choices=["standard", "simple"],
        help="Model type to train",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")

    args = parser.parse_args()

    print("🚀 MNIST CNN Training Script")
    print("=" * 60)

    # Run training
    trainer = run_training_experiment(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.val_split,
    )

    print("\n✅ Training completed successfully!")
    print(f"📁 Results saved to: {trainer.output_dir}")
    print("\n💡 To view training progress in TensorBoard:")
    print(f"   tensorboard --logdir {trainer.log_dir}")
