"""
Monte Carlo Dropout for Uncertainty Estimation

This module implements Monte Carlo Dropout (MC Dropout) for the MNIST classifier.
MC Dropout is a Bayesian approximation technique that enables uncertainty quantification
by keeping dropout active during inference and running multiple forward passes.

Key concepts:
- Epistemic uncertainty: Model uncertainty due to limited training data
- Predictive entropy: Overall uncertainty in predictions
- Mutual information: Captures model uncertainty specifically

Reference:
    Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning. ICML.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass
class MCDropoutResult:
    """Results from Monte Carlo Dropout inference.

    Attributes:
        mean_prediction: Mean probability distribution across samples
        predicted_class: Most likely class based on mean prediction
        confidence: Confidence in the prediction (max of mean probabilities)
        predictive_entropy: Entropy of mean prediction (total uncertainty)
        mutual_information: Model/epistemic uncertainty
        prediction_variance: Variance across MC samples for each class
        all_predictions: Raw predictions from all forward passes
        confidence_interval: 95% confidence interval for predicted class
    """

    mean_prediction: np.ndarray
    predicted_class: int
    confidence: float
    predictive_entropy: float
    mutual_information: float
    prediction_variance: np.ndarray
    all_predictions: np.ndarray
    confidence_interval: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": float(self.confidence),
            "mean_probabilities": {
                str(i): float(p) for i, p in enumerate(self.mean_prediction)
            },
            "uncertainty": {
                "predictive_entropy": float(self.predictive_entropy),
                "mutual_information": float(self.mutual_information),
                "confidence_interval_95": {
                    "lower": float(self.confidence_interval[0]),
                    "upper": float(self.confidence_interval[1]),
                },
            },
            "variance_per_class": {
                str(i): float(v) for i, v in enumerate(self.prediction_variance)
            },
            "num_samples": len(self.all_predictions),
        }


class MCDropoutPredictor:
    """Monte Carlo Dropout predictor for uncertainty estimation.

    This class wraps a trained model and enables MC Dropout inference
    by running multiple forward passes with dropout enabled.

    Example:
        >>> predictor = MCDropoutPredictor(model, n_samples=100)
        >>> result = predictor.predict(image_array)
        >>> print(f"Prediction: {result.predicted_class}")
        >>> print(f"Uncertainty: {result.predictive_entropy:.4f}")
    """

    def __init__(
        self,
        model: keras.Model,
        n_samples: int = 50,
        dropout_rate: float | None = None,
    ) -> None:
        """Initialize MC Dropout predictor.

        Args:
            model: Trained Keras model with dropout layers
            n_samples: Number of forward passes for MC sampling
            dropout_rate: Optional override for dropout rate (uses model's rate if None)
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # Create a function that runs the model with training=True (dropout enabled)
        self._mc_predict = self._create_mc_predict_function()

    def _create_mc_predict_function(self) -> Any:
        """Create a TensorFlow function for MC prediction with dropout enabled."""

        @tf.function
        def predict_with_dropout(x: tf.Tensor) -> tf.Tensor:
            # Run model with training=True to enable dropout
            return self.model(x, training=True)

        return predict_with_dropout

    def predict(self, x: np.ndarray) -> MCDropoutResult:
        """Run MC Dropout inference on input.

        Args:
            x: Input array of shape (1, 28, 28, 1) or (28, 28, 1)

        Returns:
            MCDropoutResult with predictions and uncertainty metrics
        """
        # Ensure batch dimension
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        # Convert to tensor
        x_tensor = tf.constant(x, dtype=tf.float32)

        # Run multiple forward passes with dropout enabled
        predictions = []
        for _ in range(self.n_samples):
            pred = self._mc_predict(x_tensor)
            predictions.append(pred.numpy()[0])

        all_predictions = np.array(predictions)

        # Calculate uncertainty metrics
        return calculate_uncertainty_metrics(all_predictions)

    def predict_batch(self, x: np.ndarray) -> list[MCDropoutResult]:
        """Run MC Dropout on a batch of inputs.

        Args:
            x: Input array of shape (batch_size, 28, 28, 1)

        Returns:
            List of MCDropoutResult for each input
        """
        results = []
        for i in range(len(x)):
            result = self.predict(x[i : i + 1])
            results.append(result)
        return results


def calculate_uncertainty_metrics(predictions: np.ndarray) -> MCDropoutResult:
    """Calculate uncertainty metrics from MC Dropout samples.

    Args:
        predictions: Array of shape (n_samples, n_classes) containing
                    probability distributions from multiple forward passes

    Returns:
        MCDropoutResult with computed metrics
    """
    # Mean prediction across samples
    mean_prediction = np.mean(predictions, axis=0)

    # Prediction variance for each class
    prediction_variance = np.var(predictions, axis=0)

    # Predicted class and confidence
    predicted_class = int(np.argmax(mean_prediction))
    confidence = float(mean_prediction[predicted_class])

    # Predictive entropy: H[y|x, D] = -sum(p * log(p))
    # Measures total uncertainty (epistemic + aleatoric)
    epsilon = 1e-10  # Avoid log(0)
    predictive_entropy = float(
        -np.sum(mean_prediction * np.log(mean_prediction + epsilon))
    )

    # Expected entropy: E[H[y|x, w]] = -mean(sum(p * log(p)))
    # Average entropy across samples (aleatoric uncertainty)
    sample_entropies = -np.sum(predictions * np.log(predictions + epsilon), axis=1)
    expected_entropy = float(np.mean(sample_entropies))

    # Mutual Information: I[y; w|x, D] = H[y|x,D] - E[H[y|x,w]]
    # Captures epistemic (model) uncertainty
    mutual_information = predictive_entropy - expected_entropy

    # 95% confidence interval for predicted class probability
    class_predictions = predictions[:, predicted_class]
    ci_lower = float(np.percentile(class_predictions, 2.5))
    ci_upper = float(np.percentile(class_predictions, 97.5))

    return MCDropoutResult(
        mean_prediction=mean_prediction,
        predicted_class=predicted_class,
        confidence=confidence,
        predictive_entropy=predictive_entropy,
        mutual_information=mutual_information,
        prediction_variance=prediction_variance,
        all_predictions=predictions,
        confidence_interval=(ci_lower, ci_upper),
    )


def mc_dropout_predict(
    model: keras.Model,
    x: np.ndarray,
    n_samples: int = 50,
) -> MCDropoutResult:
    """Convenience function for single MC Dropout prediction.

    Args:
        model: Trained Keras model
        x: Input image array
        n_samples: Number of MC samples

    Returns:
        MCDropoutResult with predictions and uncertainty
    """
    predictor = MCDropoutPredictor(model, n_samples=n_samples)
    return predictor.predict(x)


def get_uncertainty_level(result: MCDropoutResult) -> str:
    """Categorize uncertainty level based on metrics.

    Args:
        result: MC Dropout result

    Returns:
        String describing uncertainty level: "low", "medium", or "high"
    """
    # Use predictive entropy thresholds
    # For 10-class classification, max entropy is log(10) ≈ 2.3
    if result.predictive_entropy < 0.3:
        return "low"
    if result.predictive_entropy < 1.0:
        return "medium"
    return "high"


def create_uncertainty_summary(result: MCDropoutResult) -> str:
    """Create a human-readable summary of uncertainty.

    Args:
        result: MC Dropout result

    Returns:
        Formatted string with uncertainty analysis
    """
    level = get_uncertainty_level(result)
    ci_width = result.confidence_interval[1] - result.confidence_interval[0]

    lines = [
        f"Predicted digit: {result.predicted_class}",
        f"Confidence: {result.confidence:.1%}",
        f"95% CI: [{result.confidence_interval[0]:.1%}, "
        f"{result.confidence_interval[1]:.1%}]",
        f"CI width: {ci_width:.1%}",
        "",
        "Uncertainty Metrics:",
        f"  Predictive Entropy: {result.predictive_entropy:.4f}",
        f"  Mutual Information: {result.mutual_information:.4f}",
        f"  Uncertainty Level: {level.upper()}",
        "",
        "Interpretation:",
    ]

    if level == "low":
        lines.append("  The model is confident in this prediction.")
    elif level == "medium":
        lines.append("  The model shows moderate uncertainty. Consider reviewing.")
    else:
        lines.append("  High uncertainty detected. This prediction may be unreliable.")

    # Find alternative predictions
    top_indices = np.argsort(result.mean_prediction)[::-1][:3]
    if result.mean_prediction[top_indices[1]] > 0.1:
        lines.append("")
        lines.append("Alternative predictions:")
        for idx in top_indices[:3]:
            prob = result.mean_prediction[idx]
            var = result.prediction_variance[idx]
            lines.append(f"  Digit {idx}: {prob:.1%} (var: {var:.4f})")

    return "\n".join(lines)


if __name__ == "__main__":
    # Simple test/demo
    print("Monte Carlo Dropout Module")
    print("=" * 50)
    print("\nThis module provides uncertainty estimation for MNIST predictions.")
    print("\nUsage:")
    print("  from mnist_classifier.experiments import MCDropoutPredictor")
    print("  predictor = MCDropoutPredictor(model, n_samples=50)")
    print("  result = predictor.predict(image)")
    print("  print(result.predictive_entropy)")
