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
        model_type="simple",  # Use simpler model
        epochs=5,  # Just 5 epochs
        batch_size=256,  # Larger batch for speed
        validation_split=0.1,
    )

    print("\n✅ Quick training completed!")
    print("This was just a test run. For full training, use:")
    print("  python -m src.mnist_classifier.training.train")

    return trainer


if __name__ == "__main__":
    quick_train()
