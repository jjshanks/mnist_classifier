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
import os
import sys
import traceback
from pathlib import Path

# CRITICAL: Set WSL2 CUDA paths before ANY imports that might load CUDA
if "microsoft" in os.uname().release.lower():
    # Running in WSL2 - set CUDA library path
    wsl_cuda_path = "/usr/lib/wsl/lib"
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if wsl_cuda_path not in current_ld_path:
        print(
            "Note: Setting WSL2 CUDA library path. For best results, use ./train_gpu.sh"
        )
        # This won't work for already loaded libraries, but we try anyway
        os.environ["LD_LIBRARY_PATH"] = (
            f"{wsl_cuda_path}:{current_ld_path}" if current_ld_path else wsl_cuda_path
        )

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure GPU environment before importing TensorFlow
from src.mnist_classifier.utils import setup_gpu_environment

setup_gpu_environment()

from src.mnist_classifier.training.train import run_training_experiment  # noqa: E402


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
        """,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="standard",
        choices=["standard", "simple"],
        help="Model architecture (default: standard)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split (default: 0.1)"
    )

    # Quick mode
    parser.add_argument(
        "--quick", action="store_true", help="Quick training mode (5 epochs)"
    )

    args = parser.parse_args()

    # Override for quick mode
    if args.quick:
        print("🚀 Quick training mode activated!")
        args.epochs = 5
        args.model = "simple"
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
            validation_split=args.val_split,
        )

        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {trainer.output_dir}")

        # Print quick summary
        test_acc = trainer.history.history["val_accuracy"][-1]
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
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
