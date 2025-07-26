#!/usr/bin/env python3
"""Quick test of the trained model."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Configure GPU environment before importing TensorFlow
from src.mnist_classifier.utils import setup_gpu_environment

setup_gpu_environment()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow import keras

# Load the latest model
model_path = "models/experiments/mnist_standard_20250724_205451/final_model.h5"
model = keras.models.load_model(model_path)

# Load test data
from src.mnist_classifier.data_pipeline import MNISTDataPipeline

pipeline = MNISTDataPipeline()
data = pipeline.prepare_data(use_cache=False)

# Evaluate
print("\n🧪 Evaluating model on 10,000 test images...")
results = model.evaluate(data["x_test"], data["y_test"], verbose=0)

print("\n📊 FINAL TEST RESULTS:")
print(f"   Test Loss: {results[0]:.4f}")
print(f"   Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
print(f"   Test Top-3 Accuracy: {results[2]:.4f} ({results[2]*100:.2f}%)")

# Check error distribution
predictions = model.predict(data["x_test"], verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(data["y_test"], axis=1)
errors = y_pred != y_true
total_errors = np.sum(errors)

print("\n🎯 PERFORMANCE SUMMARY:")
print("   Total test images: 10,000")
print(f"   Correctly classified: {10000 - total_errors:,}")
print(f"   Misclassified: {total_errors}")
print(f"   Error rate: {total_errors/100:.2f}%")

# Most confused pairs
if total_errors > 0:
    from collections import Counter

    confusions = []
    error_indices = np.where(errors)[0]
    for idx in error_indices:
        confusions.append(f"{y_true[idx]}→{y_pred[idx]}")

    print("\n🔄 Most common confusions:")
    for confusion, count in Counter(confusions).most_common(5):
        print(f"   {confusion}: {count} times")

print("\n✨ Model training was successful!")
print(f"📁 Model saved at: {model_path}")
