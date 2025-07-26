#!/usr/bin/env python3
"""
Simple benchmark to compare CPU vs GPU training speed.
"""

import os
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress most TF messages

print("Loading TensorFlow and checking devices...")
import tensorflow as tf

# Check available devices
gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")

print("\nAvailable devices:")
print(f"  CPUs: {len(cpus)}")
print(f"  GPUs: {len(gpus)}")

if gpus:
    # Configure GPU memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU Name: {gpus[0].name}")

# Create a simple model for benchmarking
print("\nCreating benchmark model...")
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Generate dummy data
print("Generating test data...")
import numpy as np

x_train = np.random.random((10000, 28, 28, 1)).astype(np.float32)
y_train = np.random.randint(0, 10, 10000)

# Benchmark training
print("\n" + "=" * 50)
print("Running benchmark (training for 3 epochs)...")
print("=" * 50)

start_time = time.time()
history = model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)
end_time = time.time()

training_time = end_time - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Average time per epoch: {training_time/3:.2f} seconds")

if gpus:
    print("\n✅ GPU training is active!")
    print("   Your RTX 3080 Ti should train ~10-50x faster than CPU")
else:
    print("\n⚠️  Running on CPU only")
    print("   Use ./train_gpu.sh to enable GPU acceleration")
