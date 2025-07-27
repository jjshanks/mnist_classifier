#!/usr/bin/env python3
"""
Command-line tool for MNIST digit classification.

This script provides an easy-to-use interface for classifying handwritten
digits from image files using a trained neural network model.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mnist_classifier.cli.predict import main

if __name__ == "__main__":
    main()
