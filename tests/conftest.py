"""Pytest configuration and fixtures for MNIST Classifier tests."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture()
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path: Path to the temporary directory.
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture()
def mock_model_path(temp_dir: Path) -> Path:
    """Create a mock model file path.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path: Path to mock model file.
    """
    model_path = temp_dir / "model.h5"
    model_path.touch()
    return model_path
