"""Test version information."""

from mnist_classifier import __version__


def test_version() -> None:
    """Test that version is correctly set."""
    assert __version__ == "0.1.0"
    assert isinstance(__version__, str)

    # Verify version format (semantic versioning)
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
