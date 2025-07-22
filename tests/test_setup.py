"""Tests for setup and verification scripts."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_project
import verify_setup


class TestSetupProject:
    """Test cases for setup_project.py functions."""

    def test_get_activation_command_linux(self) -> None:
        """Test activation command on Linux/Mac."""
        with patch("platform.system", return_value="Linux"):
            cmd = setup_project.get_activation_command()
            assert cmd == "source .venv/bin/activate"

    def test_get_activation_command_windows(self) -> None:
        """Test activation command on Windows."""
        with patch("platform.system", return_value="Windows"):
            cmd = setup_project.get_activation_command()
            assert cmd == ".venv\\Scripts\\activate"

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    def test_create_directories(
        self, mock_exists: MagicMock, mock_mkdir: MagicMock
    ) -> None:
        """Test directory creation."""
        mock_exists.return_value = False

        setup_project.create_directories()

        expected_dirs = ["src", "data", "notebooks", "docs", "static", "templates"]
        assert mock_mkdir.call_count == len(expected_dirs)
        mock_mkdir.assert_called_with(parents=True)


class TestVerifySetup:
    """Test cases for verify_setup.py functions."""

    def test_check_mark_true(self) -> None:
        """Test check mark for True condition."""
        assert verify_setup.check_mark(True) == "✅"

    def test_check_mark_false(self) -> None:
        """Test check mark for False condition."""
        assert verify_setup.check_mark(False) == "❌"

    @patch("pathlib.Path.is_dir")
    def test_check_directories_all_exist(self, mock_is_dir: MagicMock) -> None:
        """Test directory check when all directories exist."""
        mock_is_dir.return_value = True

        result = verify_setup.check_directories()

        assert result is True
        assert mock_is_dir.call_count == 6

    @patch("pathlib.Path.is_dir")
    def test_check_directories_some_missing(self, mock_is_dir: MagicMock) -> None:
        """Test directory check when some directories are missing."""
        # First 3 exist, last 3 don't
        mock_is_dir.side_effect = [True, True, True, False, False, False]

        result = verify_setup.check_directories()

        assert result is False
        assert mock_is_dir.call_count == 6
