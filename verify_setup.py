#!/usr/bin/env python3
"""
Verification script to check if Milestone 1 is complete.
Run this to ensure all setup steps were successful.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path


def check_mark(condition: bool) -> str:
    """Return checkmark or X based on condition.

    Args:
        condition: Boolean value to check.

    Returns:
        str: Unicode checkmark if True, X if False.
    """
    return "✅" if condition else "❌"


def check_directories() -> bool:
    """Check if all required directories exist.

    Returns:
        bool: True if all directories exist, False otherwise.
    """
    print("\n📁 Checking directory structure:")

    required_dirs: list[str] = [
        "src",
        "data",
        "notebooks",
        "docs",
        "static",
        "templates",
    ]
    all_exist: bool = True

    for dir_name in required_dirs:
        exists = Path(dir_name).is_dir()
        print(f"  {check_mark(exists)} {dir_name}/")
        if not exists:
            all_exist = False

    return all_exist


def check_git() -> bool:
    """Check if git is initialized properly.

    Returns:
        bool: True if git is properly set up, False otherwise.
    """
    print("\n🔧 Checking Git setup:")

    git_exists: bool = Path(".git").exists()
    print(f"  {check_mark(git_exists)} Git repository initialized")

    gitignore_exists: bool = Path(".gitignore").exists()
    print(f"  {check_mark(gitignore_exists)} .gitignore file exists")

    # Check if there's at least one commit
    has_commits: bool = False
    if git_exists:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                capture_output=True,
                text=True,
                check=False,
            )
            has_commits = result.returncode == 0
        except Exception:
            pass

    print(f"  {check_mark(has_commits)} Has at least one commit")

    return git_exists and gitignore_exists and has_commits


def check_virtual_env() -> bool:
    """Check if virtual environment exists and is activated.

    Returns:
        bool: True if venv exists and is activated, False otherwise.
    """
    print("\n🐍 Checking virtual environment:")

    venv_exists: bool = Path(".venv").exists()
    print(f"  {check_mark(venv_exists)} Virtual environment exists")

    # Check if activated by looking at sys.prefix
    is_activated: bool = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"  {check_mark(is_activated)} Virtual environment is activated")

    return venv_exists and is_activated


def check_dependencies() -> bool:
    """Check if all required packages are installed.

    Returns:
        bool: True if all dependencies are installed, False otherwise.
    """
    print("\n📦 Checking dependencies:")

    packages: dict[str, str] = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "numpy": "numpy",
        "tensorflow": "tensorflow",
        "PIL": "pillow",
        "jupyter": "jupyter",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
    }

    all_installed: bool = True
    for import_name, display_name in packages.items():
        spec = importlib.util.find_spec(import_name)
        is_installed: bool = spec is not None
        print(f"  {check_mark(is_installed)} {display_name}")
        if not is_installed:
            all_installed = False

    pyproject_exists: bool = Path("pyproject.toml").exists()
    print(f"  {check_mark(pyproject_exists)} pyproject.toml exists")

    return all_installed and pyproject_exists


def check_documentation() -> bool:
    """Check if README.md exists.

    Returns:
        bool: True if documentation is adequate, False otherwise.
    """
    print("\n📝 Checking documentation:")

    readme_exists: bool = Path("README.md").exists()
    print(f"  {check_mark(readme_exists)} README.md exists")

    # Check if README has content
    has_content: bool = False
    if readme_exists:
        content: str = Path("README.md").read_text().strip()
        has_content = len(content) > 100  # At least 100 characters

    print(f"  {check_mark(has_content)} README.md has substantial content")

    return readme_exists and has_content


def main() -> None:
    """Run all verification checks."""
    print("🔍 MNIST Classifier - Milestone 1 Verification")
    print("=" * 50)

    # Run all checks
    checks: list[tuple[str, bool]] = [
        ("Directory Structure", check_directories()),
        ("Git Setup", check_git()),
        ("Virtual Environment", check_virtual_env()),
        ("Dependencies", check_dependencies()),
        ("Documentation", check_documentation()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed: bool = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 1 is complete!")
        print("You're ready to move on to Milestone 2: Data Acquisition")
    else:
        print("\n⚠️  Some checks failed. Please complete all tasks before proceeding.")
        print("\nTips:")
        if not checks[2][1]:  # Virtual env check
            print("- Make sure to activate your virtual environment first")
        if not checks[3][1]:  # Dependencies check
            print("- Install missing packages with: uv sync")


if __name__ == "__main__":
    main()
