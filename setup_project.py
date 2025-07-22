#!/usr/bin/env python3
"""
Automated setup script for MNIST Classifier project.
Run this to quickly set up the entire project structure.
"""

import platform
import subprocess
import sys
from pathlib import Path


def create_directories() -> None:
    """Create project directory structure."""
    dirs: list[str] = ["src", "data", "notebooks", "docs", "static", "templates"]

    print("📁 Creating directory structure...")
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"  ✅ Created {dir_name}/")
        else:
            print(f"  i  {dir_name}/ already exists")


def create_gitignore() -> None:
    """Create .gitignore file with Python-specific rules."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
.venv/
venv/
ENV/
env/
.env

# IDEs
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/

# Data and models
data/mnist/
*.h5
*.keras
*.pkl

# Logs
*.log
logs/
"""

    print("\n📝 Creating .gitignore...")
    Path(".gitignore").write_text(gitignore_content)
    print("  ✅ .gitignore created")


def init_git() -> None:
    """Initialize git repository."""
    print("\n🔧 Initializing Git repository...")

    if Path(".git").exists():
        print("  i  Git repository already initialized")
        return

    try:
        subprocess.run(["git", "init"], check=True, capture_output=True)
        print("  ✅ Git repository initialized")

        # Make initial commit
        subprocess.run(["git", "add", ".gitignore"], check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit: Add .gitignore"],
            check=True,
            capture_output=True,
        )
        print("  ✅ Initial commit created")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Git error: {e}")
        print("  Please ensure git is installed")


def setup_virtual_env() -> bool:
    """Create virtual environment using uv.

    Returns:
        bool: True if setup was successful, False otherwise.
    """
    print("\n🐍 Setting up virtual environment...")

    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ uv is not installed")
        print("  Please install from: https://github.com/astral-sh/uv")
        return False

    # Create virtual environment
    if not Path(".venv").exists():
        subprocess.run(["uv", "venv"], check=True)
        print("  ✅ Virtual environment created")
    else:
        print("  i  Virtual environment already exists")

    return True


def get_activation_command() -> str:
    """Get the correct activation command for the current platform.

    Returns:
        str: The activation command for the current platform.
    """
    system = platform.system()

    if system == "Windows":
        return ".venv\\Scripts\\activate"
    return "source .venv/bin/activate"


def main() -> None:
    """Run all setup steps."""
    print("🚀 MNIST Classifier Project Setup")
    print("=" * 40)

    # Run setup steps
    create_directories()
    create_gitignore()
    init_git()

    if setup_virtual_env():
        print("\n✨ Setup complete!")
        print("\n📋 Next steps:")
        print(f"1. Activate virtual environment: {get_activation_command()}")
        print("2. Install dependencies: uv sync")
        print("3. Create README.md (if not already created)")
        print("4. Commit changes: git add . && git commit -m 'Complete project setup'")
    else:
        print("\n⚠️  Setup incomplete. Please install uv and run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
