#!/usr/bin/env python3
"""
Automated setup script for MNIST Classifier project.
Run this to quickly set up the entire project structure.
"""

import os
import sys
import subprocess
import platform

def create_directories():
    """Create project directory structure."""
    dirs = ['src', 'data', 'notebooks', 'docs', 'static', 'templates']
    
    print("📁 Creating directory structure...")
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  ✅ Created {dir_name}/")
        else:
            print(f"  ℹ️  {dir_name}/ already exists")

def create_gitignore():
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
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("  ✅ .gitignore created")

def init_git():
    """Initialize git repository."""
    print("\n🔧 Initializing Git repository...")
    
    if os.path.exists('.git'):
        print("  ℹ️  Git repository already initialized")
        return
    
    try:
        subprocess.run(['git', 'init'], check=True, capture_output=True)
        print("  ✅ Git repository initialized")
        
        # Make initial commit
        subprocess.run(['git', 'add', '.gitignore'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit: Add .gitignore'], 
                      check=True, capture_output=True)
        print("  ✅ Initial commit created")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Git error: {e}")
        print("  Please ensure git is installed")

def setup_virtual_env():
    """Create virtual environment using uv."""
    print("\n🐍 Setting up virtual environment...")
    
    # Check if uv is installed
    try:
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ uv is not installed")
        print("  Please install from: https://github.com/astral-sh/uv")
        return False
    
    # Create virtual environment
    if not os.path.exists('.venv'):
        subprocess.run(['uv', 'venv'], check=True)
        print("  ✅ Virtual environment created")
    else:
        print("  ℹ️  Virtual environment already exists")
    
    return True

def get_activation_command():
    """Get the correct activation command for the current platform."""
    system = platform.system()
    
    if system == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"

def main():
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
        print("4. Commit your changes: git add . && git commit -m 'Complete project setup'")
    else:
        print("\n⚠️  Setup incomplete. Please install uv and run again.")
        sys.exit(1)

if __name__ == "__main__":
    main()