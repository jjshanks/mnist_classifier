#!/usr/bin/env python3
"""
Comprehensive verification script for Milestone 5.
Ensures the web interface is properly implemented and functional.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def check_api_module():
    """Check if the API module exists and can be imported."""
    print("\n📦 Checking API module:")

    # Check if file exists
    api_path = Path("src") / "mnist_classifier" / "api" / "app.py"
    file_exists = api_path.exists()
    print(f"  {check_mark(file_exists)} app.py exists at {api_path}")

    # Try to import the module
    can_import = False
    has_app = False
    if file_exists:
        try:
            # Add src to Python path
            sys.path.insert(0, "src")
            from mnist_classifier.api import app as api_module

            can_import = True

            # Check if FastAPI app exists
            has_app = hasattr(api_module, "app")
        except Exception as e:
            print(f"  ❌ Import error: {e}")

    print(f"  {check_mark(can_import)} API module can be imported")
    print(f"  {check_mark(has_app)} FastAPI app instance exists")

    return file_exists and can_import and has_app


def check_templates():
    """Check if HTML templates exist."""
    print("\n📄 Checking templates:")

    templates = [
        ("templates/index.html", "Main page template"),
    ]

    all_exist = True
    for path, description in templates:
        exists = Path(path).exists()
        print(f"  {check_mark(exists)} {description}: {path}")
        if not exists:
            all_exist = False

    return all_exist


def check_static_files():
    """Check if static files exist."""
    print("\n🎨 Checking static files:")

    static_files = [
        ("static/css/style.css", "CSS styles"),
        ("static/js/main.js", "JavaScript code"),
    ]

    all_exist = True
    for path, description in static_files:
        exists = Path(path).exists()
        print(f"  {check_mark(exists)} {description}: {path}")
        if not exists:
            all_exist = False

    return all_exist


def check_model_exists():
    """Check if a trained model exists."""
    print("\n🧠 Checking for trained model:")

    model_paths = [
        "models/mnist_model",
        "models/mnist_cnn_model",
    ]

    model_found = False
    for path in model_paths:
        if Path(path).exists():
            print(f"  ✅ Found model at: {path}")
            model_found = True
            break

    if not model_found:
        print("  ❌ No trained model found")
        print("  💡 Run training first: python train_model.py")

    return model_found


def test_server_startup():
    """Test if the server can start."""
    print("\n🚀 Testing server startup:")

    try:
        # Start the server in a subprocess
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.mnist_classifier.api.app:app",
            "--host",
            "0.0.0.0",  # nosec B104
            "--port",
            "8000",
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Give the server time to start
        print("  ⏳ Starting server...")
        time.sleep(3)

        # Check if process is still running
        if process.poll() is None:
            print("  ✅ Server started successfully")

            # Try to access the root endpoint
            try:
                response = requests.get("http://localhost:8000/", timeout=5)
                print(f"  ✅ Root endpoint accessible (status: {response.status_code})")

                # Check API docs
                docs_response = requests.get("http://localhost:8000/docs", timeout=5)
                print(f"  ✅ API docs accessible (status: {docs_response.status_code})")

                server_ok = True
            except requests.RequestException as e:
                print(f"  ❌ Could not access endpoints: {e}")
                server_ok = False

            # Stop the server
            process.terminate()
            process.wait(timeout=5)
            print("  ✅ Server stopped cleanly")

            return server_ok

        # Process died
        stdout, stderr = process.communicate()
        print("  ❌ Server failed to start")
        if stderr:
            print(f"  Error: {stderr.strip()}")
        return False

    except Exception as e:
        print(f"  ❌ Error testing server: {e}")
        return False


def test_api_endpoints():
    """Test API endpoints if server is running."""
    print("\n🔌 Testing API endpoints:")

    # Check if server is already running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        server_running = response.status_code == 200
    except Exception:
        server_running = False

    if not server_running:
        print(
            "  ℹ  Server not running. Start with: "  # noqa: RUF001
            "uvicorn src.mnist_classifier.api.app:app --reload"
        )
        return False

    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        health_ok = response.status_code == 200
        print(f"  {check_mark(health_ok)} Health check endpoint")
    except Exception:
        health_ok = False
        print("  ❌ Health check endpoint failed")

    # Test predict endpoint with dummy data
    try:
        # Create a simple test image (28x28 white square)
        import base64
        import io

        import numpy as np
        from PIL import Image

        # Create white image
        img_array = np.ones((28, 28), dtype=np.uint8) * 255
        img = Image.fromarray(img_array)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Send prediction request
        response = requests.post(
            "http://localhost:8000/predict", json={"image": img_base64}, timeout=30
        )

        predict_ok = response.status_code == 200
        print(f"  {check_mark(predict_ok)} Prediction endpoint")

        if predict_ok:
            result = response.json()
            print(f"    Predicted digit: {result.get('predicted_digit', 'N/A')}")
            print(f"    Confidence: {result.get('confidence', 0):.2%}")
    except Exception as e:
        predict_ok = False
        print(f"  ❌ Prediction endpoint failed: {e}")

    return health_ok and predict_ok


def check_frontend_functionality():
    """Check if frontend files have required functionality."""
    print("\n🎯 Checking frontend functionality:")

    # Check JavaScript for required functions
    js_path = Path("static/js/main.js")
    if js_path.exists():
        with js_path.open() as f:
            js_content = f.read()

        required_functions = [
            ("clearCanvas", "Canvas clearing"),
            ("predict", "Prediction submission"),
            ("draw", "Mouse/touch drawing"),
        ]

        for func, desc in required_functions:
            has_func = func in js_content
            print(f"  {check_mark(has_func)} {desc}")
    else:
        print("  ❌ JavaScript file not found")
        return False

    # Check HTML for required elements
    html_path = Path("templates/index.html")
    if html_path.exists():
        with html_path.open() as f:
            html_content = f.read()

        required_elements = [
            ("<canvas", "Drawing canvas"),
            ("Clear", "Clear button"),
            ("Predict", "Predict button"),
        ]

        all_found = True
        for element, desc in required_elements:
            has_element = element in html_content
            print(f"  {check_mark(has_element)} {desc}")
            if not has_element:
                all_found = False

        return all_found

    print("  ❌ HTML template not found")
    return False


def main():
    """Run all verification checks."""
    print("🔍 MNIST Classifier - Milestone 5 Verification")
    print("=" * 50)

    # Check if in correct directory
    if not Path("src/mnist_classifier").exists():
        print("❌ Error: Not in project root directory")
        print("Please run from the mnist_classifier directory")
        sys.exit(1)

    # Run all checks
    checks = [
        ("API Module", check_api_module()),
        ("Templates", check_templates()),
        ("Static Files", check_static_files()),
        ("Trained Model", check_model_exists()),
        ("Frontend Functionality", check_frontend_functionality()),
        ("Server Startup", test_server_startup()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 5 is complete!")
        print("\n🚀 To run your web application:")
        print(
            "  1. Start the server: uvicorn src.mnist_classifier.api.app:app --reload"
        )
        print("  2. Open browser: http://localhost:8000")
        print("  3. Draw a digit and click Predict!")
        print("\nYou're ready for Milestone 6: Visualizing Neural Network State")
    else:
        print("\n⚠️  Some checks failed. Please complete all tasks before proceeding.")
        print("\n💡 Tips:")
        if not checks[0][1]:  # API module
            print("  - Ensure app.py is in src/mnist_classifier/api/")
            print("  - Check for import errors in your code")
        if not checks[3][1]:  # Model
            print("  - Train a model first: python train_model.py")
        if not checks[5][1]:  # Server
            print("  - Check for syntax errors in app.py")
            print("  - Ensure all dependencies are installed")


if __name__ == "__main__":
    main()
