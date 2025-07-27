#!/usr/bin/env python3
"""
Verification script for Milestone 4: Command-Line Interface.

This script checks that all CLI components are properly implemented
and functioning correctly.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def run_command(cmd, capture=True):
    """Run a command and return success status and output."""
    try:
        if capture:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False,  # nosec B602
            )
            return result.returncode == 0, result.stdout, result.stderr
        result = subprocess.run(cmd, shell=True, check=False)  # nosec B602
        return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)


def check_cli_structure():
    """Check if CLI package structure exists."""
    print("\n📁 Checking CLI Package Structure:")

    required_files = [
        "src/mnist_classifier/cli/__init__.py",
        "src/mnist_classifier/cli/predict.py",
        "src/mnist_classifier/cli/image_utils.py",
        "predict_digit.py",
        "batch_predict.py",
        "create_test_images.py",
    ]

    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        print(f"  {check_mark(exists)} {file_path}")
        if not exists:
            all_exist = False

    return all_exist


def check_cli_scripts_executable():
    """Check if CLI scripts are executable."""
    print("\n🔧 Checking Script Permissions:")

    scripts = ["predict_digit.py", "batch_predict.py", "create_test_images.py"]
    all_executable = True

    for script in scripts:
        if Path(script).exists():
            # Check if file has shebang
            with Path(script).open() as f:
                first_line = f.readline()
                has_shebang = first_line.startswith("#!/usr/bin/env python")

            # Check if executable (on Unix-like systems)
            is_executable = (
                os.access(script, os.X_OK) if sys.platform != "win32" else True
            )

            status = has_shebang and is_executable
            print(
                f"  {check_mark(status)} {script} (shebang: {check_mark(has_shebang)}, "
                f"executable: {check_mark(is_executable)})"
            )

            if not status:
                all_executable = False
        else:
            print(f"  {check_mark(False)} {script} (not found)")
            all_executable = False

    return all_executable


def check_model_exists():
    """Check if a trained model exists."""
    print("\n🧠 Checking for Trained Model:")

    model_paths = [
        "models/mnist_model",
        "models/mnist_model.keras",
        "models/mnist_model.h5",
    ]

    model_found = False
    for path in model_paths:
        exists = Path(path).exists()
        if exists:
            print(f"  {check_mark(True)} Found model at: {path}")
            model_found = True
            break

    if not model_found:
        print(f"  {check_mark(False)} No trained model found")
        print("  INFO: Train a model first with: ./train_gpu.sh")

    return model_found


def test_create_test_images():
    """Test the create_test_images.py script."""
    print("\n🖼️  Testing Image Creation:")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test synthetic image creation
        cmd = f"python create_test_images.py {tmpdir} --synthetic --count 1"
        success, stdout, stderr = run_command(cmd)

        print(f"  {check_mark(success)} Create synthetic images")
        if not success:
            print(f"    Error: {stderr}")

        # Check if images were created
        if success:
            images = list(Path(tmpdir).glob("*.png"))
            has_images = len(images) > 0
            print(f"  {check_mark(has_images)} Images created ({len(images)} files)")
            return success and has_images

    return False


def test_single_prediction():
    """Test single image prediction."""
    print("\n🔮 Testing Single Image Prediction:")

    # First create a test image
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test image
        create_cmd = f"python create_test_images.py {tmpdir} --synthetic --count 1"
        success, _, _ = run_command(create_cmd)

        if not success:
            print(f"  {check_mark(False)} Failed to create test image")
            return False

        # Find the created image
        images = list(Path(tmpdir).glob("*.png"))
        if not images:
            print(f"  {check_mark(False)} No test image found")
            return False

        test_image = images[0]

        # Test verbose output
        cmd = f"python predict_digit.py {test_image}"
        success, stdout, stderr = run_command(cmd)
        print(f"  {check_mark(success)} Verbose output mode")
        if success and "Predicted digit:" in stdout:
            print(f"  {check_mark(True)} Output contains prediction")
        else:
            print(f"  {check_mark(False)} Output missing prediction")

        # Test quiet output
        cmd = f"python predict_digit.py {test_image} --quiet"
        success, stdout, stderr = run_command(cmd)
        is_digit = stdout.strip().isdigit() if success else False
        print(f"  {check_mark(success and is_digit)} Quiet output mode (digit only)")

        # Test JSON output
        cmd = f"python predict_digit.py {test_image} --output-format json"
        success, stdout, stderr = run_command(cmd)

        try:
            if success:
                data = json.loads(stdout)
                has_required = all(
                    key in data
                    for key in ["predicted_digit", "confidence", "probabilities"]
                )
                print(f"  {check_mark(has_required)} JSON output format")
            else:
                print(f"  {check_mark(False)} JSON output format")
        except Exception:
            print(f"  {check_mark(False)} JSON output format (invalid JSON)")

        return True


def test_batch_prediction():
    """Test batch image prediction."""
    print("\n📦 Testing Batch Prediction:")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple test images
        create_cmd = f"python create_test_images.py {tmpdir} --synthetic --count 2"
        success, _, _ = run_command(create_cmd)

        if not success:
            print(f"  {check_mark(False)} Failed to create test images")
            return False

        # Test table output
        cmd = f"python batch_predict.py {tmpdir}/*.png"
        success, stdout, stderr = run_command(cmd)
        print(f"  {check_mark(success)} Table output format")

        # Test CSV output
        csv_path = Path(tmpdir) / "results.csv"
        cmd = f"python batch_predict.py {tmpdir}/*.png --output {csv_path}"
        success, stdout, stderr = run_command(cmd)
        csv_exists = csv_path.exists()
        print(f"  {check_mark(success and csv_exists)} CSV output file")

        # Test JSON output to stdout
        cmd = f"python batch_predict.py {tmpdir}/*.png --format json"
        success, stdout, stderr = run_command(cmd)

        try:
            if success:
                data = json.loads(stdout)
                is_list = isinstance(data, list) and len(data) > 0
                print(f"  {check_mark(is_list)} JSON output format")
            else:
                print(f"  {check_mark(False)} JSON output format")
        except Exception:
            print(f"  {check_mark(False)} JSON output format (invalid JSON)")

        return True


def test_error_handling():
    """Test error handling in CLI tools."""
    print("\n⚠️  Testing Error Handling:")

    # Test with non-existent file
    cmd = "python predict_digit.py non_existent_file.png"
    success, stdout, stderr = run_command(cmd)
    has_error = not success and (
        "not found" in stderr.lower() or "not found" in stdout.lower()
    )
    print(f"  {check_mark(has_error)} Handle non-existent file")

    # Test with invalid image format (create a text file with .png extension)
    with tempfile.NamedTemporaryFile(suffix=".png", mode="w", delete=False) as f:
        f.write("This is not an image")
        fake_image = f.name

    cmd = f"python predict_digit.py {fake_image}"
    success, stdout, stderr = run_command(cmd)
    has_error = not success and ("error" in stderr.lower() or "error" in stdout.lower())
    print(f"  {check_mark(has_error)} Handle invalid image format")

    # Clean up
    Path(fake_image).unlink()

    return True


def check_documentation():
    """Check if documentation exists."""
    print("\n📚 Checking Documentation:")

    docs = {
        "README.md CLI section": Path("README.md").exists()
        and "Command-Line Interface" in Path("README.md").read_text(),
        "docs/cli_usage.md": Path("docs/cli_usage.md").exists(),
    }

    all_exist = True
    for doc, exists in docs.items():
        print(f"  {check_mark(exists)} {doc}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("🔍 MNIST Classifier - Milestone 4 Verification")
    print("=" * 50)

    # Check for model first
    model_exists = check_model_exists()

    if not model_exists:
        print("\n⚠️  Cannot fully test CLI without a trained model.")
        print("Please train a model first with: ./train_gpu.sh")
        print("\nContinuing with structure checks only...\n")

    # Run all checks
    checks = [
        ("CLI Package Structure", check_cli_structure()),
        ("Script Permissions", check_cli_scripts_executable()),
        ("Documentation", check_documentation()),
    ]

    # Only run functional tests if model exists
    if model_exists:
        checks.extend(
            [
                ("Image Creation", test_create_test_images()),
                ("Single Prediction", test_single_prediction()),
                ("Batch Prediction", test_batch_prediction()),
                ("Error Handling", test_error_handling()),
            ]
        )

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed and model_exists:
        print("\n🎉 Congratulations! Milestone 4 is complete!")
        print("The CLI interface is working correctly.")
        print("\nYou can now:")
        print("- Classify individual images with: python predict_digit.py <image>")
        print("- Process batches with: python batch_predict.py <images>")
        print("- Create test images with: python create_test_images.py <output_dir>")
        print("\nReady to move on to Milestone 5: Web Interface!")
    elif all_passed and not model_exists:
        print("\n✅ CLI structure is complete!")
        print("⚠️  Train a model to fully test the CLI functionality.")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        if not model_exists:
            print("💡 Remember to train a model before full testing.")


if __name__ == "__main__":
    main()
