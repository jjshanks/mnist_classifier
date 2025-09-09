#!/usr/bin/env python3
"""
Verification script for Milestone 6 completion.
"""

import os
import sys
from pathlib import Path


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return "✅" if condition else "❌"


def check_files_exist():
    """Check if all required files exist."""
    print("\n📁 Checking required files:")

    required_files = [
        ("src/mnist_classifier/models/cnn_model.py", "Updated model module"),
        ("src/mnist_classifier/visualization/__init__.py", "Visualization package"),
        (
            "src/mnist_classifier/visualization/activation_viz.py",
            "Activation visualization",
        ),
        ("src/mnist_classifier/api/app.py", "Updated API with visualizations"),
        ("static/js/visualizations.js", "Frontend visualization code"),
        ("static/css/visualizations.css", "Visualization styles"),
        ("templates/index.html", "Updated HTML template"),
    ]

    all_exist = True
    for filepath, description in required_files:
        exists = os.path.exists(filepath)
        print(f"  {check_mark(exists)} {description}: {filepath}")
        if not exists:
            all_exist = False

    return all_exist


def check_visualization_functions():
    """Check if visualization functions are implemented."""
    print("\n🔧 Checking visualization functions:")

    try:
        # Check model functions
        from src.mnist_classifier.models.cnn_model import (
            create_feature_map_grid,
            create_visualization_model,
            process_activations,
        )

        print(f"  {check_mark(True)} Model visualization functions imported")

        # Check visualization utilities
        from src.mnist_classifier.visualization.activation_viz import (
            create_activation_plot,
            create_layer_summary_plot,
            create_probability_chart,
        )

        print(f"  {check_mark(True)} Visualization utilities imported")

        return True

    except ImportError as e:
        print(f"  {check_mark(False)} Import error: {e}")
        return False


def check_api_endpoints():
    """Check if API has been updated for visualizations."""
    print("\n🌐 Checking API updates:")

    try:
        # Read API file
        api_path = Path("src/mnist_classifier/api/app.py")
        if not api_path.exists():
            print(f"  {check_mark(False)} API file not found")
            return False

        with open(api_path) as f:
            api_content = f.read()

        # Check for visualization-related code
        checks = [
            ("viz_model" in api_content, "Visualization model loading"),
            ("process_activations" in api_content, "Activation processing"),
            ("visualizations" in api_content, "Visualization response data"),
            ("/model-info" in api_content, "Model info endpoint"),
        ]

        all_good = True
        for condition, description in checks:
            print(f"  {check_mark(condition)} {description}")
            if not condition:
                all_good = False

        return all_good

    except Exception as e:
        print(f"  {check_mark(False)} Error checking API: {e}")
        return False


def check_frontend_integration():
    """Check if frontend has visualization components."""
    print("\n🎨 Checking frontend integration:")

    # Check HTML
    html_path = Path("templates/index.html")
    if html_path.exists():
        with open(html_path) as f:
            html_content = f.read()

        html_checks = [
            ("visualization-section" in html_content, "Visualization section in HTML"),
            ("viz-tabs" in html_content, "Layer tabs interface"),
            ("probability-section" in html_content, "Probability display"),
            ("education-section" in html_content, "Educational content"),
        ]

        for condition, description in html_checks:
            print(f"  {check_mark(condition)} {description}")
    else:
        print(f"  {check_mark(False)} HTML template not found")
        return False

    # Check JavaScript
    js_path = Path("static/js/visualizations.js")
    js_exists = js_path.exists()
    print(f"  {check_mark(js_exists)} Visualization JavaScript file")

    # Check CSS
    css_path = Path("static/css/visualizations.css")
    css_exists = css_path.exists()
    print(f"  {check_mark(css_exists)} Visualization styles")

    return all(condition for condition, _ in html_checks) and js_exists and css_exists


def check_educational_content():
    """Check if educational explanations are included."""
    print("\n📚 Checking educational content:")

    files_to_check = [
        ("templates/index.html", ["How It Works", "explanation", "layer-info"]),
        ("static/js/visualizations.js", ["explanation", "updateLayerInfo"]),
    ]

    all_good = True
    for filepath, keywords in files_to_check:
        if os.path.exists(filepath):
            with open(filepath) as f:
                content = f.read()

            has_content = any(keyword in content for keyword in keywords)
            print(f"  {check_mark(has_content)} Educational content in {filepath}")
            if not has_content:
                all_good = False
        else:
            print(f"  {check_mark(False)} {filepath} not found")
            all_good = False

    return all_good


def main():
    """Run all verification checks."""
    print("🔍 Milestone 6 Verification")
    print("=" * 50)

    # Make sure we're in the right directory
    if not os.path.exists("src/mnist_classifier"):
        print("❌ Error: Not in project root directory!")
        print("Please run this from the mnist_classifier project root.")
        sys.exit(1)

    # Add project root to path
    sys.path.insert(0, os.path.abspath("."))

    # Run all checks
    checks = [
        ("Required Files", check_files_exist()),
        ("Visualization Functions", check_visualization_functions()),
        ("API Updates", check_api_endpoints()),
        ("Frontend Integration", check_frontend_integration()),
        ("Educational Content", check_educational_content()),
    ]

    # Summary
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in checks:
        print(f"  {check_mark(passed)} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Congratulations! Milestone 6 is complete!")
        print("\nYou have successfully:")
        print("  ✓ Created visualization models to extract activations")
        print("  ✓ Updated backend to serve visualization data")
        print("  ✓ Built interactive frontend visualizations")
        print("  ✓ Added educational explanations")
        print("  ✓ Completed the entire MNIST classifier project!")
        print("\n🚀 Your application now features:")
        print("  - Real-time digit recognition")
        print("  - Neural network visualization")
        print("  - Educational explanations")
        print("  - Professional web interface")
        print("  - Command-line tools")
        print("\nRun the complete app: python run_web_app.py")
    else:
        print("\n⚠️  Some checks failed. Please review and fix the issues above.")
        print("\nTips:")
        print("  - Ensure all visualization files are created")
        print("  - Update the API to include visualization endpoints")
        print("  - Add visualization components to the frontend")
        print("  - Include educational explanations")


if __name__ == "__main__":
    main()
