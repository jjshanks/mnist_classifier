#!/usr/bin/env python3
"""
Test script for the MNIST Web API.

This script tests various API endpoints and functionality.
"""

import base64
import io
import json
import sys
import time

import requests
from PIL import Image, ImageDraw


def create_test_digit_image(digit=7):
    """
    Create a test image with a digit.

    Returns:
        Base64 encoded image string
    """
    # Create image
    img = Image.new("RGBA", (280, 280), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw a simple digit (7)
    if digit == 7:
        draw.line([(70, 70), (210, 70)], fill=(0, 0, 0), width=20)
        draw.line([(210, 70), (140, 210)], fill=(0, 0, 0), width=20)
    elif digit == 1:
        draw.line([(140, 70), (140, 210)], fill=(0, 0, 0), width=20)

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def test_health_endpoint(base_url):
    """Test the health check endpoint."""
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_api_info_endpoint(base_url):
    """Test the API info endpoint."""
    print("\n2. Testing /api/info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/info", timeout=10)
        print(f"   Status Code: {response.status_code}")
        data = response.json()
        print(f"   API Name: {data.get('name')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Model Loaded: {data.get('model', {}).get('loaded')}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_prediction_endpoint(base_url):
    """Test the prediction endpoint."""
    print("\n3. Testing /predict endpoint...")

    # Create test image
    print("   Creating test image...")
    test_image = create_test_digit_image(7)

    # Send prediction request
    print("   Sending prediction request...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json={"image": test_image},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        elapsed_time = (time.time() - start_time) * 1000

        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed_time:.1f}ms")

        if response.status_code == 200:
            data = response.json()
            print(f"   Predicted Digit: {data['predicted_digit']}")
            print(f"   Confidence: {data['confidence']:.1%}")
            print(f"   Processing Time: {data['processing_time']:.1f}ms")

            # Show top 3 probabilities
            probs = data["probabilities"]
            sorted_probs = sorted(
                probs.items(), key=lambda x: float(x[1]), reverse=True
            )[:3]
            print("   Top 3 predictions:")
            for digit, prob in sorted_probs:
                print(f"     Digit {digit}: {float(prob):.1%}")

            return True

        print(f"   ❌ Error Response: {response.text}")
        return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_main_page(base_url):
    """Test the main web page."""
    print("\n4. Testing main page...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Content Type: {response.headers.get('content-type')}")
        print(f"   Page Title Found: {'MNIST Digit Classifier' in response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_static_files(base_url):
    """Test static file serving."""
    print("\n5. Testing static files...")

    static_files = ["/static/css/style.css", "/static/js/main.js"]

    all_good = True
    for file_path in static_files:
        try:
            response = requests.get(f"{base_url}{file_path}", timeout=10)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"   {status} {file_path} - Status: {response.status_code}")
            if response.status_code != 200:
                all_good = False
        except Exception as e:
            print(f"   ❌ {file_path} - Error: {e}")
            all_good = False

    return all_good


def main():
    """Run all API tests."""
    # Default API URL
    base_url = "http://localhost:8000"

    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip("/")

    print("🧪 Testing MNIST Web API")
    print(f"📍 Base URL: {base_url}")
    print("=" * 50)

    # Check if server is running
    try:
        requests.get(base_url, timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to server")
        print(f"   Make sure the server is running at {base_url}")
        print("   Start with: python run_web_app.py")
        sys.exit(1)

    # Run tests
    tests = [
        test_health_endpoint(base_url),
        test_api_info_endpoint(base_url),
        test_prediction_endpoint(base_url),
        test_main_page(base_url),
        test_static_files(base_url),
    ]

    # Summary
    passed = sum(tests)
    total = len(tests)

    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed! The web API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
