#!/usr/bin/env python3
"""
Run the complete MNIST classifier web application with visualizations.
"""

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_requirements():
    """Check if all requirements are met."""
    # Check for trained model
    model_paths = list(Path("models/experiments").glob("**/*.h5"))
    if not model_paths:
        print("⚠️  No trained model found!")
        print("Please train a model first: python train_model.py")
        return False

    print(f"✅ Found trained model: {model_paths[0].name}")
    return True


def main():
    """Main entry point for web app launcher."""
    parser = argparse.ArgumentParser(
        description="Launch MNIST Neural Network Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run in development mode
  %(prog)s --port 8080        # Run on different port
  %(prog)s --production       # Run in production mode
  %(prog)s --host 0.0.0.0     # Allow external connections
  %(prog)s --no-browser       # Don't open browser automatically

After starting, open http://localhost:8000 in your browser.
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Run in production mode (no auto-reload)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (production only)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    print("🚀 Starting MNIST Neural Network Visualizer")
    print("=" * 50)

    if not check_requirements():
        sys.exit(1)

    # Import uvicorn
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed")
        print("Install with: pip install uvicorn[standard]")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════╗
║     MNIST Neural Network Visualizer          ║
╠══════════════════════════════════════════════╣
║  Host: {args.host:<37} ║
║  Port: {args.port:<37} ║
║  Mode: {'Production' if args.production else 'Development':<37} ║
╚══════════════════════════════════════════════╝
    """)

    # Configure uvicorn
    config = {
        "app": "src.mnist_classifier.api.app:app",
        "host": args.host,
        "port": args.port,
        "log_level": "info",
    }

    if args.production:
        config.update(
            {
                "workers": args.workers,
                "reload": False,
            }
        )
        print("\n🚀 Starting in PRODUCTION mode...")
    else:
        config.update(
            {
                "reload": True,
                "reload_dirs": ["src", "templates", "static"],
            }
        )
        print("\n🔧 Starting in DEVELOPMENT mode (auto-reload enabled)...")

    print(f"\n📍 Access the app at: http://{args.host}:{args.port}")
    print(f"📍 API documentation at: http://{args.host}:{args.port}/docs")
    print("\nPress CTRL+C to stop the server\n")

    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # Run the server
    try:
        if not args.production and not args.no_browser:
            # Open browser after a short delay
            def open_browser():
                time.sleep(3)
                print("\n🌐 Opening browser...")
                webbrowser.open(f"http://{args.host}:{args.port}")

            import threading

            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()

        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        print("✅ Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
