#!/usr/bin/env python3
"""
Launch the MNIST Digit Classifier web application.

This script starts the FastAPI server with appropriate settings
for development or production.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for web app launcher."""
    parser = argparse.ArgumentParser(
        description="Launch MNIST Digit Classifier Web App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run in development mode
  %(prog)s --port 8080        # Run on different port
  %(prog)s --production       # Run in production mode
  %(prog)s --host 0.0.0.0     # Allow external connections

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

    args = parser.parse_args()

    # Import uvicorn
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed")
        print("Install with: pip install uvicorn[standard]")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════╗
║     MNIST Digit Classifier Web App           ║
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

    # Run the server
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
