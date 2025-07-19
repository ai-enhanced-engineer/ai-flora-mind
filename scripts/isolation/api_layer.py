import argparse
import os

import uvicorn

# Set default environment variables for AI Flora Mind service
os.environ.setdefault("SERVICE_NAME", "ai_flora_mind")
os.environ.setdefault("SERVICE_VERSION", "0.1.0")
os.environ.setdefault("LOGGING_LEVEL", "INFO")
os.environ.setdefault("LOG_FORMAT", "json")
os.environ.setdefault("STREAM", "stdout")

# Ensure we're in development mode
os.environ.setdefault("ENVIRONMENT", "development")

if __name__ == "__main__":
    from ai_flora_mind.server.main import get_app

    parser = argparse.ArgumentParser(description="Run the AI Flora Mind API in isolation for testing.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the service on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the service on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Override logging level if provided
    if args.log_level:
        os.environ["LOGGING_LEVEL"] = args.log_level.upper()

    print(f"Starting AI Flora Mind API on {args.host}:{args.port}")
    print(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print(f"Logging level: {os.environ.get('LOGGING_LEVEL', 'INFO')}")
    print("API Documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")

    uvicorn.run(get_app(), host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
