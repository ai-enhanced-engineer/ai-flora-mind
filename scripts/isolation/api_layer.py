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
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["heuristic", "random_forest", "decision_tree", "xgboost"],
        help="Model type to use for predictions (default: uses environment variable or heuristic)",
    )

    args = parser.parse_args()

    # Override logging level if provided
    if args.log_level:
        os.environ["LOGGING_LEVEL"] = args.log_level.upper()

    # Override model configuration if provided
    if args.model_type:
        os.environ["FLORA_CLASSIFIER_TYPE"] = args.model_type
        print(f"Model type set to: {args.model_type}")

    print(f"Starting AI Flora Mind API on {args.host}:{args.port}")
    print(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print(f"Logging level: {os.environ.get('LOGGING_LEVEL', 'INFO')}")
    print(f"Model type: {os.environ.get('FLORA_CLASSIFIER_TYPE', 'heuristic')}")
    print("API Documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("Example prediction test:")
    print(
        '  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" '
        '-d \'{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}\''
    )

    uvicorn.run(
        "ai_flora_mind.server.main:get_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
