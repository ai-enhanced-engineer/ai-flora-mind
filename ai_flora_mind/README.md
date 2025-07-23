# AI Flora Mind - Production Application

This directory contains the production-ready iris classification API with enterprise-grade features and clean architecture principles.

[← Back to main README](../README.md)

## Architecture Overview

### Design Principles

The application follows a **configuration-driven, polymorphic architecture** with clear separation of concerns. This design pattern uses dependency injection with a factory to instantiate the appropriate ML model based on environment configuration. The FastAPI layer remains agnostic to which specific model it uses - it simply calls a common interface that all models implement. This architecture enables seamless model switching through a single environment variable (`FLORA_CLASSIFIER_TYPE`), making it trivial to deploy different models or conduct A/B testing by deploying different services with unique algorithms. The diagram below shows how components interact: configuration flows from the environment to the API layer, which uses a factory to create the right predictor based on the model registry, with all predictors implementing the same interface:

```
+---------------------+     +------------------+
|   FastAPI Layer     |---->|  Configuration   |
|   (server/main.py)  |     |  (configs.py)    |
+---------------------+     +------------------+
           |                           |
           v                           v
+---------------------+     +------------------+
|  Dependency Factory |<----|  Model Registry  |
|   (factory.py)      |     |  (ModelType)     |
+---------------------+     +------------------+
           |
           v
+---------------------+
| Predictor Interface |
|   (base.py)         |
+---------------------+
           |
    +------+------+--------+--------+
    v             v        v        v
+----------+  +----------+ +----------+ +----------+
|Heuristic |  |Dec Tree  | |Random    | |XGBoost   |
|Predictor |  |Predictor | |Forest    | |Predictor |
+----------+  +----------+ +----------+ +----------+
```

### Key Components

1. **API Layer** (`server/main.py`)
   - FastAPI application with automatic OpenAPI documentation (`make api-docs` opens Swagger UI)
   - Request/response validation with Pydantic schemas
   - Comprehensive error handling and structured logging
   - Async context manager for lifecycle management
   - Health check endpoint for monitoring

2. **Configuration Management** (`configs.py`)
   - Environment-based configuration using Pydantic
   - Type-safe settings with validation
   - Model type enumeration for available classifiers
   - Dynamic model path resolution based on deployment context

3. **Dependency Injection** (`factory.py`)
   - Factory pattern for predictor instantiation
   - Runtime model selection via environment variables
   - Consistent interface across all model types
   - Proper error handling for missing models

4. **Predictor Interface** (`predictors/base.py`)
   - Abstract base class ensuring consistent API
   - Polymorphic design allowing seamless model switching
   - Type-safe prediction interface

5. **Model Implementations** (`predictors/`)
   - `heuristic.py`: Rule-based classifier (96% accuracy from [research experiments](../research/experiments/rule_based_heuristic/EXPERIMENT.md))
   - `decision_tree.py`: Scikit-learn decision tree (96.7% accuracy from [research experiments](../research/experiments/decision_tree/EXPERIMENT.md))
   - `random_forest.py`: Ensemble method (96% accuracy from [research experiments](../research/experiments/random_forest/EXPERIMENT.md))
   - `xgboost.py`: Gradient boosting (96% accuracy from [research experiments](../research/experiments/xgboost/EXPERIMENT.md))

6. **Supporting Components**
   - `logging.py`: Structured logging configuration with structlog
   - `features.py`: Feature engineering utilities for model preprocessing
   - `server/schemas.py`: Pydantic models for API request/response validation
   - `server/gunicorn_config.py`: Production WSGI server configuration

## Production Features

### 1. Configuration-Driven Architecture
```bash
# Switch models without code changes
FLORA_CLASSIFIER_TYPE=decision_tree make api-run  # Local development
FLORA_CLASSIFIER_TYPE=decision_tree make service-start  # Docker deployment
```

### 2. Comprehensive Error Handling
- Graceful degradation for missing models
- Detailed error messages for debugging
- Proper HTTP status codes
- Structured error responses

### 3. Observability
```python
# Structured logging throughout
logger.info("prediction_made", 
    model_type="decision_tree",
    prediction="setosa",
    inference_time_ms=1.2
)

# Request correlation IDs
# Performance metrics
# Health check endpoint: GET /health
```

### 4. API Documentation
- Automatic OpenAPI/Swagger generation
- Interactive API explorer at `/docs`
- ReDoc alternative at `/redoc`
- Example requests and responses

### 5. Type Safety
- Full MyPy compliance
- Pydantic models for all I/O
- Runtime validation
- Clear type annotations throughout

## Usage Examples

See the [main README's Quick Start section](../README.md#quick-start-get-it-running-in-2-minutes) for comprehensive usage examples including:
- Local development with `make api-run`
- Docker deployment with `make service-start`
- API testing with curl commands
- Model switching via `FLORA_CLASSIFIER_TYPE`

For health checks: `curl http://localhost:8000/health` returns `{"status": "healthy"}`.

## Docker Deployment

The application includes a production-ready Dockerfile:

```dockerfile
# Multi-stage build for minimal image size
# Non-root user for security
# Health checks included
# Optimized layer caching
```

Build and run:
```bash
docker build -t ai-flora-mind .
docker run -p 8000:8000 -e FLORA_CLASSIFIER_TYPE=decision_tree ai-flora-mind
```

## Performance Characteristics

| Model | Accuracy (from research) |
|-------|--------------------------|
| Heuristic | 96.0% |
| Decision Tree | 96.7% |
| Random Forest | 96.0% |
| XGBoost | 96.0% |

*Note: Inference times are sub-millisecond for heuristic and tree-based models, with ensemble methods taking slightly longer due to multiple tree evaluations.*

For detailed performance analysis and validation results, see the [research findings summary](../research/README.md#key-findings-summary).

## Security Considerations

- Input validation on all endpoints
- No direct file system access from API
- Environment-based secrets management
- CORS configuration for web security
- Rate limiting ready (configure reverse proxy)

## Monitoring and Operations

- Health endpoint for load balancer integration
- Structured JSON logging for log aggregation
- Graceful shutdown handling
- Docker health checks
- Prometheus metrics ready (add middleware)

## Next Steps

See the [main README's Next Steps section](../README.md#next-steps--future-enhancements) for future enhancements and ideas for extending the system.

---

This production application demonstrates enterprise-grade software engineering practices while maintaining simplicity and extensibility. The clean architecture ensures that business logic (model predictions) remains decoupled from infrastructure concerns (API, configuration), making the system maintainable and testable.