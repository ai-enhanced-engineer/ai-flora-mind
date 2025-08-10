# ML Production Service

A reference implementation for taking machine learning models from research to production. This project demonstrates the complete journey from exploratory data analysis through model training to production-grade API deployment, showcasing best practices for building maintainable ML services.

## Why This Project?

Building production ML systems involves more than just training models. This project bridges the gap between data science experimentation and reliable production services by providing:

- **Complete ML Pipeline** - From exploratory data analysis through model selection to production deployment
- **Pluggable Architecture** - Swap between multiple ML algorithms via environment variables
- **Production-Ready** - Comprehensive error handling, structured logging, health checks, and Docker deployment
- **Research-Driven** - Systematic experimentation with multiple algorithms and validation strategies
- **Clean Architecture** - Configuration-driven design with dependency injection and polymorphic interfaces
- **Fully Tested** - 97% test coverage with unit, functional, and integration tests

## Use Cases

This reference implementation demonstrates patterns applicable to:

- **ML Service Development** - Build services that can hot-swap models for A/B testing
- **Model Deployment** - Deploy multiple algorithms side-by-side for comparison
- **Research to Production** - Transform notebook experiments into reliable APIs
- **Architecture Patterns** - Learn clean architecture principles for ML systems
- **Testing ML Code** - Comprehensive testing strategies for data science applications

## Architecture

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│   Client Apps   │────▶│ ML Production API   │────▶│  Model Registry  │
│   (REST API)    │     │    (FastAPI)        │     │  (4 algorithms)  │
└─────────────────┘     └──────────┬──────────┘     └──────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   ML Predictors     │
                        │  ┌───────────────┐  │
                        │  │  Heuristic    │  │ ← 96.0% accuracy
                        │  ├───────────────┤  │
                        │  │ Decision Tree │  │ ← 96.7% accuracy
                        │  ├───────────────┤  │
                        │  │ Random Forest │  │ ← 96.0% accuracy
                        │  ├───────────────┤  │
                        │  │   XGBoost     │  │ ← 96.0% accuracy
                        │  └───────────────┘  │
                        └─────────────────────┘
```

The service uses a polymorphic predictor interface with dependency injection, allowing runtime model selection through environment configuration. When a prediction request arrives, the API uses the configured model type to instantiate the appropriate predictor and return results. All models implement the same interface, enabling seamless switching between algorithms.

## Project Structure

```
ml-production-service/
├── ml_production_service/  # Core application (see ml_production_service/README.md)
│   ├── predictors/         # ML model implementations
│   ├── server/             # FastAPI application
│   └── factory.py          # Dependency injection
├── research/               # ML experiments (see research/README.md)
│   ├── eda/                # Exploratory data analysis
│   ├── experiments/        # Algorithm implementations
│   └── models/             # Trained model artifacts
├── tests/                  # Comprehensive test suite (97% coverage)
├── scripts/                # Validation and utility scripts
├── registry/               # Production model storage
├── .github/                # CI/CD workflows
├── Dockerfile              # Production container
├── Makefile                # Development commands
└── pyproject.toml          # Dependencies
```

Each major directory contains detailed documentation about its contents and purpose.

## Quick Start

Get the service running with your choice of ML algorithm in under 2 minutes:

### Prerequisites

- Python 3.10-3.12
- `uv` package manager (`pip install uv`) or standard `pip`
- Docker (optional, for containerized deployment)

### 1. Setup

Clone and set up your environment:

```bash
git clone https://github.com/yourusername/ml-production-service
cd ml-production-service
make environment-create
```

### 2. Choose Your Model

The service includes 4 production-ready models with different characteristics, demonstrated using the classic Iris dataset:

| Model | Accuracy | Inference Speed | Interpretability |
|-------|----------|-----------------|------------------|
| **Heuristic** | 96.0% | <1ms | High (simple rules) |
| **Decision Tree** | 96.7% | <1ms | High (tree viz) |
| **Random Forest** | 96.0% | ~2ms | Medium (feature importance) |
| **XGBoost** | 96.0% | ~3ms | Low (black box) |

### 3. Run Locally

Start the API with your chosen model:

```bash
# Using the heuristic model (fastest, most interpretable)
MPS_MODEL_TYPE=heuristic make api-run

# Using the decision tree (best accuracy)
MPS_MODEL_TYPE=decision_tree make api-run

# Using ensemble methods
MPS_MODEL_TYPE=random_forest make api-run
MPS_MODEL_TYPE=xgboost make api-run
```

The API will start at `http://localhost:8000` with interactive docs at `/docs`.

### 4. Test the API

```bash
# Predict iris species from measurements (example using iris dataset)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Response: {"prediction": "setosa"}
```

### 5. Docker Deployment

For production deployment with Docker:

```bash
# Build and start with your chosen model
MPS_MODEL_TYPE=decision_tree make service-start

# Validate the service
make service-validate

# Stop when done
make service-stop
```

## API Documentation

### Endpoints

#### `POST /predict`
Classify input based on measurements (demonstrated with iris species).

**Request:**
```json
{
  "sepal_length": 5.1,  // Feature 1
  "sepal_width": 3.5,   // Feature 2
  "petal_length": 1.4,  // Feature 3
  "petal_width": 0.2    // Feature 4
}
```

**Response:**
```json
{
  "prediction": "setosa"  // Classification result
}
```

#### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `MPS_MODEL_TYPE` | ML algorithm to use | `heuristic` | `heuristic`, `decision_tree`, `random_forest`, `xgboost` |
| `MPS_MODEL_PATH` | Path to model artifacts | `registry/prd` | Any valid path |
| `MPS_LOG_LEVEL` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MPS_API_HOST` | API bind address | `0.0.0.0` | Any valid host |
| `MPS_API_PORT` | API port | `8000` | Any valid port |

## Development

### Development Workflow

```bash
# Code quality
make format              # Auto-format with Ruff
make lint                # Lint checks
make type-check          # MyPy type checking
make validate-branch     # All checks before commit

# Testing
make unit-test           # Unit tests only
make functional-test     # Functional tests
make integration-test    # Integration tests
make all-test           # All tests with coverage

# Research experiments (using iris dataset as example)
make eval-heuristic      # Evaluate rule-based baseline
make train-decision-tree # Train decision tree variants
make train-random-forest # Train random forest variants
make train-xgboost      # Train XGBoost variants
```

### Adding New Models

The architecture makes it easy to add new ML algorithms:

1. Create predictor class in `ml_production_service/predictors/`
2. Implement the `BasePredictor` interface
3. Add model type to `ModelType` enum
4. Register in factory function
5. Add tests and documentation

See [CLAUDE.md](CLAUDE.md#adding-new-predictors--complete-integration-process) for detailed integration guide.

## ML Research

This project includes comprehensive ML research demonstrating the journey from baseline to state-of-the-art:

### Research Highlights

- **Comprehensive EDA** revealing class separation and feature relationships
- **Systematic experimentation** progressing from simple rules to ensemble methods
- **Multiple validation strategies** including LOOCV, k-fold CV, and OOB estimation
- **Production considerations** balancing accuracy, interpretability, and performance

### Key Findings

Using the Iris dataset as our demonstration:

1. **Simple models match complex ones** - The dataset has a ~96% performance ceiling
2. **Feature engineering helps** - Adding derived features improves tree-based models
3. **Ensemble methods plateau** - Diminishing returns beyond simple algorithms
4. **Heuristics are powerful** - Domain knowledge encoded as rules achieves 96% accuracy

Explore the complete research journey in [`research/EXPERIMENTS_JOURNEY.md`](research/EXPERIMENTS_JOURNEY.md).

## Deployment

### Docker

Production-ready multi-stage Dockerfile included:

```bash
# Build image
docker build -t ml-production-service .

# Run with model selection
docker run -p 8000:8000 \
  -e MPS_MODEL_TYPE=decision_tree \
  ml-production-service
```

### Cloud Platforms

Deploy to any container platform:

```bash
# Google Cloud Run
gcloud run deploy ml-production-service \
  --image gcr.io/your-project/ml-production-service \
  --set-env-vars MPS_MODEL_TYPE=decision_tree

# AWS ECS, Kubernetes, etc.
# Use platform-specific deployment configs
```

### Production Considerations

- Use environment-based configuration for model selection
- Implement monitoring on health check endpoint
- Set up structured logging aggregation
- Consider API gateway for rate limiting
- Deploy multiple instances with different models for A/B testing

## Testing

Comprehensive test coverage (97%) across multiple test types:

```bash
# Run all tests
make all-test

# Run specific test suites
make unit-test        # Test individual components
make functional-test  # Test complete workflows
make integration-test # Test with running API

# Validate deployment
make service-validate # Test deployed service
```

## Contributing

This project demonstrates best practices for ML service development. Contributions that enhance these patterns are welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Documentation

- [`ml_production_service/README.md`](ml_production_service/README.md) - Application architecture details
- [`research/README.md`](research/README.md) - ML research documentation
- [`CLAUDE.md`](CLAUDE.md) - Comprehensive development guide
- [`registry/prd/README.md`](registry/prd/README.md) - Model registry documentation

## License

Apache License 2.0 - see LICENSE file for details

---

This project serves as a reference implementation for building production ML services, demonstrating the complete journey from research notebooks to deployed APIs. Whether you're learning ML engineering practices or building your own services, this codebase provides patterns and examples for creating reliable, maintainable ML systems.