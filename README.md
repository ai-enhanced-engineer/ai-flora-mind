# ML Production Service

A **production-grade machine learning service** demonstrating how to transform research notebooks into scalable, maintainable APIs. This is a living example of enterprise ML architecture patterns, not just another ML tutorial—it's a fully deployed service you can run, modify, and learn from.

## From Research to Production: Complete ML Pipeline

This project solves the critical gap between data science experimentation and production deployment by showing the **complete transformation process**:

**Research Phase** (`research/`) → **Production Service** (`ml_production_service/`)
- Jupyter notebooks with EDA → Structured logging and monitoring
- Experimental model training → Production model registry
- Ad-hoc validation → 97% test coverage with CI/CD
- Prototype code → Clean architecture with dependency injection

**Key Production Insight**: Our research revealed that simple heuristic models achieve 96% accuracy—matching complex XGBoost performance. This finding drives our production architecture principle: **start simple, add complexity only when data justifies it**.

## What You'll Learn: Enterprise ML Patterns

- **Research-to-Production Pipeline** - Transform notebook experiments into reliable APIs
- **Pluggable Architecture** - Hot-swap ML algorithms for A/B testing via environment configuration
- **Production-Grade Engineering** - Error handling, structured logging, health checks, Docker deployment
- **Model Lifecycle Management** - From research artifacts to production registry to live deployment
- **Testing ML Systems** - Comprehensive strategies for validating data science applications
- **Clean ML Architecture** - Configuration-driven design enabling rapid model iteration

## Real-World Application Patterns

This production service architecture applies to any feature-based classification problem:

- **ML Service Engineering** - Build services supporting model A/B testing and hot-swapping
- **Research Operationalization** - Transform notebook experiments into reliable, scalable APIs  
- **Enterprise ML Architecture** - Learn production patterns: dependency injection, factory patterns, configuration-driven design
- **Model Lifecycle Management** - Manage the complete journey from research to deployment
- **Production ML Testing** - Comprehensive validation strategies for data science applications

*Demonstrated using the classic Iris dataset for clear, reproducible examples that transfer to any classification domain.*

## Production-Grade Architecture

```
Research Pipeline          Production Service
┌─────────────────┐       ┌─────────────────────┐     ┌──────────────────┐
│ Jupyter Notebooks│ ───▶ │ ML Production API   │────▶│  Model Registry  │
│ (EDA + Training) │       │    (FastAPI)        │     │  (4 algorithms)  │
└─────────────────┘       └──────────┬──────────┘     └──────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   ML Predictors     │
                          │  ┌───────────────┐  │
                          │  │  Heuristic    │  │ ← 96.0% (simple rules)
                          │  ├───────────────┤  │
                          │  │ Decision Tree │  │ ← 96.7% (interpretable)
                          │  ├───────────────┤  │
                          │  │ Random Forest │  │ ← 96.0% (robust)
                          │  ├───────────────┤  │
                          │  │   XGBoost     │  │ ← 96.0% (complex)
                          │  └───────────────┘  │
                          └─────────────────────┘
```

**Enterprise Architecture Features:**
- **Polymorphic Interface**: All models implement `BasePredictor` for seamless runtime switching
- **Dependency Injection**: Factory pattern with environment-based configuration  
- **Hot Model Swapping**: Change algorithms via environment variables without code changes
- **Production Registry**: Research models promoted to production through systematic selection process
- **A/B Testing Ready**: Deploy multiple model variants simultaneously for comparison

## Research-to-Production Project Structure

This project demonstrates the **complete transformation** from research notebooks to production services:

```
ml-production-service/
├── research/               # 📊 RESEARCH PHASE
│   ├── eda/                #   Jupyter notebooks with exploratory data analysis
│   ├── experiments/        #   Algorithm implementations and training
│   ├── models/             #   Trained model artifacts with timestamps
│   └── results/            #   Performance metrics and evaluation reports
│
├── ml_production_service/  # 🚀 PRODUCTION PHASE  
│   ├── predictors/         #   Production model implementations from research
│   ├── server/             #   FastAPI application with monitoring
│   ├── factory.py          #   Dependency injection and model selection
│   └── configs.py          #   Production configuration management
│
├── registry/prd/           # 🏭 MODEL REGISTRY
│   └── *.joblib            #   Production models promoted from research/models/
│
├── tests/                  # ✅ PRODUCTION VALIDATION
│   ├── predictors/         #   Unit tests for each model implementation  
│   ├── test_api_layer.py   #   Integration tests for API endpoints
│   └── conftest.py         #   Test fixtures and utilities
│
├── scripts/validation/     # 🔍 QUALITY ASSURANCE
├── .github/workflows/      # ⚙️  CI/CD AUTOMATION
├── Dockerfile              # 📦 PRODUCTION DEPLOYMENT
└── Makefile                # 🛠️  DEVELOPMENT WORKFLOW
```

**Key Transformation Examples:**
- `research/eda/EDA.ipynb` insights → `ml_production_service/predictors/heuristic.py` rules
- `research/experiments/random_forest/` training → `registry/prd/random_forest.joblib` 
- Ad-hoc notebook validation → `tests/` with 97% coverage
- Manual model comparison → `factory.py` with hot-swapping capability

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

### 2. Production-Ready Model Options

The service includes 4 **production-grade models** with different characteristics and deployment strategies:

| Model | Accuracy | Speed | Interpretability | Production Use Case |
|-------|----------|-------|------------------|-------------------|
| **Heuristic** | 96.0% | <1ms | High (simple rules) | Real-time, explainable systems |
| **Decision Tree** | 96.7% | <1ms | High (tree viz) | Regulatory compliance, debugging |
| **Random Forest** | 96.0% | ~2ms | Medium (feature importance) | Robust production workloads |
| **XGBoost** | 96.0% | ~3ms | Low (black box) | High-volume inference |

*Performance metrics validated using the classic Iris dataset—patterns transfer to any feature-based classification problem.*

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
# Classify features into categories (demonstrated with iris species classification)
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
Classify input features into categories (demonstrated with iris species classification).

**Request Schema:**
```json
{
  "sepal_length": 5.1,  // Numeric feature 1
  "sepal_width": 3.5,   // Numeric feature 2  
  "petal_length": 1.4,  // Numeric feature 3
  "petal_width": 0.2    // Numeric feature 4
}
```

**Response Schema:**
```json
{
  "prediction": "setosa"  // Classification category
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

# Research experiments (demonstrated with iris classification)
make eval-heuristic      # Evaluate rule-based baseline approach
make train-decision-tree # Train decision tree with feature engineering
make train-random-forest # Train random forest with regularization
make train-xgboost      # Train XGBoost with hyperparameter optimization
```

### Adding New Models

The architecture makes it easy to add new ML algorithms:

1. Create predictor class in `ml_production_service/predictors/`
2. Implement the `BasePredictor` interface
3. Add model type to `ModelType` enum
4. Register in factory function
5. Add tests and documentation

See [CLAUDE.md](CLAUDE.md#adding-new-predictors--complete-integration-process) for detailed integration guide.

## Research-to-Production Methodology

This project demonstrates **systematic experimentation methodology** for production model selection—a complete research pipeline you can apply to any classification problem:

### Research Process Architecture

- **Comprehensive EDA** → Feature understanding and baseline rule development  
- **Systematic experimentation** → Progressive complexity from heuristics to ensemble methods
- **Multiple validation strategies** → LOOCV, k-fold CV, and OOB estimation for robust evaluation
- **Production-driven evaluation** → Balance accuracy, interpretability, and performance requirements

### Critical Production Insights

Research findings that directly informed our production architecture decisions:

1. **Start Simple Principle** - Simple models matched complex performance (96% ceiling), validating heuristic-first deployment
2. **Feature Engineering Impact** - Derived features improved tree models, guiding our feature pipeline design
3. **Complexity ROI Analysis** - Ensemble methods showed diminishing returns, informing model selection strategy
4. **Domain Knowledge Value** - Rule-based heuristics achieved 96% accuracy, proving expert knowledge capture value

**Production Impact**: These findings justify our architecture's heuristic-first approach and environment-driven model selection capability.

### Complete Research Documentation
- **Methodology**: [`research/EXPERIMENTS_JOURNEY.md`](research/EXPERIMENTS_JOURNEY.md) - Complete experimental evolution
- **Individual Experiments**: `research/experiments/*/EXPERIMENT.md` - Detailed algorithm analysis
- **EDA Insights**: `research/eda/EDA.ipynb` - Data exploration driving production rules

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

### Production Deployment Patterns

This service demonstrates enterprise-ready deployment patterns:

- **Environment-Based Configuration**: Model selection via `MPS_MODEL_TYPE` environment variable
- **Health Check Monitoring**: `/health` endpoint for container orchestration and load balancers
- **Structured Logging**: JSON-formatted logs for centralized aggregation and analysis  
- **A/B Testing Architecture**: Deploy multiple instances with different models simultaneously
- **Container Orchestration**: Kubernetes/Docker Swarm ready with proper resource limits
- **API Gateway Integration**: Rate limiting, authentication, and traffic management ready

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

## Contributing to Production ML Patterns

This project showcases enterprise ML service architecture—contributions that strengthen production patterns are welcome:

**Focus Areas for Contributions:**
- Additional model implementations following the `BasePredictor` interface
- Enhanced monitoring and observability features  
- Deployment automation and Infrastructure as Code examples
- Performance optimization and scalability improvements
- Security enhancements and best practices

**Contribution Process:**
1. Fork the repository
2. Create a feature branch focusing on production enhancement
3. Add comprehensive tests maintaining 97% coverage
4. Ensure all validation steps pass (`make validate-branch`)
5. Submit a pull request with clear production impact description

## Documentation

- [`ml_production_service/README.md`](ml_production_service/README.md) - Application architecture details
- [`research/README.md`](research/README.md) - ML research documentation
- [`CLAUDE.md`](CLAUDE.md) - Comprehensive development guide
- [`registry/prd/README.md`](registry/prd/README.md) - Model registry documentation

## License

Apache License 2.0 - see LICENSE file for details

---

## Production ML Service: A Living Example

This **production-grade ML service** demonstrates the complete transformation from research experimentation to enterprise deployment. Unlike typical ML tutorials, this is a fully functional service showcasing real production architecture patterns that scale.

**Use this codebase to:**
- **Learn enterprise ML patterns** through working, tested code
- **Understand research-to-production pipelines** via concrete examples  
- **Accelerate your ML service development** using proven architecture patterns
- **Reference production-ready implementations** when building similar systems

The service runs live, scales horizontally, handles errors gracefully, and supports A/B testing—everything you need to move from prototype to production.