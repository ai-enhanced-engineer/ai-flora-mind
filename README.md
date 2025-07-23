# AI Flora Mind - ML Engineering Assessment Solution

This is my solution to the MaintainX Applied Machine Learning Engineer assessment. Given the API requirements of the assessment, I identified that the system could be efficiently powered by a classification algorithm. What started as a simple classification task evolved into a comprehensive machine learning journey, resulting in a production-ready application backed by a family of classifiers: from elegant heuristics to sophisticated ensemble methods.

## What I Built

- **Comprehensive Exploratory Data Analysis (EDA)**
  - Interactive Jupyter notebook: [`research/eda/EDA.ipynb`](research/eda/EDA.ipynb)
  - Modeling strategy derived from insights: [`research/eda/MODELING_STRATEGY.md`](research/eda/MODELING_STRATEGY.md)
  - Key discoveries: perfect Setosa separation, bimodal petal distributions, 92% variance in 2 PCs
  - These insights directly informed the heuristic baseline achieving 96% accuracy
  - This enabled deployment of a working model in less than a day!

- **Complete research journey** with comprehensive documentation
  - Full narrative in [`research/EXPERIMENTS_JOURNEY.md`](research/EXPERIMENTS_JOURNEY.md)
  - Research structure and commands in [`research/README.md`](research/README.md)
  - Systematic progression from baseline to state-of-the-art
  - Rigorous validation with LOOCV, k-fold CV, and OOB
  - Discovery that simple models match complex ones

- **4 ML models** forming a complete classifier family:
  - Rule-based heuristic (96% accuracy in 3 lines!) - [`EXPERIMENT.md`](research/experiments/rule_based_heuristic/EXPERIMENT.md)
  - Decision tree (96.7% accuracy) - [`EXPERIMENT.md`](research/experiments/decision_tree/EXPERIMENT.md)
  - Random forest (96% accuracy) - [`EXPERIMENT.md`](research/experiments/random_forest/EXPERIMENT.md)
  - XGBoost (96% accuracy) - [`EXPERIMENT.md`](research/experiments/xgboost/EXPERIMENT.md)

- **Production-ready Application** with pluggable model architecture
  - Configuration-driven design using Pydantic models and dependency injection via factory pattern
  - Polymorphic predictor interface allows seamless model switching through environment variables
  - 97% test coverage with comprehensive test suite
  - Docker deployment with multi-stage builds
  - Health checks, validation, and OpenAPI documentation
  - Full architecture details in [`ai_flora_mind/README.md`](ai_flora_mind/README.md)
  - Integration guide for new models in [`CLAUDE.md#adding-new-predictors--complete-integration-process`](CLAUDE.md#adding-new-predictors---complete-integration-process)

- **Professional development workflow from day one**
  - CI/CD pipeline with GitHub Actions (v0.1.0)
  - 17 PRs with automated testing and semantic versioning
  - Industry-standard practices throughout development

## Quick Start (Get It Running in 2 Minutes)

### Option 1: Local Development
```bash
# 1. Setup environment (one-time setup)
make environment-create

# 2. Run linters and tests to ensure everything is working
make all-test-validate-branch

# 3. Run the API locally with the model of your choice
FLORA_CLASSIFIER_TYPE=heuristic make api-run       # Rule-based classifier (96% accuracy)
FLORA_CLASSIFIER_TYPE=decision_tree make api-run   # Decision tree (96.7% accuracy) - best performing!
FLORA_CLASSIFIER_TYPE=random_forest make api-run   # Random forest (96% accuracy)
FLORA_CLASSIFIER_TYPE=xgboost make api-run         # XGBoost (96% accuracy)

# 4. In another terminal, validate the API (runs inference on 50 random samples)
make api-validate
```

### Option 2: Docker Deployment (Production-Ready)
```bash
# Build and run with Docker using the model of your choice
FLORA_CLASSIFIER_TYPE=heuristic make service-start     # Rule-based (96% accuracy)
FLORA_CLASSIFIER_TYPE=decision_tree make service-start # Decision Tree (96.7% accuracy) - best performing!
FLORA_CLASSIFIER_TYPE=random_forest make service-start # Random Forest (96% accuracy)
FLORA_CLASSIFIER_TYPE=xgboost make service-start       # XGBoost (96% accuracy)

# In another terminal, validate the service (runs inference on 50 random samples)
make service-validate

# Stop the service when done
make service-stop
```

### Test the API
The same curl commands work for both local (api-run) and Docker (service-start) deployments:

```bash
# Test 1: Setosa example
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
# Expected: {"prediction": "setosa"}

# Test 2: Versicolor example
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}'
# Expected: {"prediction": "versicolor"}

# Test 3: Virginica example
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.3, "petal_length": 5.7, "petal_width": 2.1}'
# Expected: {"prediction": "virginica"}
```

## Project Structure

```
.
├── ai_flora_mind/      # Core application code (see ai_flora_mind/README.md)
├── research/           # ML experiments and findings (see research/README.md)
├── tests/              # Comprehensive test suite with 97% coverage
├── scripts/            # Utility scripts for validation and deployment
├── registry/           # Production model storage (see [registry/prd/README.md](registry/prd/README.md))
├── .github/            # CI/CD workflows and automation
├── Dockerfile          # Multi-stage production build
├── Makefile            # All commands you need
├── CLAUDE.md           # Comprehensive development guide
└── pyproject.toml      # Project dependencies and configuration
```

Each major directory contains its own README with detailed documentation.

## Product-First Development Approach

This project demonstrates a pragmatic, product-focused development workflow. The development was organized to deliver value incrementally through [17 pull requests](https://github.com/maintainx-take-home/leo-garcia-vargas/pulls?q=is%3Apr+is%3Aclosed) and [15 releases](https://github.com/maintainx-take-home/leo-garcia-vargas/releases):

### Key Milestones

**[v0.1.0](https://github.com/maintainx-take-home/leo-garcia-vargas/releases/tag/v0.1.0) - Project Foundation**
- Established project structure with CI/CD workflows
- Set up automated testing and release infrastructure

**[v0.3.0](https://github.com/maintainx-take-home/leo-garcia-vargas/releases/tag/v0.3.0) - Minimalistic API** (PR #3)
- Deployed containerized API with `/predict` endpoint
- Initially returned hardcoded responses to unblock integration teams
- No ML bottleneck for frontend/backend development

**[v0.4.0](https://github.com/maintainx-take-home/leo-garcia-vargas/releases/tag/v0.4.0) - Production Docker** (PR #4)
- Added production-grade Docker containerization
- Multi-stage builds for optimized images
- Environment-based configuration

**[v0.5.0](https://github.com/maintainx-take-home/leo-garcia-vargas/releases/tag/v0.5.0) - First Model Deployed** (PR #9)
- Operationalized heuristic classifier achieving 96% accuracy
- Production-ready solution with <1ms inference time
- Started collecting real-world data

**[v0.7.0](https://github.com/maintainx-take-home/leo-garcia-vargas/releases/tag/v0.7.0) - Decision Tree Integration** (PR #14)
- Added decision tree predictor (96.7% accuracy)
- Introduced pluggable model architecture
- Service management improvements

**[v0.8.0](https://github.com/maintainx-take-home/leo-garcia-vargas/releases/tag/v0.8.0) - Random Forest Deployment** (PR #13)
- Deployed random forest with production model registry
- Enhanced validation and testing infrastructure

**[v0.9.0](https://github.com/maintainx-take-home/leo-garcia-vargas/releases/tag/v0.9.0) - XGBoost & Full ML Suite** (PR #16)
- Complete family of 4 classifiers available
- A/B testing ready with environment variable switching
- Comprehensive research documentation

This incremental approach ensured teams were never blocked while we pursued model improvements.

## How I Enhanced the Requirements

While the assessment explicitly stated "We do not expect a production grade solution," I chose to build one anyway—this is an area where I excel, and with modern tooling and generative AI assistance, creating production-ready systems is both feasible and enjoyable within the given timeframe.

| Requirement | Basic Delivery | My Enhanced Implementation |
|-------------|----------------|---------------------------|
| **Train a model** | Single model with basic training | Family of 4 models with systematic progression from heuristic to XGBoost |
| **HTTP API** | Basic `/predict` endpoint | Production FastAPI with health checks, validation, and OpenAPI docs |
| **Model inference** | Simple prediction function | Pluggable predictor architecture with dependency injection |
| **Reasonable predictions** | Working classifier | 96%+ accuracy across all models with comprehensive validation |
| **Showcase areas of excellence** | Pick 1-2 areas | Demonstrated excellence in ML research, software engineering, and DevOps |
| **Support 1h discussion** | Basic implementation | Comprehensive research journey, architecture decisions, and performance analysis |


## Some project highlights

### ML Engineering
- **Comprehensive EDA** revealing perfect Setosa separation and bimodal petal distributions
- **Data-driven model selection** - EDA insights directly inspired the 3-line heuristic baseline
- **Systematic progression** from baseline to state-of-the-art models, validating EDA findings
- **Multiple validation strategies** (LOOCV, k-fold CV, OOB) for robust evaluation
- **Feature engineering** based on petal dominance discovered in EDA
- **Recognition of dataset's theoretical performance ceiling** (~96%)

### Software Engineering
- **Pluggable predictor architecture** - Abstract base class with polymorphic implementations, enabling runtime model switching via configuration
- **Dependency injection via factory pattern** - Centralized component registry decouples instantiation from usage, simplifying testing and extensibility
- **Configuration-driven design** - Pydantic models with validation, environment mapping, and hierarchical config support
- **97% test coverage** - Unit, functional, and integration tests with custom abstractions (no mocks), ensuring reliability
- **Structured error handling** - Graceful degradation with semantic logging, correlation IDs, and proper error propagation
- **Type safety throughout** - Full MyPy compliance with precise annotations, catching errors at development time

### Production Readiness
- **CI/CD pipeline from v0.1.0** with automated testing and releases
- **Docker containerization** with optimized multi-stage builds
- **Configuration management** via environment variables
- **Structured logging** with correlation IDs for observability
- **API documentation** with Swagger/OpenAPI

## Next Steps & Future Enhancements

1. **Ensemble Voting** - Combining models could achieve 98% accuracy
2. **Confidence Scoring** - Return prediction confidence for uncertainty quantification
3. **A/B Testing Infrastructure** - Deploy multiple models simultaneously
4. **Model Monitoring** - Track prediction distributions and detect drift
5. **Extended API** - Batch predictions, model explanations

## Additional Commands for Deep Dive

### Research & Experiments
```bash
# Reproduce ML experiments
make eval-heuristic                               # Evaluate rule-based baseline
make train-decision-tree-comprehensive            # Train with LOOCV validation
make train-random-forest-comprehensive            # Train with OOB + LOOCV
make train-xgboost-optimized                      # Train with heavy regularization
```

### Testing & Quality
```bash
# Different test suites
make unit-test                                    # Unit tests only
make functional-test                              # Functional tests only
make integration-test                             # Integration tests (requires API)
make all-test                                     # All tests with coverage report

# Code quality
make format                                       # Auto-format with Ruff
make lint                                         # Lint checks
make type-check                                   # MyPy type checking
```

---

## Original Assignment

### MaintainX - Take Home Assessment Applied Machine Learning Engineer

The goal of this take home assessment is to evaluate your machine learning knowledge and your software development skills.

### Task

Your task for this assessment is to train a model and expose it's inference function through an HTTP API. The choice of the model is yours.

The dataset to use is the [iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) from scikit-learn.

The focus will be on the depth and considerations taken for the project and your overall thought process. For the model itself, we will be checking that it works (makes reasonable predictions), but we won't evaluate its overall performance.
We do not expect a production grade solution, but we would like you to showcase the areas in which you excel.

You're welcome to explore any topic of interest, though the project should be substantial enough to support a 1h technical discussion upon submission. Here are some ideas to inspire you:

- Exploratory data analysis of the `iris` dataset
- Model training pipeline
- Comparing models
- Reasoning behind model choice / features
- Robust inference API
- Containerization
- Automated testing
- Project documentation

NOTE: You are free to use any tools to help you with this. Generative AI tools like Github Copilot (or similar) are not prohibited and are even recommended, but be sure to make good use of it (ie.: use in moderation where applicable) and review what it generated.

### Requirement

Your API will be need to follow theses requirements:

`HTTP POST /predict`

Request:

```json
{
  "sepal_length": 0.0, // float representing the sepal length in cm
  "sepal_width": 0.0, // float representing the sepal width in cm
  "petal_length": 0.0, // float representing the petal length in cm
  "petal_width": 0.0 // float representing the petal width in cm
}
```

Response

```json
{
  "prediction": "setosa" // The predicted iris type between "setosa", "versicolor" or "virginica"
}
```

### Testing

We have joined a postman collection and a bash script depending on your preference to test your API. Feel free to edit them to match your API's config (ex.: port, host, etc.).