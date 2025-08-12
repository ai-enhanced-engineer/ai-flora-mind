# ML Production Service

**Production-Grade Machine Learning Service** - A reference implementation demonstrating the complete transformation from research notebooks to scalable, maintainable production APIs with multiple model deployment strategies.

📚 **Part of [AI Enhanced Engineer](https://aienhancedengineer.substack.com/)** - Exploring production patterns for ML systems at scale.

## 🏗️ The Research-to-Production Pipeline

In ["Hidden Technical Debt in Machine Learning Systems"](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)<sup>[1](#ref1)</sup>, Sculley et al. from Google revealed that **ML code comprises less than 5% of real ML systems**—the remaining 95% involves configuration, monitoring, serving infrastructure, and data verification. This repository bridges that gap by demonstrating the **complete transformation** from experimental notebooks to production-grade services.

The journey from a Jupyter notebook to a production API involves **fundamental architectural transformations**. Your experimental code becomes a distributed system requiring **configuration management**, **dependency injection**, **error handling**, **monitoring**, and **deployment strategies**<sup>[2](#ref2)</sup>. This repository shows you exactly how to make that transformation while maintaining model performance and adding production resilience.

## 💡 The Reality of Production ML Systems

Everyone celebrates achieving high accuracy in notebooks, but **production is where ML systems prove their value**. Through deployment experience serving millions of predictions daily, we've learned that the hardest challenges aren't about model accuracy—they're about **operational excellence**<sup>[3](#ref3)</sup>.

Consider the iris classification problem demonstrated here. Our research revealed that simple heuristics match sophisticated ensemble methods in accuracy. But in production, models must handle **malformed inputs**, **API rate limits**, **deployment rollbacks**, **A/B testing**, **monitoring alerts**, and **configuration changes**—all while maintaining sub-10ms latency<sup>[4](#ref4)</sup>.

**This repository shows you how.** We've transformed a classical ML problem into a production system demonstrating patterns applicable to fraud detection, customer segmentation, quality control, or any feature-based classification task.

## 🔧 Key Architecture Components

### Multi-Model Architecture with Hot Swapping

Our architecture supports **seamless runtime switching** between four distinct models through environment configuration, enabling A/B testing and gradual rollouts without code changes:

| Model | Accuracy | Latency | Memory | Interpretability | Production Use Case |
|-------|----------|---------|--------|------------------|-------------------|
| **Heuristic** | 96.0% | <1ms | 10MB | High (rules) | Real-time, regulated industries |
| **Decision Tree** | 96.7% | <1ms | 15MB | High (tree viz) | Explainable AI requirements |
| **Random Forest** | 96.0% | ~2ms | 50MB | Medium | Balanced performance |
| **XGBoost** | 96.0% | ~3ms | 75MB | Low | High-throughput systems |

The [Factory pattern](ml_production_service/factory.py) combined with [configuration-driven design](ml_production_service/configs.py) enables this flexibility:

```python
# Switch models via environment variable
MPS_MODEL_TYPE=heuristic make api-run        # Development
MPS_MODEL_TYPE=decision_tree make api-run    # Staging
MPS_MODEL_TYPE=random_forest make api-run    # Production
```

### Clean Architecture with Dependency Injection

Following **SOLID principles**<sup>[5](#ref5)</sup>, our architecture ensures maintainability and testability:

```python
# Abstract interface for all predictors
class BasePredictor(ABC):
    @abstractmethod
    def predict(self, measurements: IrisMeasurements) -> str:
        pass

# Factory pattern for model instantiation
def get_predictor(config: ServiceConfig) -> BasePredictor:
    match config.model_type:
        case ModelType.HEURISTIC:
            return HeuristicPredictor()
        case ModelType.RANDOM_FOREST:
            return MLModelPredictor(model_path=config.get_model_path())
```

### Comprehensive Testing & Observability

**97% test coverage** with parametrized testing across all models ensures consistent behavior:

```python
@pytest.mark.parametrize("model_type", [
    ModelType.HEURISTIC, ModelType.DECISION_TREE,
    ModelType.RANDOM_FOREST, ModelType.XGBOOST
])
async def test_prediction_endpoint(model_type):
    # Ensures consistent API behavior across implementations
```

Production observability includes structured logging, health checks, latency tracking, and comprehensive error handling<sup>[6](#ref6)</sup>.

## ⚡ Quick Start

### Prerequisites

- Python 3.10-3.12
- `uv` package manager or standard `pip`
- Docker (optional, for containerized deployment)
- Make (for automation commands)

### Installation

```bash
# Clone and setup
git clone https://github.com/leogarciavargas/ml-production-service
cd ml-production-service
make environment-create

# Validate setup
make validate-branch
```

### Run Locally

```bash
# Start API with chosen model
MPS_MODEL_TYPE=decision_tree make api-run

# Access endpoints
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

### Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, 
       "petal_length": 1.4, "petal_width": 0.2}'

# Response: {"prediction": "setosa"}
```

## 🏭 Production Deployment

### Docker Deployment

```bash
# Quick start (build + run)
MPS_MODEL_TYPE=random_forest make service-quick-start

# Or step by step
make service-build
make service-start
make service-validate
make service-stop
```

### A/B Testing Configuration

Deploy multiple model variants simultaneously:

```yaml
# docker-compose.yml
services:
  model-a:
    image: ml-production-service:latest
    environment:
      - MPS_MODEL_TYPE=random_forest
    ports:
      - "8000:8000"
  
  model-b:
    image: ml-production-service:latest
    environment:
      - MPS_MODEL_TYPE=decision_tree
    ports:
      - "8001:8000"
```

### Cloud Platform Deployment

```bash
# Google Cloud Run
gcloud run deploy ml-production-service \
  --image gcr.io/your-project/ml-production-service \
  --set-env-vars MPS_MODEL_TYPE=random_forest \
  --memory 512Mi --cpu 1

# AWS ECS
aws ecs create-service \
  --service-name ml-production-service \
  --task-definition ml-production-service:latest \
  --desired-count 3

# Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl set env deployment/ml-production-service MPS_MODEL_TYPE=decision_tree
```

### Environment Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `MPS_MODEL_TYPE` | Model selection | `heuristic` | `heuristic`, `decision_tree`, `random_forest`, `xgboost` |
| `MPS_LOG_LEVEL` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MPS_API_HOST` | API bind address | `0.0.0.0` | Any valid host |
| `MPS_API_PORT` | API port | `8000` | Any valid port |

## 🛠️ Development Workflow

### Essential Commands

```bash
# Environment Management
make environment-create     # First-time setup
make environment-sync       # Update dependencies

# Code Quality
make format                # Auto-format with Ruff
make lint                  # Linting checks
make type-check           # MyPy validation
make validate-branch      # All quality checks

# Testing
make unit-test            # Unit tests only
make functional-test      # Functional tests
make all-test            # Complete test suite
make all-test-validate-branch  # Tests + quality

# API Development
make api-run             # Start dev server
make api-validate        # Test running API

# Research & Training
make eval-heuristic          # Evaluate baseline
make train-decision-tree     # Train decision tree
make train-random-forest     # Train random forest
make train-xgboost          # Train XGBoost
```

## 📊 Research Methodology & Results

### Systematic Experimentation

Our research follows a **production-driven model selection** approach:

1. **Baseline Establishment**: Rule-based heuristic as performance floor
2. **Progressive Complexity**: Systematic evaluation from simple to complex
3. **Multiple Validation**: LOOCV, k-fold CV, OOB estimation
4. **Production Metrics**: Beyond accuracy—latency, interpretability, maintenance

### Key Findings

Research insights that shaped our architecture:

- **Accuracy Ceiling**: All models plateau at 96-97% (data limitation)
- **Validation Impact**: LOOCV vs split validation shows 5.6% difference
- **Feature Engineering**: Derived features (petal_area) improve tree models
- **Complexity ROI**: Diminishing returns beyond decision trees

### Model Lifecycle

```
Research Phase → Evaluation → Promotion → Deployment
research/models/ → research/results/ → registry/prd/ → production
```

Complete documentation available in [research/EXPERIMENTS_JOURNEY.md](research/EXPERIMENTS_JOURNEY.md) and individual experiment reports in [research/experiments/](research/experiments/).

## Project Structure

```
ml-production-service/
├── ml_production_service/     # Production service
│   ├── predictors/           # Model implementations
│   ├── server/              # FastAPI application
│   ├── configs.py          # Configuration
│   └── factory.py          # Dependency injection
│
├── research/                 # Experimentation
│   ├── experiments/         # Training code
│   ├── models/             # Trained artifacts
│   └── results/            # Performance metrics
│
├── registry/prd/            # Production models
├── tests/                   # Test suite (97% coverage)
├── Dockerfile              # Container definition
└── Makefile               # Automation commands
```

## Adding New Models

1. Create predictor class implementing `BasePredictor`
2. Register in `ModelType` enum
3. Add to factory function
4. Write comprehensive tests

See [CLAUDE.md](CLAUDE.md#adding-new-predictors--complete-integration-process) for detailed guide.

## 🤝 Contributing

We welcome contributions that strengthen production ML patterns:
- New model implementations following `BasePredictor` interface
- Production patterns (monitoring, deployment strategies)
- Performance optimizations
- Testing strategies for ML systems

Fork → Branch → Test (maintain 97% coverage) → PR with clear description

## Related Resources

### Essential Reading
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) - Sculley et al., NeurIPS 2015
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) - Martin Zinkevich, Google
- [MLOps: Continuous delivery and automation pipelines](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) - Google Cloud

### Project Documentation
- [Application Architecture](ml_production_service/README.md)
- [Research Documentation](research/README.md)
- [Development Guide](CLAUDE.md)
- [Model Registry](registry/prd/README.md)

## References

<a id="ref1"></a><sup>1</sup> Sculley, D., et al. (2015). ["Hidden Technical Debt in Machine Learning Systems"](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). NeurIPS 2015.

<a id="ref2"></a><sup>2</sup> Polyzotis, N., et al. (2017). ["Data Management Challenges in Production Machine Learning"](https://dl.acm.org/doi/10.1145/3035918.3054782). SIGMOD 2017.

<a id="ref3"></a><sup>3</sup> Breck, E., et al. (2017). ["The ML Test Score: A Rubric for ML Production Readiness"](https://research.google/pubs/pub46555/). IEEE Big Data 2017.

<a id="ref4"></a><sup>4</sup> Shankar, S., et al. (2024). ["Operationalizing Machine Learning: An Interview Study"](https://arxiv.org/abs/2209.09125).

<a id="ref5"></a><sup>5</sup> Martin, R.C. (2017). ["Clean Architecture: A Craftsman's Guide to Software Structure and Design"](https://www.pearson.com/en-us/subject-catalog/p/clean-architecture-a-craftsmans-guide-to-software-structure-and-design/P200000009528).

<a id="ref6"></a><sup>6</sup> Huyen, C. (2022). ["Designing Machine Learning Systems"](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/). O'Reilly.

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

🚀 **Ready to deploy production ML?** Start with `make environment-create` and have your first model API running in under 2 minutes.

*From research notebooks to production APIs. For ML engineers shipping real systems.*