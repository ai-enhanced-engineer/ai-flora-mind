# Docker Deployment Guide

This guide covers how to deploy AI Flora Mind using Docker and Docker Compose with configurable model selection.

## Quick Start

### Using Service Commands (Recommended)

The easiest way to manage the AI Flora Mind service:

```bash
# Build and start service (decision_tree model by default)
make service-quick-start

# Or step by step:
make service-build    # Build the service
make service-start    # Start the service
make service-stop     # Stop the service

# Use different models:
FLORA_CLASSIFIER_TYPE=heuristic make service-start
FLORA_CLASSIFIER_TYPE=random_forest make service-start
FLORA_CLASSIFIER_TYPE=decision_tree make service-start
```

### Using Docker Compose Directly

Alternative approach using docker-compose commands:
```bash
# Build services
make docker-compose-build
# or
docker-compose build

# Start services
docker-compose up ai-flora-mind-service

# With specific model types
FLORA_CLASSIFIER_TYPE=random_forest docker-compose up ai-flora-mind-service
FLORA_CLASSIFIER_TYPE=decision_tree docker-compose up ai-flora-mind-service
```

Service will be available at: `http://localhost:8000`

### 3. Configure Development Service

For custom configuration, use environment variables directly:
```bash
# Start development service with specific model
FLORA_CLASSIFIER_TYPE=random_forest docker-compose up ai-flora-mind-service
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `FLORA_CLASSIFIER_TYPE` | Model type to use | `heuristic` | `heuristic`, `random_forest`, `decision_tree`, `xgboost` |

### Packaged Models

All models are packaged inside the Docker image:

| Model Type | Internal Path | Description |
|------------|---------------|-------------|
| `heuristic` | N/A (rule-based) | Fast, lightweight, no file needed |
| `random_forest` | `registry/prd/random_forest.joblib` | High accuracy, 14 features, regularized |
| `decision_tree` | `registry/prd/decision_tree.joblib` | High interpretability, 5 features, fast |
| `xgboost` | `registry/prd/xgboost.joblib` | Maximum performance (not implemented) |

## Advanced Deployment

### Make Targets

Use convenient Make targets for common operations:

**Service Management (Recommended):**
```bash
make service-build        # Build AI Flora Mind service
make service-start        # Start AI Flora Mind service  
make service-stop         # Stop AI Flora Mind service
make service-quick-start  # Build and start in one command
```

**Docker Compose Operations:**
```bash
make docker-compose-build    # Build all services
make docker-compose-up       # Start services with helpful guidance
make docker-compose-down     # Stop all services  
make docker-compose-test     # Test deployment
```

### Load Balancer Setup

Start multiple services with nginx load balancer:
```bash
docker-compose --profile load-balancer up
```

This starts:
- Heuristic service (weight: 3)
- Random Forest service (weight: 1) 
- Nginx load balancer on port 80

### Custom Model Path

Use a different Random Forest model:
```bash
docker-compose up ai-flora-random-forest -e FLORA_RF_MODEL_PATH=research/models/random_forest_comprehensive_2025_07_19_233147.joblib
```

### Health Checks

All services include health checks accessible at `/health`:
```bash
curl http://localhost:8000/health
# Returns: {"status": "healthy"}
```

## Example API Usage

### Test Prediction

```bash
# Test heuristic model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Test decision tree model  
FLORA_CLASSIFIER_TYPE=decision_tree docker run --rm -p 8000:8000 -e FLORA_CLASSIFIER_TYPE=decision_tree ai-flora-mind &
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

Expected response:
```json
{"prediction": "setosa"}
```

## Production Considerations

### Performance Characteristics

| Model | Startup Time | Memory Usage | Prediction Speed | Accuracy |
|-------|--------------|--------------|------------------|----------|
| Heuristic | ~1s | ~50MB | ~1ms | Good |
| Decision Tree | ~3s | ~80MB | ~2ms | Very Good |
| Random Forest | ~10s | ~150MB | ~5ms | Excellent |

### Scaling Recommendations

1. **High Traffic**: Use heuristic model with horizontal scaling
2. **Balanced Performance**: Use decision tree for good accuracy with fast responses
3. **High Accuracy**: Use Random Forest with load balancer
4. **Mixed Workload**: Use nginx load balancer with multiple models

### Monitoring

Health check endpoints are configured for all services:
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 10s (heuristic) / 30s (Random Forest)

## Troubleshooting

### Common Issues

1. **Model file not found**: Ensure `research/models/` directory is mounted
2. **Slow startup**: Random Forest models take 10-30s to load
3. **Memory issues**: Random Forest requires ~150MB RAM

### Logs

View service logs:
```bash
# View heuristic service logs
docker-compose logs ai-flora-heuristic

# View random forest service logs
docker-compose logs ai-flora-random-forest

# Follow logs in real-time
docker-compose logs -f ai-flora-dev
```

### Debug Mode

Start services in debug mode:
```bash
# Override command for debugging
docker-compose run --rm ai-flora-dev /bin/bash
```

## Docker Build

### Build Image Manually

```bash
# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.13 -t ai-flora-mind .

# Run manually with environment variables
docker run -p 8000:8000 \
  -e FLORA_CLASSIFIER_TYPE=random_forest \
  -v $(pwd)/research/models:/app/research/models:ro \
  ai-flora-mind
```

### Multi-architecture Build

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t ai-flora-mind .
```