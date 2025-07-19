.PHONY: default help clean-project environment-create environment-sync environment-delete environment-list sync-env format lint type-check unit-test functional-test integration-test all-test validate-branch validate-branch-strict test-validate-branch all-test-validate-branch local-run build-engine auth-gcloud

GREEN_LINE=@echo "\033[0;32m--------------------------------------------------\033[0m"

SOURCE_DIR = ai_flora_mind/
TEST_DIR = tests/
PROJECT_VERSION := $(shell awk '/^\[project\]/ {flag=1; next} /^\[/{flag=0} flag && /^version/ {gsub(/"/, "", $$2); print $$2}' pyproject.toml)
PYTHON_VERSION := 3.12
CLIENT_ID = leogv

default: help

help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-35s\033[0m %s\n", $$1, $$2}'

# ----------------------------
# Environment Management
# ----------------------------

clean-project: ## Clean Python caches and tooling artifacts
	@echo "Cleaning project caches..."
	find . -type d \( -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' -o -name '__pycache__' \) -exec rm -rf {} +
	$(GREEN_LINE)

environment-create: ## Set up Python version, venv, and install dependencies
	@echo "Installing uv and pre-commit if missing..."
	@if ! command -v uv >/dev/null 2>&1; then \
		python3 -m pip install --user --upgrade uv; \
	fi
	@echo "Setting up Python $(PYTHON_VERSION) environment..."
	uv python install $(PYTHON_VERSION)
	uv venv --python $(PYTHON_VERSION)
	uv sync --extra dev
	uv pip install -e '.[dev]'
	uv pip install pre-commit
	uv run pre-commit install
	$(GREEN_LINE)

environment-sync: ## Re-sync project dependencies using uv
	@echo "Syncing up environment..."
	uv sync --extra dev
	uv pip install -e '.[dev]'
	$(GREEN_LINE)

sync-env: environment-sync ## Alias for environment-sync

environment-delete: ## Remove the virtual environment folder
	@echo "Deleting virtual environment..."
	rm -rf .venv
	$(GREEN_LINE)

environment-list: ## List installed packages
	@echo "Listing packages in environment..."
	uv pip list

# ----------------------------
# Code Quality
# ----------------------------

format: ## Format codebase using ruff
	@echo "Formatting code with ruff..."
	uv run ruff format
	$(GREEN_LINE)

lint: ## Lint code using ruff and autofix issues
	@echo "Running lint checks with ruff..."
	uv run ruff check . --fix
	$(GREEN_LINE)

type-check: ## Perform static type checks using mypy
	@echo "Running type checks with mypy..."
	uv run --extra dev mypy $(SOURCE_DIR) research/
	$(GREEN_LINE)

# ----------------------------
# Tests
# ----------------------------

unit-test: ## Run unit tests with pytest
	@echo "Running UNIT tests with pytest..."
	uv run python -m pytest -vv --verbose -s $(TEST_DIR)

functional-test: ## Run functional tests with pytest
	@echo "Running FUNCTIONAL tests with pytest..."
	uv run python -m pytest -m functional -vv --verbose -s $(TEST_DIR)

integration-test: ## Run integration tests with pytest
	@echo "Running INTEGRATION tests with pytest..."
	uv run python -m pytest -m integration -vv --verbose -s $(TEST_DIR)

all-test: ## Run all tests with coverage report
	@echo "Running ALL tests with pytest..."
	uv run python -m pytest -m "not integration" -vv -s $(TEST_DIR) \
		--cov=ai_flora_mind \
		--cov-config=pyproject.toml \
		--cov-fail-under=80 \
		--cov-report=term-missing

# ----------------------------
# Branch Validation
# ----------------------------

validate-branch: ## Run formatting, linting, and tests (equivalent to old behavior)
	@echo "🔍 Running validation checks..."
	@echo "📝 Running linting..."
	uv run ruff check .
	@echo "✅ Linting passed!"
	@echo "🧪 Running tests..."
	uv run python -m pytest
	@echo "✅ All tests passed!"
	@echo "🎉 Branch validation successful - ready for PR!"

validate-branch-strict: ## Run formatting, linting, type checks, and tests
	$(MAKE) sync-env
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check

test-validate-branch: ## Validate branch and run unit tests
	$(MAKE) validate-branch
	$(MAKE) unit-test
	$(MAKE) clean-project

all-test-validate-branch: ## Validate branch and run all tests
	$(MAKE) validate-branch
	$(MAKE) all-test
	$(MAKE) clean-project

# ----------------------------
# Local Development
# ----------------------------

local-run: ## Run the flora mind service locally with auto-reload
	@echo "Starting flora mind service locally..."
	uv run uvicorn ai_flora_mind.main:app --reload --host 0.0.0.0 --port 8000
	$(GREEN_LINE)

api-layer-isolate: ## Isolate the API layer locally for testing and debugging
	@echo "Starting AI Flora Mind API in isolation..."
	uv run python -m scripts.isolation.api_layer
	$(GREEN_LINE)

api-layer-ping: ## Test the API layer with curl requests (assumes API is running on localhost:8000)
	@echo "Testing AI Flora Mind API endpoints..."
	@echo "🔍 Testing health endpoint..."
	@curl -s -X GET http://localhost:8000/health | jq '.' || echo "Health check failed or jq not available"
	@echo ""
	@echo "🌸 Testing prediction endpoint..."
	@curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' \
		| jq '.' || echo "Prediction test failed or jq not available"
	@echo ""
	@echo "✅ API ping tests completed!"
	$(GREEN_LINE)

api-docs: ## Open Swagger UI documentation (starts API if not running)
	@echo "🚀 Starting AI Flora Mind API with Swagger UI..."
	@echo "📖 Swagger UI will be available at: http://localhost:8000/docs"
	@echo "📋 ReDoc will be available at: http://localhost:8000/redoc"
	@echo "📄 OpenAPI JSON at: http://localhost:8000/openapi.json"
	@echo ""
	@echo "🌐 Opening Swagger UI in browser..."
	@(sleep 2 && open http://localhost:8000/docs) &
	uv run python -m scripts.isolation.api_layer
	$(GREEN_LINE)

# ----------------------------
# Research and Modeling
# ----------------------------

eval-heuristic: ## Evaluate rule-based heuristic classifier on full Iris dataset
	@echo "🌸 Evaluating Rule-Based Heuristic Iris Classifier..."
	@echo "📊 Running comprehensive performance evaluation..."
	uv run python -m research.baseline.rule_based_heuristic.iris_heuristic_classifier
	$(GREEN_LINE)

eval-decision-tree: ## Train decision tree with train/test split (original experiment)
	@echo "🌳 Training Decision Tree Iris Classifier (Split Experiment)..."
	@echo "📊 Running model training and evaluation..."
	uv run python -m research.baseline.decision_tree.iris_decision_tree_classifier --experiment split
	$(GREEN_LINE)

eval-decision-tree-comprehensive: ## Train decision tree with comprehensive validation (full dataset + LOOCV + repeated k-fold)
	@echo "🌳 Training Decision Tree Iris Classifier (Comprehensive Validation)..."
	@echo "📊 Running comprehensive validation with LOOCV and repeated k-fold CV..."
	uv run python -m research.baseline.decision_tree.iris_decision_tree_classifier --experiment comprehensive
	$(GREEN_LINE)

# ----------------------------
# Build and Deployment
# ----------------------------

docker-build: ## Build Docker image for AI Flora Mind API
	@echo "Building AI Flora Mind Docker image..."
	DOCKER_BUILDKIT=1 docker build -t ai-flora-mind:latest .
	$(GREEN_LINE)

docker-run: ## Run AI Flora Mind API in Docker container
	@echo "Running AI Flora Mind API in Docker..."
	@echo "API will be available at: http://localhost:8000"
	@echo "Swagger UI at: http://localhost:8000/docs"
	docker run -p 8000:8000 ai-flora-mind:latest
	$(GREEN_LINE)

docker-build-run: ## Build and run Docker container in one command
	$(MAKE) docker-build
	$(MAKE) docker-run
