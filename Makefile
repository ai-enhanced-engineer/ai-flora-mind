.PHONY: default help clean-project clean-research environment-create environment-sync environment-delete environment-list sync-env format lint type-check unit-test functional-test integration-test all-test validate-branch validate-branch-strict test-validate-branch all-test-validate-branch api-run api-validate api-kill api-docs service-build service-start service-stop service-quick-start service-validate

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
# First time setup:     make environment-create
# Start development:    make api-run
# Run quality checks:   make validate-branch
# Run all tests:        make all-test-validate-branch
# Deploy locally:       make service-quick-start
# Clean everything:     make clean-project clean-research
#
# Change model type:    MPS_MODEL_TYPE=random_forest make api-run
# Research tasks:       make -f research.mk help
# ==============================================================================

GREEN_LINE=@echo "\033[0;32m--------------------------------------------------\033[0m"

SOURCE_DIR = ml_production_service/
TEST_DIR = tests/
PROJECT_VERSION := $(shell awk '/^\[project\]/ {flag=1; next} /^\[/{flag=0} flag && /^version/ {gsub(/"/, "", $$2); print $$2}' pyproject.toml)
PYTHON_VERSION := $(shell cat .python-version)
CLIENT_ID = leogv

# Include research and modeling targets
include research.mk

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

clean-research: ## Clean all research outputs (models and results)
	@echo "🧹 Cleaning research outputs..."
	@echo "Removing all saved models..."
	@if [ -d "research/models" ]; then rm -rf research/models/*; echo "✅ Models directory cleaned"; else echo "ℹ️  Models directory doesn't exist"; fi
	@echo "Removing all experiment results..."
	@if [ -d "research/results" ]; then rm -rf research/results/*; echo "✅ Results directory cleaned"; else echo "ℹ️  Results directory doesn't exist"; fi
	@echo "🎉 Research cleanup completed!"
	$(GREEN_LINE)

environment-create: ## Set up Python version, venv, and install dependencies
	@echo "🔧 Installing uv if missing..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "📦 Installing uv..."; \
		python3 -m pip install --user --upgrade uv; \
	else \
		echo "✅ uv is already installed"; \
	fi
	@echo "🐍 Checking Python $(PYTHON_VERSION) availability..."
	@if ! uv python list | grep -q "cpython-$(PYTHON_VERSION)"; then \
		echo "📥 Python $(PYTHON_VERSION) not found, installing..."; \
		uv python install $(PYTHON_VERSION); \
		if [ $$? -eq 0 ]; then \
			echo "✅ Python $(PYTHON_VERSION) installed successfully"; \
		else \
			echo "❌ Failed to install Python $(PYTHON_VERSION)"; \
			exit 1; \
		fi; \
	else \
		echo "✅ Python $(PYTHON_VERSION) is already available"; \
	fi
	@echo "🏗️  Creating virtual environment with Python $(PYTHON_VERSION)..."
	uv venv --python $(PYTHON_VERSION)
	@echo "📦 Installing project dependencies..."
	uv sync --extra dev
	@echo "🪝 Setting up pre-commit hooks..."
	uv run pre-commit install
	@echo "🎉 Environment setup complete!"
	$(GREEN_LINE)

environment-sync: ## Re-sync project dependencies using uv
	@echo "🔄 Syncing project dependencies..."
	@if [ ! -d ".venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make environment-create' first."; \
		exit 1; \
	fi
	uv sync --extra dev
	@echo "✅ Dependencies synced successfully!"
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
		--cov=ml_production_service \
		--cov-config=pyproject.toml \
		--cov-fail-under=97 \
		--cov-report=term-missing

# ----------------------------
# Branch Validation
# ----------------------------

validate-branch: ## Run formatting, linting, and tests (equivalent to old behavior)
	@echo "🔍 Running validation checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
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

api-run: environment-sync ## Start API server in dev mode. Example: MPS_MODEL_TYPE=random_forest make api-run'
	@echo "Starting ML Production Service API in development mode..."
	@echo "🤖 Current model: $(shell echo $${MPS_MODEL_TYPE:-heuristic})"
	@echo ""
	@echo "📝 Model selection examples:"
	@echo "   MPS_MODEL_TYPE=heuristic make api-run       # Rule-based classifier"
	@echo "   MPS_MODEL_TYPE=decision_tree make api-run   # Decision tree (96% accuracy)"
	@echo "   MPS_MODEL_TYPE=random_forest make api-run   # Random forest (96% accuracy)"
	@echo "   MPS_MODEL_TYPE=xgboost make api-run         # XGBoost (98%+ accuracy)"
	@echo ""
	@echo "🔧 Advanced options using ARGS:"
	@echo "   make api-run ARGS='--model-type decision_tree --log-level debug'"
	@echo "   make api-run ARGS='--port 8001 --host localhost'"
	@echo "   make api-run ARGS='--help'  # Show all CLI options"
	uv run python -m scripts.isolation.api_layer --reload $(ARGS)
	$(GREEN_LINE)

api-validate: ## Run comprehensive API validation using full iris dataset (requires running server)
	@echo "🧪 Running comprehensive API validation with full iris dataset..."
	@echo "💡 Ensure API server is running first: make api-run"
	uv run python -m scripts.validation.api_comprehensive_test
	$(GREEN_LINE)

api-kill: ## Kill running API development server
	@echo "🛑 Stopping API development server..."
	@pkill -f "scripts.isolation.api_layer" && echo "✅ API server stopped successfully" || echo "ℹ️  No API server process found"
	$(GREEN_LINE)

api-docs: environment-sync ## Open Swagger UI documentation (starts API if not running)
	@echo "🚀 Starting ML Production Service API with Swagger UI..."
	@echo "📖 Swagger UI will be available at: http://localhost:8000/docs"
	@echo "📋 ReDoc will be available at: http://localhost:8000/redoc"
	@echo "📄 OpenAPI JSON at: http://localhost:8000/openapi.json"
	@echo "🤖 Model type: $(shell echo $${MPS_MODEL_TYPE:-heuristic})"
	@echo ""
	@echo "🌐 Opening Swagger UI in browser..."
	@(sleep 2 && open http://localhost:8000/docs) &
	uv run python -m scripts.isolation.api_layer --reload
	$(GREEN_LINE)

api-test-all-models: environment-sync ## Test all 4 models via API with sample predictions
	@echo "🧪 Testing all models via API..."
	@for MODEL in heuristic decision_tree random_forest xgboost; do \
		echo ""; \
		echo "=== Testing $$MODEL ==="; \
		MPS_MODEL_TYPE=$$MODEL make api-run > /tmp/api_$$MODEL.log 2>&1 & \
		API_PID=$$!; \
		sleep 3; \
		echo "Test 1 (Setosa):"; \
		curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'; \
		echo ""; \
		echo "Test 2 (Versicolor):"; \
		curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}'; \
		echo ""; \
		echo "Test 3 (Virginica):"; \
		curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 6.7, "sepal_width": 3.3, "petal_length": 5.7, "petal_width": 2.1}'; \
		echo ""; \
		kill $$API_PID 2>/dev/null || true; \
		sleep 1; \
	done
	@echo ""
	@echo "✅ All models tested successfully!"
	$(GREEN_LINE)

# ----------------------------
# Build and Deployment
# ----------------------------

service-build: environment-sync  ## Build ML Production Service service
	@echo "Building ML Production Service service..."
	docker-compose build
	$(GREEN_LINE)

service-start: service-build ## Start ML Production Service service
	@echo "Starting ML Production Service service..."
	@echo "API will be available at: http://localhost:8000"
	@echo "Swagger UI at: http://localhost:8000/docs"
	@echo ""
	@echo "📝 Model Configuration:"
	@echo "   Current model: ${MPS_MODEL_TYPE:-xgboost}"
	@echo "   Available models: heuristic, decision_tree, random_forest, xgboost"
	@echo ""
	@echo "🔧 To change the model type:"
	@echo "   MPS_MODEL_TYPE=heuristic make service-start     # Rule-based (90%)"
	@echo "   MPS_MODEL_TYPE=decision_tree make service-start # Decision Tree (95%+)"
	@echo "   MPS_MODEL_TYPE=random_forest make service-start # Random Forest (97%+)"
	@echo "   MPS_MODEL_TYPE=xgboost make service-start       # XGBoost (98%+)"
	@echo ""
	@echo "   💡 Tip: To permanently change the default, edit docker-compose.yml"
	@echo ""
	docker-compose up ml-production-service
	$(GREEN_LINE)

service-stop: ## Stop ML Production Service service
	@echo "Stopping ML Production Service service..."
	docker-compose down
	$(GREEN_LINE)

service-quick-start: ## Build and start ML Production Service service in one command
	@echo "🚀 Quick start options:"
	@echo "   make service-quick-start                          # Uses default (xgboost)"
	@echo "   MPS_MODEL_TYPE=heuristic make service-quick-start"
	@echo "   MPS_MODEL_TYPE=decision_tree make service-quick-start"
	@echo "   MPS_MODEL_TYPE=random_forest make service-quick-start"
	@echo "   MPS_MODEL_TYPE=xgboost make service-quick-start"
	@echo ""
	$(MAKE) service-build
	$(MAKE) service-start

service-validate: environment-sync ## Start service and run comprehensive validation with full iris dataset
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8000/health || (echo "Health check failed" && exit 1)
	@echo "Running comprehensive API validation with full iris dataset..."
	uv run python -m scripts.validation.api_comprehensive_test || (echo "Comprehensive validation failed" && $(MAKE) service-stop && exit 1)
	@echo "✅ Service comprehensive validation passed!"
	$(GREEN_LINE)

# ----------------------------
# Research and Modeling (from research.mk)
# ----------------------------
# See research.mk for all research and modeling targets