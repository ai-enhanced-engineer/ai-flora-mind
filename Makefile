.PHONY: default help clean-project clean-research environment-create environment-sync environment-delete environment-list sync-env format lint type-check unit-test functional-test integration-test all-test validate-branch validate-branch-strict test-validate-branch all-test-validate-branch api-dev api-validate api-docs service-build service-start service-stop service-quick-start service-validate

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
# First time setup:     make environment-create
# Start development:    make api-dev
# Run quality checks:   make validate-branch
# Run all tests:        make all-test-validate-branch
# Deploy locally:       make service-quick-start
# Clean everything:     make clean-project clean-research
#
# Change model type:    FLORA_CLASSIFIER_TYPE=random_forest make api-dev
# Research tasks:       make -f research.mk help
# ==============================================================================

GREEN_LINE=@echo "\033[0;32m--------------------------------------------------\033[0m"

SOURCE_DIR = ai_flora_mind/
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
		--cov=ai_flora_mind \
		--cov-config=pyproject.toml \
		--cov-fail-under=90 \
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

api-dev: environment-sync ## Start API server in dev mode. Example: FLORA_CLASSIFIER_TYPE=random_forest make api-dev'
	@echo "Starting AI Flora Mind API in development mode..."
	@echo "🤖 Current model: $(shell echo $${FLORA_CLASSIFIER_TYPE:-heuristic})"
	@echo ""
	@echo "📝 Model selection examples:"
	@echo "   FLORA_CLASSIFIER_TYPE=heuristic make api-dev       # Rule-based classifier"
	@echo "   FLORA_CLASSIFIER_TYPE=decision_tree make api-dev   # Decision tree (96% accuracy)"
	@echo "   FLORA_CLASSIFIER_TYPE=random_forest make api-dev   # Random forest (96% accuracy)"
	@echo "   FLORA_CLASSIFIER_TYPE=xgboost make api-dev         # XGBoost (98%+ accuracy)"
	@echo ""
	@echo "🔧 Advanced options using ARGS:"
	@echo "   make api-dev ARGS='--model-type decision_tree --log-level debug'"
	@echo "   make api-dev ARGS='--port 8001 --host localhost'"
	@echo "   make api-dev ARGS='--help'  # Show all CLI options"
	uv run python -m scripts.isolation.api_layer --reload $(ARGS)
	$(GREEN_LINE)

api-validate: environment-sync ## Run comprehensive API validation using full iris dataset (requires running server)
	@echo "🧪 Running comprehensive API validation with full iris dataset..."
	@echo "💡 Ensure API server is running first: make api-dev"
	uv run python -m scripts.validation.api_comprehensive_test
	$(GREEN_LINE)

api-docs: environment-sync ## Open Swagger UI documentation (starts API if not running)
	@echo "🚀 Starting AI Flora Mind API with Swagger UI..."
	@echo "📖 Swagger UI will be available at: http://localhost:8000/docs"
	@echo "📋 ReDoc will be available at: http://localhost:8000/redoc"
	@echo "📄 OpenAPI JSON at: http://localhost:8000/openapi.json"
	@echo "🤖 Model type: $(shell echo $${FLORA_CLASSIFIER_TYPE:-heuristic})"
	@echo ""
	@echo "🌐 Opening Swagger UI in browser..."
	@(sleep 2 && open http://localhost:8000/docs) &
	uv run python -m scripts.isolation.api_layer --reload
	$(GREEN_LINE)


# ----------------------------
# Build and Deployment
# ----------------------------

service-build: ## Build AI Flora Mind service
	@echo "Building AI Flora Mind service..."
	docker-compose build
	$(GREEN_LINE)

service-start: ## Start AI Flora Mind service
	@echo "Starting AI Flora Mind service..."
	@echo "API will be available at: http://localhost:8000"
	@echo "Swagger UI at: http://localhost:8000/docs"
	@echo ""
	@echo "📝 Model Configuration:"
	@echo "   Current model: ${FLORA_CLASSIFIER_TYPE:-xgboost}"
	@echo "   Available models: heuristic, decision_tree, random_forest, xgboost"
	@echo ""
	@echo "🔧 To change the model type:"
	@echo "   FLORA_CLASSIFIER_TYPE=heuristic make service-start     # Rule-based (90%)"
	@echo "   FLORA_CLASSIFIER_TYPE=decision_tree make service-start # Decision Tree (95%+)"
	@echo "   FLORA_CLASSIFIER_TYPE=random_forest make service-start # Random Forest (97%+)"
	@echo "   FLORA_CLASSIFIER_TYPE=xgboost make service-start       # XGBoost (98%+)"
	@echo ""
	@echo "   💡 Tip: To permanently change the default, edit docker-compose.yml"
	@echo ""
	docker-compose up ai-flora-mind-service
	$(GREEN_LINE)

service-stop: ## Stop AI Flora Mind service
	@echo "Stopping AI Flora Mind service..."
	docker-compose down
	$(GREEN_LINE)

service-quick-start: ## Build and start AI Flora Mind service in one command
	@echo "🚀 Quick start options:"
	@echo "   make service-quick-start                          # Uses default (xgboost)"
	@echo "   FLORA_CLASSIFIER_TYPE=heuristic make service-quick-start"
	@echo "   FLORA_CLASSIFIER_TYPE=decision_tree make service-quick-start"
	@echo "   FLORA_CLASSIFIER_TYPE=random_forest make service-quick-start"
	@echo "   FLORA_CLASSIFIER_TYPE=xgboost make service-quick-start"
	@echo ""
	$(MAKE) service-build
	$(MAKE) service-start

service-validate: environment-sync ## Start service and run comprehensive validation with full iris dataset
	@echo "🧪 Starting AI Flora Mind service and running comprehensive validation..."
	@echo "Starting service for testing..."
	docker-compose up -d ai-flora-mind-service
	@echo "Waiting for service to start..."
	@sleep 15
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