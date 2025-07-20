.PHONY: default help clean-project clean-research environment-create environment-sync environment-delete environment-list sync-env format lint type-check unit-test functional-test integration-test all-test validate-branch validate-branch-strict test-validate-branch all-test-validate-branch local-run api-layer-isolate docker-compose-build docker-compose-up docker-compose-down docker-compose-test eval-all-experiments eval-xgboost eval-xgboost-comprehensive eval-xgboost-optimized

GREEN_LINE=@echo "\033[0;32m--------------------------------------------------\033[0m"

SOURCE_DIR = ai_flora_mind/
TEST_DIR = tests/
PROJECT_VERSION := $(shell awk '/^\[project\]/ {flag=1; next} /^\[/{flag=0} flag && /^version/ {gsub(/"/, "", $$2); print $$2}' pyproject.toml)
PYTHON_VERSION := $(shell cat .python-version)
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

local-run: ## Run the flora mind service locally with auto-reload
	@echo "Starting flora mind service locally..."
	@echo "🤖 Model type: $(shell echo $${FLORA_MODEL_TYPE:-heuristic})"
	@echo "📝 To change model: FLORA_MODEL_TYPE=random_forest make local-run"
	@echo "📊 Available models: heuristic, random_forest, decision_tree, xgboost"
	uv run uvicorn ai_flora_mind.server.main:get_app --factory --reload --host 0.0.0.0 --port 8000
	$(GREEN_LINE)

api-layer-isolate: ## Start the API server locally for testing and debugging
	@echo "Starting AI Flora Mind API in isolation..."
	@echo "🤖 Model type: $(shell echo $${FLORA_MODEL_TYPE:-heuristic})"
	@echo "📝 To change model: FLORA_MODEL_TYPE=random_forest make api-layer-isolate"
	@echo "📊 Available models: heuristic, random_forest, decision_tree, xgboost"
	uv run python -m scripts.isolation.api_layer --reload
	$(GREEN_LINE)

api-layer-validate: ## Run comprehensive API validation using full iris dataset (requires running server)
	@echo "🧪 Running comprehensive API validation with full iris dataset..."
	@echo "💡 Ensure API server is running first: make api-layer-isolate"
	uv run python -m scripts.validation.api_comprehensive_test
	$(GREEN_LINE)

api-docs: ## Open Swagger UI documentation (starts API if not running)
	@echo "🚀 Starting AI Flora Mind API with Swagger UI..."
	@echo "📖 Swagger UI will be available at: http://localhost:8000/docs"
	@echo "📋 ReDoc will be available at: http://localhost:8000/redoc"
	@echo "📄 OpenAPI JSON at: http://localhost:8000/openapi.json"
	@echo "🤖 Model type: $(shell echo $${FLORA_MODEL_TYPE:-heuristic})"
	@echo ""
	@echo "🌐 Opening Swagger UI in browser..."
	@(sleep 2 && open http://localhost:8000/docs) &
	uv run python -m scripts.isolation.api_layer --reload
	$(GREEN_LINE)

# ----------------------------
# Research and Modeling
# ----------------------------

eval-heuristic: ## Evaluate rule-based heuristic classifier on full Iris dataset
	@echo "🌸 Evaluating Rule-Based Heuristic Iris Classifier..."
	@echo "📊 Running comprehensive performance evaluation..."
	uv run python -m research.experiments.rule_based_heuristic.iris_heuristic_classifier
	$(GREEN_LINE)

eval-decision-tree: ## Train decision tree with train/test split (original experiment)
	@echo "🌳 Training Decision Tree Iris Classifier (Split Experiment)..."
	@echo "📊 Running model training and evaluation..."
	uv run python -m research.experiments.decision_tree.iris_decision_tree_classifier --experiment split
	$(GREEN_LINE)

eval-decision-tree-comprehensive: ## Train decision tree with comprehensive validation (full dataset + LOOCV + repeated k-fold)
	@echo "🌳 Training Decision Tree Iris Classifier (Comprehensive Validation)..."
	@echo "📊 Running comprehensive validation with LOOCV and repeated k-fold CV..."
	uv run python -m research.experiments.decision_tree.iris_decision_tree_classifier --experiment comprehensive
	$(GREEN_LINE)

eval-random-forest: ## Train Random Forest with train/test split (targeting 98-99% accuracy)
	@echo "🌲 Training Random Forest Iris Classifier (Split Experiment)..."
	@echo "📊 Running ensemble learning with all 14 features..."
	uv run python -m research.experiments.random_forest.iris_random_forest_classifier --experiment split
	$(GREEN_LINE)

eval-random-forest-comprehensive: ## Train Random Forest with comprehensive validation (full dataset + LOOCV + repeated k-fold)
	@echo "🌲 Training Random Forest Iris Classifier (Comprehensive Validation)..."
	@echo "📊 Running comprehensive validation with LOOCV and repeated k-fold CV..."
	uv run python -m research.experiments.random_forest.iris_random_forest_classifier --experiment comprehensive
	$(GREEN_LINE)

eval-random-forest-regularized: ## Train Random Forest with regularized configuration to prevent overfitting
	@echo "🌲 Training Random Forest Iris Classifier (Regularized Configuration)..."
	@echo "📊 Running overfitting-prevention experiment with depth limits and reduced trees..."
	uv run python -m research.experiments.random_forest.iris_random_forest_classifier --experiment regularized
	$(GREEN_LINE)

eval-xgboost: ## Train XGBoost with train/test split (targeting theoretical maximum 98-99.5% accuracy)
	@echo "🚀 Training XGBoost Iris Classifier (Split Experiment)..."
	@echo "📊 Running gradient boosting with targeted high-discriminative features..."
	uv run python -m research.experiments.xgboost.iris_xgboost_classifier --experiment split
	$(GREEN_LINE)

eval-xgboost-comprehensive: ## Train XGBoost with comprehensive validation (full dataset + LOOCV)
	@echo "🚀 Training XGBoost Iris Classifier (Comprehensive Validation)..."
	@echo "📊 Running comprehensive validation with overfitting monitoring..."
	uv run python -m research.experiments.xgboost.iris_xgboost_classifier --experiment comprehensive
	$(GREEN_LINE)

eval-xgboost-optimized: ## Train XGBoost with optimized hyperparameters and overfitting prevention
	@echo "🚀 Training XGBoost Iris Classifier (Optimized Configuration)..."
	@echo "📊 Running theoretical performance ceiling experiment with aggressive regularization..."
	uv run python -m research.experiments.xgboost.iris_xgboost_classifier --experiment optimized
	$(GREEN_LINE)

eval-all-experiments: ## Run all iris classifier experiments in sequence
	@echo "🚀 Running ALL Iris Classifier Experiments..."
	@echo "This will run all experiments: heuristic, decision tree (split + comprehensive), random forest (split + comprehensive + regularized), and xgboost (split + comprehensive + optimized)"
	@echo ""
	$(MAKE) eval-heuristic
	$(MAKE) eval-decision-tree
	$(MAKE) eval-decision-tree-comprehensive
	$(MAKE) eval-random-forest
	$(MAKE) eval-random-forest-comprehensive
	$(MAKE) eval-random-forest-regularized
	$(MAKE) eval-xgboost
	$(MAKE) eval-xgboost-comprehensive
	$(MAKE) eval-xgboost-optimized
	@echo ""
	@echo "🎉 All experiments completed successfully!"
	@echo "📂 Check research/results/ for experiment outputs"
	@echo "🤖 Check research/models/ for saved models"
	$(GREEN_LINE)

# ----------------------------
# Build and Deployment
# ----------------------------

docker-build: ## Build Docker image for AI Flora Mind API
	@echo "Building AI Flora Mind Docker image with Python $(PYTHON_VERSION)..."
	DOCKER_BUILDKIT=1 docker build --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -t ai-flora-mind:latest .
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

# ----------------------------
# Docker Compose Deployment
# ----------------------------

docker-compose-build: ## Build Docker Compose services
	@echo "Building Docker Compose services..."
	docker-compose build
	$(GREEN_LINE)

docker-compose-up: ## Start Docker Compose services (use docker-compose.yml to configure models)
	@echo "Starting AI Flora Mind services..."
	@echo "📝 Configure models in docker-compose.yml or use .env file"
	@echo "📊 Available services:"
	@echo "  - ai-flora-heuristic (port 8000)"
	@echo "  - ai-flora-random-forest (port 8001)"
	@echo "  - ai-flora-dev (port 8002, configurable)"
	@echo ""
	@echo "Examples:"
	@echo "  docker-compose up ai-flora-heuristic"
	@echo "  docker-compose up ai-flora-random-forest"
	@echo "  docker-compose up ai-flora-dev"
	@echo "  docker-compose up  # All services"
	docker-compose up

docker-compose-down: ## Stop all Docker Compose services
	@echo "Stopping all AI Flora Mind services..."
	docker-compose down
	$(GREEN_LINE)

docker-compose-test: ## Test Docker Compose deployment
	@echo "Testing Docker Compose deployment..."
	@echo "Starting development service for testing..."
	docker-compose up -d ai-flora-dev
	@echo "Waiting for service to start..."
	@sleep 15
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8002/health || (echo "Health check failed" && exit 1)
	@echo "Testing prediction endpoint..."
	@curl -X POST http://localhost:8002/predict \
		-H "Content-Type: application/json" \
		-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' \
		|| (echo "Prediction test failed" && exit 1)
	@echo "✅ Docker Compose deployment test passed!"
	docker-compose down
	$(GREEN_LINE)
