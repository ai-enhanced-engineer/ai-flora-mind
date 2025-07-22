# =============================================================================
# Research and Modeling Makefile
# =============================================================================
# 
# This file contains all research and modeling targets for AI Flora Mind.
# These targets handle model training, evaluation, and experimentation.
#
# Usage:
#   make eval-heuristic              # Evaluate heuristic classifier
#   make train-decision-tree         # Train decision tree (split experiment)
#   make train-random-forest         # Train random forest (split experiment)
#   make train-xgboost               # Train XGBoost (split experiment)
#   make run-all-experiments         # Run all experiments in sequence
#
# Include this file in the main Makefile with: include research.mk
#

# PHONY declarations for research targets
.PHONY: eval-heuristic train-decision-tree train-decision-tree-comprehensive train-random-forest train-random-forest-comprehensive train-random-forest-regularized train-xgboost train-xgboost-comprehensive train-xgboost-optimized run-all-experiments

# ----------------------------
# Research and Modeling
# ----------------------------

eval-heuristic: environment-sync ## Evaluate rule-based heuristic classifier on full Iris dataset
	@echo "🌸 Evaluating Rule-Based Heuristic Iris Classifier..."
	@echo "📊 Running comprehensive performance evaluation..."
	uv run python -m research.experiments.rule_based_heuristic.iris_heuristic_classifier
	$(GREEN_LINE)

train-decision-tree: environment-sync ## Train decision tree with train/test split (original experiment)
	@echo "🌳 Training Decision Tree Iris Classifier (Split Experiment)..."
	@echo "📊 Running model training and evaluation..."
	uv run python -m research.experiments.decision_tree.split
	$(GREEN_LINE)

train-decision-tree-comprehensive: environment-sync ## Train decision tree with comprehensive validation (full dataset + LOOCV + repeated k-fold)
	@echo "🌳 Training Decision Tree Iris Classifier (Comprehensive Validation)..."
	@echo "📊 Running comprehensive validation with LOOCV and repeated k-fold CV..."
	uv run python -m research.experiments.decision_tree.comprehensive
	$(GREEN_LINE)

train-random-forest: environment-sync ## Train Random Forest with train/test split (targeting 98-99% accuracy)
	@echo "🌲 Training Random Forest Iris Classifier (Split Experiment)..."
	@echo "📊 Running ensemble learning with all 14 features..."
	uv run python -m research.experiments.random_forest.split
	$(GREEN_LINE)

train-random-forest-comprehensive: environment-sync ## Train Random Forest with comprehensive validation (full dataset + LOOCV + repeated k-fold)
	@echo "🌲 Training Random Forest Iris Classifier (Comprehensive Validation)..."
	@echo "📊 Running comprehensive validation with LOOCV and repeated k-fold CV..."
	uv run python -m research.experiments.random_forest.comprehensive
	$(GREEN_LINE)

train-random-forest-regularized: environment-sync ## Train Random Forest with regularized configuration to prevent overfitting
	@echo "🌲 Training Random Forest Iris Classifier (Regularized Configuration)..."
	@echo "📊 Running overfitting-prevention experiment with depth limits and reduced trees..."
	uv run python -m research.experiments.random_forest.regularized
	$(GREEN_LINE)

train-xgboost: environment-sync ## Train XGBoost with train/test split (targeting theoretical maximum 98-99.5% accuracy)
	@echo "🚀 Training XGBoost Iris Classifier (Split Experiment)..."
	@echo "📊 Running gradient boosting with targeted high-discriminative features..."
	uv run python -m research.experiments.xgboost.split
	$(GREEN_LINE)

train-xgboost-comprehensive: environment-sync ## Train XGBoost with comprehensive validation (full dataset + LOOCV)
	@echo "🚀 Training XGBoost Iris Classifier (Comprehensive Validation)..."
	@echo "📊 Running comprehensive validation with overfitting monitoring..."
	uv run python -m research.experiments.xgboost.comprehensive
	$(GREEN_LINE)

train-xgboost-optimized: environment-sync ## Train XGBoost with optimized hyperparameters and overfitting prevention
	@echo "🚀 Training XGBoost Iris Classifier (Optimized Configuration)..."
	@echo "📊 Running theoretical performance ceiling experiment with aggressive regularization..."
	uv run python -m research.experiments.xgboost.optimized
	$(GREEN_LINE)

run-all-experiments: ## Run all iris classifier experiments in sequence
	@echo "🚀 Running ALL Iris Classifier Experiments..."
	@echo "This will run all experiments: heuristic, decision tree (split + comprehensive), random forest (split + comprehensive + regularized), and xgboost (split + comprehensive + optimized)"
	@echo ""
	$(MAKE) eval-heuristic
	$(MAKE) train-decision-tree
	$(MAKE) train-decision-tree-comprehensive
	$(MAKE) train-random-forest
	$(MAKE) train-random-forest-comprehensive
	$(MAKE) train-random-forest-regularized
	$(MAKE) train-xgboost
	$(MAKE) train-xgboost-comprehensive
	$(MAKE) train-xgboost-optimized
	@echo ""
	@echo "🎉 All experiments completed successfully!"
	@echo "📂 Check research/results/ for experiment outputs"
	@echo "🤖 Check research/models/ for saved models"
	$(GREEN_LINE)