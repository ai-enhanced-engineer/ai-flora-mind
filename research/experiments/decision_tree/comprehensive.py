"""
Decision Tree Iris Classifier - Comprehensive Experiment

This module implements the comprehensive experiment for decision tree on the Iris dataset.
Uses Leave-One-Out Cross-Validation (LOOCV) to maximize data usage on small dataset.

Configuration:
- Max depth: 3 (maintains interpretability)
- Features: Original 4 + petal_area
- Validation: LOOCV + repeated k-fold CV
"""

from datetime import datetime

from sklearn.tree import DecisionTreeClassifier

from ml_production_service.logging import get_logger
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.base import (
    extract_feature_importance,
    load_and_prepare_data,
    perform_comprehensive_validation,
    save_experiment_results,
)
from research.features import ModelType

logger = get_logger(__name__)


def main() -> None:
    """Run decision tree comprehensive experiment with full dataset validation."""
    logger.info("Starting Decision Tree Comprehensive Experiment")

    # Load and prepare data
    X_engineered, y, iris_data, feature_names = load_and_prepare_data(ModelType.DECISION_TREE)

    # Model configuration
    model_params = {
        "max_depth": 3,
        "random_state": 42,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini",
    }

    # Perform comprehensive validation
    model, validation_results = perform_comprehensive_validation(
        DecisionTreeClassifier, model_params, X_engineered, y, cv_folds=10, cv_repeats=10
    )

    # Make predictions on full dataset for evaluation
    y_pred = model.predict(X_engineered)
    results = evaluate_model(y, y_pred, iris_data, X_engineered)

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=len(feature_names))

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "decision_tree",
        "experiment_type": "comprehensive",
        "hyperparameters": model_params,
        "feature_importance": sorted_importance,
        "training_accuracy": validation_results["training_accuracy"],
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "loocv_accuracy": validation_results["loocv"]["accuracy"],
        "loocv_std": validation_results["loocv"]["std"],
        "repeated_cv_accuracy": validation_results["repeated_kfold"]["accuracy"],
        "repeated_cv_std": validation_results["repeated_kfold"]["std"],
    }

    # Store comprehensive validation data
    results["validation_results"] = validation_results

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "decision_tree_comprehensive",
        "timestamp": datetime.now().isoformat(),
        "model_type": "iris_classifier",
        "approach": "comprehensive_validation",
        "validation_method": "LOOCV + Repeated K-Fold",
    }

    results["performance_metrics"] = {
        "overall_accuracy": results["overall_accuracy"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "misclassification_count": len(results["misclassifications"]),
    }

    # Log performance and save results
    log_performance_summary(results, "decision_tree_comprehensive")
    save_experiment_results(results, "decision_tree", "comprehensive", model)

    logger.info(
        "Experiment completed",
        loocv_accuracy=validation_results["loocv"]["accuracy"],
        training_accuracy=validation_results["training_accuracy"],
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
