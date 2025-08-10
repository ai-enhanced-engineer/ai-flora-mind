"""
Random Forest Iris Classifier - Comprehensive Experiment

This module implements the comprehensive experiment for Random Forest on the Iris dataset.
Uses OOB (Out-of-Bag) scoring and comprehensive cross-validation strategies.

Configuration:
- 300 trees for comprehensive validation
- All 14 features (4 original + 10 engineered)
- OOB + LOOCV validation
"""

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

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
    """Run Random Forest comprehensive experiment with full dataset validation."""
    logger.info("Starting Random Forest Comprehensive Experiment")

    # Load and prepare data
    X_engineered, y, iris_data, feature_names = load_and_prepare_data(ModelType.RANDOM_FOREST)

    # Model configuration with OOB scoring
    model_params = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "oob_score": True,  # Enable out-of-bag scoring for additional validation
    }

    # Perform comprehensive validation
    model, validation_results = perform_comprehensive_validation(
        RandomForestClassifier, model_params, X_engineered, y, cv_folds=10, cv_repeats=10
    )

    # Make predictions on full dataset for evaluation
    y_pred = model.predict(X_engineered)
    results = evaluate_model(y, y_pred, iris_data, X_engineered)

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=10)

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "random_forest",
        "experiment_type": "comprehensive",
        "hyperparameters": model_params,
        "feature_importance": sorted_importance,
        "training_accuracy": validation_results["training_accuracy"],
        "oob_score": validation_results.get("oob_score"),
        "loocv_accuracy": validation_results["loocv"]["accuracy"],
        "loocv_std": validation_results["loocv"]["std"],
        "repeated_cv_accuracy": validation_results["repeated_kfold"]["accuracy"],
        "repeated_cv_std": validation_results["repeated_kfold"]["std"],
    }

    # Store comprehensive validation data
    results["validation_results"] = validation_results

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "random_forest_comprehensive",
        "timestamp": datetime.now().isoformat(),
        "model_type": "iris_classifier",
        "approach": "comprehensive_validation",
        "validation_method": "OOB + LOOCV + Repeated K-Fold",
    }

    results["performance_metrics"] = {
        "overall_accuracy": results["overall_accuracy"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "misclassification_count": len(results["misclassifications"]),
    }

    # Log performance and save results
    log_performance_summary(results, "random_forest_comprehensive")
    save_experiment_results(results, "random_forest", "comprehensive", model)

    logger.info(
        "Experiment completed",
        oob_score=validation_results.get("oob_score"),
        loocv_accuracy=validation_results["loocv"]["accuracy"],
        training_accuracy=validation_results["training_accuracy"],
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
