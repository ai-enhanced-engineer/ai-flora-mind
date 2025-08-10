"""
XGBoost Iris Classifier - Comprehensive Experiment

This module implements the comprehensive experiment for XGBoost on the Iris dataset.
Uses Leave-One-Out Cross-Validation (LOOCV) and repeated k-fold validation.

Configuration:
- Moderate hyperparameters (max_depth=4, learning_rate=0.1)
- 200 trees with increased depth for comprehensive evaluation
- Full validation suite: LOOCV + repeated k-fold
"""

from datetime import datetime

import numpy as np
import xgboost as xgb

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
    """Run XGBoost comprehensive experiment with full dataset validation."""
    logger.info("Starting XGBoost Comprehensive Experiment")

    # Load and prepare data (XGBoost needs numeric labels)
    X_engineered, y, iris_data, feature_names = load_and_prepare_data(ModelType.XGBOOST, numeric_labels=True)

    # Model configuration with moderate tuning
    model_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    }

    # Perform comprehensive validation
    model, validation_results = perform_comprehensive_validation(
        xgb.XGBClassifier, model_params, X_engineered, y, cv_folds=10, cv_repeats=10
    )

    # Make predictions on full dataset for evaluation
    y_pred = model.predict(X_engineered)

    # Convert to string labels for evaluation
    y_names = np.array([iris_data.target_names[i] for i in y])
    y_pred_names = np.array([iris_data.target_names[i] for i in y_pred])

    results = evaluate_model(y_names, y_pred_names, iris_data, X_engineered)

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=10)

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "xgboost",
        "experiment_type": "comprehensive",
        "hyperparameters": model_params,
        "feature_importance": sorted_importance,
        "training_accuracy": validation_results["training_accuracy"],
        "loocv_accuracy": validation_results["loocv"]["accuracy"],
        "loocv_std": validation_results["loocv"]["std"],
        "repeated_cv_accuracy": validation_results["repeated_kfold"]["accuracy"],
        "repeated_cv_std": validation_results["repeated_kfold"]["std"],
    }

    # Store comprehensive validation data
    results["validation_results"] = validation_results

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "xgboost_comprehensive",
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
    log_performance_summary(results, "xgboost_comprehensive")
    save_experiment_results(results, "xgboost", "comprehensive", model)

    logger.info(
        "Experiment completed",
        loocv_accuracy=validation_results["loocv"]["accuracy"],
        training_accuracy=validation_results["training_accuracy"],
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
