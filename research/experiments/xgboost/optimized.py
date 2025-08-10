"""
XGBoost Iris Classifier - Optimized Experiment

This module implements the optimized experiment for XGBoost on the Iris dataset.
Heavy regularization and hyperparameter tuning to prevent overfitting.

Configuration:
- Aggressive regularization (L1, L2, gamma, min_child_weight)
- Early stopping for overfitting prevention
- Lower learning rate and careful hyperparameter tuning
"""

from datetime import datetime
from typing import Any, List

import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, train_test_split

from ml_production_service.logging import get_logger
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.base import (
    extract_feature_importance,
    load_and_prepare_data,
    save_experiment_results,
)
from research.features import ModelType

logger = get_logger(__name__)


def main() -> None:
    """Run XGBoost optimized experiment with heavy regularization."""
    logger.info("Starting XGBoost Optimized Experiment")

    # Load and prepare data (XGBoost needs numeric labels)
    X_engineered, y, iris_data, feature_names = load_and_prepare_data(ModelType.XGBOOST, numeric_labels=True)

    # Split data for early stopping (validation set)
    X_train, X_val, y_train, y_val = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)

    # Configure XGBoost with heavy regularization
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,  # Reduced back for simplicity
        learning_rate=0.05,  # Halved for better generalization
        subsample=0.7,  # More aggressive subsampling
        colsample_bytree=0.7,
        min_child_weight=3,  # Regularization
        gamma=0.1,  # Minimum loss reduction
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        early_stopping_rounds=20,  # Stop if no improvement
    )

    # Train with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # LOOCV validation on full dataset
    loocv = LeaveOneOut()
    loocv_predictions: List[Any] = []
    loocv_true_labels: List[Any] = []

    for train_idx, test_idx in loocv.split(X_engineered):
        # Create new model for each fold
        fold_model = xgb.XGBClassifier(**model.get_params())
        fold_model.fit(X_engineered[train_idx], y[train_idx], verbose=False)
        pred = fold_model.predict(X_engineered[test_idx])[0]
        loocv_predictions.append(pred)
        loocv_true_labels.append(y[test_idx][0])

    loocv_predictions_array = np.array(loocv_predictions)
    loocv_true_labels_array = np.array(loocv_true_labels)
    loocv_accuracy = float((loocv_predictions_array == loocv_true_labels_array).mean())

    # Make final predictions on full dataset
    y_pred = model.predict(X_engineered)

    # Convert to string labels for evaluation
    y_names = np.array([iris_data.target_names[i] for i in y])
    y_pred_names = np.array([iris_data.target_names[i] for i in y_pred])

    results = evaluate_model(y_names, y_pred_names, iris_data, X_engineered)

    # Calculate training accuracy
    training_accuracy = model.score(X_engineered, y)

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=10)

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "xgboost",
        "experiment_type": "optimized",
        "hyperparameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "min_child_weight": model.min_child_weight,
            "gamma": model.gamma,
            "reg_alpha": model.reg_alpha,
            "reg_lambda": model.reg_lambda,
        },
        "feature_importance": sorted_importance,
        "training_accuracy": training_accuracy,
        "loocv_accuracy": loocv_accuracy,
        "early_stopping_rounds": getattr(model, "best_iteration", None),
        "regularization_impact": {
            "learning_rate_reduction": "50% (0.1 → 0.05)",
            "regularization_params": "gamma=0.1, alpha=0.1, lambda=0.1",
            "subsampling": "70% rows, 70% columns",
        },
    }

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "xgboost_optimized",
        "timestamp": datetime.now().isoformat(),
        "model_type": "iris_classifier",
        "approach": "heavily_regularized",
        "validation_method": "LOOCV + Early Stopping",
    }

    results["performance_metrics"] = {
        "overall_accuracy": results["overall_accuracy"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "misclassification_count": len(results["misclassifications"]),
    }

    # Log performance and save results
    log_performance_summary(results, "xgboost_optimized")
    save_experiment_results(results, "xgboost", "optimized", model)

    logger.info(
        "Experiment completed",
        loocv_accuracy=loocv_accuracy,
        training_accuracy=training_accuracy,
        early_stopping_round=getattr(model, "best_iteration", None),
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
