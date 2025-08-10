"""
XGBoost Iris Classifier - Optimized Experiment

This module implements the optimized experiment for XGBoost on the Iris dataset.
Heavy regularization and hyperparameter tuning to prevent overfitting.

Configuration:
- Aggressive regularization (L1, L2, gamma, min_child_weight)
- Early stopping for overfitting prevention
- Lower learning rate and careful hyperparameter tuning
"""

import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, train_test_split

from ml_production_service.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run XGBoost optimized experiment with heavy regularization."""
    logger.info("Starting XGBoost Optimized Experiment")

    # Load data
    X, y_names, iris_data = load_iris_data()
    feature_names = iris_data.feature_names
    # XGBoost requires numeric targets
    y = iris_data.target
    logger.info(
        "Data loaded successfully", samples=len(X), features=len(feature_names), classes=len(iris_data.target_names)
    )

    # Engineer targeted features for XGBoost
    X_engineered, engineered_feature_names = engineer_features(X, feature_names, ModelType.XGBOOST)
    logger.info(
        "Feature engineering completed",
        original_features=len(feature_names),
        engineered_features=len(engineered_feature_names),
        total_features=X_engineered.shape[1],
    )

    # Split data for early stopping validation
    X_train, X_val, y_train, y_val = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)

    # Create highly optimized XGBoost classifier focused on overfitting prevention
    model = xgb.XGBClassifier(
        n_estimators=150,  # Moderate tree count
        max_depth=3,  # Shallow trees for small dataset
        learning_rate=0.05,  # Lower learning rate for stability
        subsample=0.7,  # Aggressive subsampling
        colsample_bytree=0.7,  # Aggressive column subsampling
        min_child_weight=3,  # Higher minimum child weight
        gamma=0.1,  # Minimum split loss
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=20,  # Aggressive early stopping
    )

    # Train with early stopping validation
    logger.info("Training XGBoost model with aggressive regularization and early stopping")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    logger.info(
        "Model training completed",
        best_iteration=model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators,
        trees_used=model.best_ntree_limit if hasattr(model, "best_ntree_limit") else model.n_estimators,
    )

    # Comprehensive cross-validation with prediction tracking
    logger.info("Running optimized LOOCV validation with prediction tracking")
    loo = LeaveOneOut()
    loocv_predictions = []
    loocv_true_labels = []
    loocv_indices = []

    # Manually perform LOOCV to track predictions
    for train_idx, test_idx in loo.split(X_engineered):
        # Create and train model for this fold
        fold_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric="mlogloss",
            # No early stopping for LOOCV
        )
        fold_model.fit(X_engineered[train_idx], y[train_idx])

        # Make prediction for the single test sample
        pred = fold_model.predict(X_engineered[test_idx])[0]
        true_label = y[test_idx][0]

        # Track predictions and indices
        loocv_predictions.append(pred)
        loocv_true_labels.append(true_label)
        loocv_indices.append(test_idx[0])

    # Convert to numpy arrays
    loocv_predictions_array = np.array(loocv_predictions)
    loocv_true_labels_array = np.array(loocv_true_labels)
    loocv_scores = (loocv_predictions_array == loocv_true_labels_array).astype(float)
    loocv_mean = np.mean(loocv_scores)
    loocv_std = np.std(loocv_scores)

    logger.info(
        "LOOCV completed",
        loocv_accuracy=f"{loocv_mean:.1%}",
        loocv_std=f"{loocv_std:.1%}",
        iterations=len(loocv_scores),
        correct_predictions=int(loocv_scores.sum()),
    )

    # Training accuracy for overfitting assessment
    training_accuracy = model.score(X_engineered, y)
    overfitting_gap = abs(training_accuracy - loocv_mean)

    # Overfitting assessment
    if overfitting_gap <= 0.02:  # 2% threshold
        overfitting_status = "Overfitting successfully controlled"
    elif overfitting_gap <= 0.04:  # 4% threshold
        overfitting_status = "Mild overfitting detected"
    else:
        overfitting_status = "Overfitting concern - consider more regularization"

    logger.info(
        "Overfitting analysis",
        training_accuracy=f"{training_accuracy:.1%}",
        validation_accuracy=f"{loocv_mean:.1%}",
        gap=f"{overfitting_gap:.1%}",
        status=overfitting_status,
    )

    # Feature importance analysis - convert to native float for JSON serialization
    feature_importance = dict(zip(engineered_feature_names, [float(imp) for imp in model.feature_importances_]))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Generate proper evaluation metrics from LOOCV predictions
    loocv_eval_results = evaluate_model(loocv_true_labels_array, loocv_predictions_array, iris_data, X_engineered)

    # Create results structure
    results = {
        "experiment_metadata": {
            "experiment_name": "xgboost_optimized",
            "timestamp": datetime.now().isoformat(),
            "model_type": "iris_classifier",
            "approach": "baseline",
        },
        "performance_metrics": {
            "overall_accuracy": loocv_mean,
            "total_samples": len(X_engineered),
            "correct_predictions": int(loocv_mean * len(X_engineered)),
            "misclassification_count": len(X_engineered) - int(loocv_mean * len(X_engineered)),
        },
        "validation_metrics": {
            "loocv_accuracy": loocv_mean,
            "loocv_std": loocv_std,
        },
        "algorithm_details": {
            "experiment_type": "optimized",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "min_child_weight": model.min_child_weight,
            "gamma": model.gamma,
            "reg_alpha": model.reg_alpha,
            "reg_lambda": model.reg_lambda,
            "features_used": engineered_feature_names,
            "feature_importance": sorted_importance,
            "training_samples": len(X_engineered),
            "training_accuracy": training_accuracy,
            "overfitting_gap": overfitting_gap,
            "overfitting_status": overfitting_status,
        },
        # Add classification results from LOOCV evaluation
        "per_class_accuracy": loocv_eval_results["per_class_accuracy"],
        "classification_report": loocv_eval_results["classification_report"],
        "confusion_matrix": loocv_eval_results["confusion_matrix"],
        # Store raw LOOCV data
        "loocv_predictions": loocv_predictions,
        "loocv_true_labels": loocv_true_labels,
        "loocv_indices": loocv_indices,
    }

    # Fix misclassification indices to match original dataset
    misclassifications = []
    for misc in loocv_eval_results["misclassifications"]:
        # Map from LOOCV index to original dataset index
        original_idx = loocv_indices[misc["index"]]
        misc_fixed = misc.copy()
        misc_fixed["index"] = int(original_idx)  # Convert numpy int to Python int
        misclassifications.append(misc_fixed)

    results["misclassifications"] = misclassifications

    # Log performance summary
    experiment_name = "xgboost_optimized"
    log_performance_summary(results, experiment_name)

    # Save model and results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    model_filename = f"{experiment_name}_{timestamp}.joblib"
    results_filename = f"{experiment_name}_{timestamp}.json"

    # Use centralized directory constants
    results_dir = RESULTS_DIR
    models_dir = MODELS_DIR

    # Save model
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, model_path)
    logger.info("Model saved", model_path=model_path)

    # Save results
    results_path = os.path.join(results_dir, results_filename)
    import json

    # Convert any numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    json_safe_results = convert_numpy_to_list(results)

    with open(results_path, "w") as f:
        json.dump(json_safe_results, f, indent=2)
    logger.info("Results saved", results_path=results_path)

    # Final summary
    summary = (
        f"XGBoost Optimized Experiment: {results['performance_metrics']['overall_accuracy']:.1%} LOOCV accuracy. "
        f"Training accuracy: {results['algorithm_details']['training_accuracy']:.1%}. "
        f"Overfitting gap: {results['algorithm_details']['overfitting_gap']:.1%} ({results['algorithm_details']['overfitting_status'].lower()}). "
        f"Top features: {', '.join(list(results['algorithm_details']['feature_importance'].keys())[:3])}"
    )

    logger.info(
        "XGBoost optimized experiment completed successfully",
        accuracy=f"{results['performance_metrics']['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
