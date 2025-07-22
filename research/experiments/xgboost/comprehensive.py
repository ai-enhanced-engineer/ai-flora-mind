"""
XGBoost Iris Classifier - Comprehensive Experiment

This module implements the comprehensive validation experiment for XGBoost on the Iris dataset.
Uses Leave-One-Out Cross-Validation (LOOCV) and repeated k-fold to maximize data usage.

Configuration:
- Moderate complexity (max_depth=4, 200 trees)
- Full dataset utilization with cross-validation
- Comprehensive feature importance analysis
"""

import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, cross_val_score

from ai_flora_mind.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run XGBoost comprehensive experiment with full dataset validation."""
    logger.info("Starting XGBoost Comprehensive Experiment")

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

    # Create XGBoost classifier with moderate complexity
    model = xgb.XGBClassifier(
        n_estimators=200,  # More trees for full dataset
        max_depth=4,  # Slightly deeper for pattern capture
        learning_rate=0.1,  # Conservative learning rate
        subsample=0.8,  # Feature subsampling
        colsample_bytree=0.8,  # Column subsampling
        random_state=42,
        eval_metric="mlogloss",
    )

    # Train on full dataset
    logger.info("Training XGBoost model on full dataset")
    model.fit(X_engineered, y)

    # Comprehensive cross-validation with prediction tracking
    logger.info("Running Leave-One-Out Cross-Validation with prediction tracking")
    loo = LeaveOneOut()
    loocv_predictions = []
    loocv_true_labels = []
    loocv_indices = []

    # Manually perform LOOCV to track predictions
    for train_idx, test_idx in loo.split(X_engineered):
        # Create and train model for this fold
        fold_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
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

    # Repeated stratified k-fold for robustness
    logger.info("Running Repeated Stratified K-Fold Cross-Validation")
    repeated_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    repeated_scores = cross_val_score(model, X_engineered, y, cv=repeated_cv, scoring="accuracy")
    repeated_mean = np.mean(repeated_scores)
    repeated_std = np.std(repeated_scores)

    logger.info(
        "Repeated CV completed",
        repeated_accuracy=f"{repeated_mean:.1%}",
        repeated_std=f"{repeated_std:.1%}",
        iterations=len(repeated_scores),
    )

    # Training accuracy for overfitting assessment
    training_accuracy = model.score(X_engineered, y)

    # Feature importance analysis - convert to native float for JSON serialization
    feature_importance = dict(zip(engineered_feature_names, [float(imp) for imp in model.feature_importances_]))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Generate proper evaluation metrics from LOOCV predictions
    loocv_eval_results = evaluate_model(loocv_true_labels_array, loocv_predictions_array, iris_data, X_engineered)

    # Create results structure
    results = {
        "experiment_metadata": {
            "experiment_name": "xgboost_comprehensive",
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
            "repeated_cv_accuracy": repeated_mean,
            "repeated_cv_std": repeated_std,
        },
        "algorithm_details": {
            "experiment_type": "comprehensive",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "features_used": engineered_feature_names,
            "feature_importance": sorted_importance,
            "training_samples": len(X_engineered),
            "training_accuracy": training_accuracy,
            "overfitting_gap": abs(training_accuracy - loocv_mean),
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
    experiment_name = "xgboost_comprehensive"
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
        f"XGBoost Comprehensive Experiment: {results['performance_metrics']['overall_accuracy']:.1%} LOOCV accuracy. "
        f"Training accuracy: {results['algorithm_details']['training_accuracy']:.1%}. "
        f"Overfitting gap: {results['algorithm_details']['overfitting_gap']:.1%}. "
        f"Top features: {', '.join(list(results['algorithm_details']['feature_importance'].keys())[:3])}"
    )

    logger.info(
        "XGBoost comprehensive experiment completed successfully",
        accuracy=f"{results['performance_metrics']['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
