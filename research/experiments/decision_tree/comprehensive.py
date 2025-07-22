"""
Decision Tree Iris Classifier - Comprehensive Experiment

This module implements the comprehensive validation experiment for decision tree on the Iris dataset.
Uses Leave-One-Out Cross-Validation (LOOCV) and repeated k-fold to maximize data usage.

Configuration:
- Max depth: 3 (maintains human readability)
- Full dataset utilization with cross-validation
- Multiple validation strategies for robust assessment
"""

import os
from datetime import datetime
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from ai_flora_mind.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run decision tree comprehensive experiment with full dataset validation."""
    logger.info("Starting Decision Tree Comprehensive Experiment")

    # Load data
    X, y_true, iris_data = load_iris_data()

    # Engineer features for decision tree
    X_enhanced, feature_names = engineer_features(X, list(iris_data.feature_names), ModelType.DECISION_TREE)
    logger.info(
        "Feature engineering completed",
        original_features=len(iris_data.feature_names),
        total_features=len(feature_names),
        engineered_features=feature_names,
    )

    # Step 1: Train final model on full dataset
    logger.info("Step 1: Training model on full dataset", total_samples=len(X_enhanced))
    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
    )
    model.fit(X_enhanced, y_true)

    logger.info(
        "Model training completed",
        tree_depth=model.get_depth(),
        n_leaves=model.get_n_leaves(),
        training_accuracy=model.score(X_enhanced, y_true),
    )

    # Step 2: Leave-One-Out Cross-Validation with prediction tracking
    logger.info("Step 2: Performing Leave-One-Out Cross-Validation with prediction tracking")
    loocv = LeaveOneOut()
    loocv_predictions = []
    loocv_true_labels = []
    loocv_indices = []

    # Manually perform LOOCV to track predictions
    for train_idx, test_idx in loocv.split(X_enhanced):
        # Create and train model for this fold
        fold_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        fold_model.fit(X_enhanced[train_idx], y_true[train_idx])

        # Make prediction for the single test sample
        pred = fold_model.predict(X_enhanced[test_idx])[0]
        true_label = y_true[test_idx][0]

        # Track predictions and indices
        loocv_predictions.append(pred)
        loocv_true_labels.append(true_label)
        loocv_indices.append(test_idx[0])

    # Convert to numpy arrays
    loocv_predictions_array = np.array(loocv_predictions)
    loocv_true_labels_array = np.array(loocv_true_labels)
    loocv_scores = (loocv_predictions_array == loocv_true_labels_array).astype(float)

    logger.info(
        "LOOCV completed",
        loocv_accuracy=float(loocv_scores.mean()),
        loocv_std=float(loocv_scores.std()),
        total_iterations=len(loocv_scores),
        correct_predictions=int(loocv_scores.sum()),
    )

    # Step 3: Repeated Stratified K-Fold Cross-Validation
    logger.info("Step 3: Performing Repeated Stratified K-Fold Cross-Validation")
    cv_folds = 10
    cv_repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)
    repeated_scores = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=42), X_enhanced, y_true, cv=rskf, scoring="accuracy"
    )
    repeated_scores = np.array(repeated_scores)  # Ensure it's a numpy array
    logger.info(
        "Repeated K-Fold completed",
        repeated_cv_accuracy=float(repeated_scores.mean()),
        repeated_cv_std=float(repeated_scores.std()),
        total_iterations=len(repeated_scores),
    )

    # Extract feature importance
    feature_importance = {
        name: float(importance) for name, importance in zip(feature_names, model.feature_importances_)
    }
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Use LOOCV predictions to generate proper evaluation metrics
    loocv_eval_results = evaluate_model(loocv_true_labels_array, loocv_predictions_array, iris_data, X_enhanced)

    # Compile comprehensive results
    results: Dict[str, Any] = {
        "algorithm_details": {
            "algorithm_type": "decision_tree",
            "experiment_type": "comprehensive",
            "max_depth": 3,
            "features_used": feature_names,
            "feature_importance": sorted_importance,
            "training_samples": len(X_enhanced),
            "tree_depth": int(model.get_depth()),
            "n_leaves": int(model.get_n_leaves()),
            "decision_rules": export_text(model, feature_names=feature_names),
        },
        "validation_results": {
            "loocv_accuracy": float(loocv_scores.mean()),
            "loocv_std": float(loocv_scores.std()),
            "loocv_scores": loocv_scores.tolist(),
            "repeated_cv_accuracy": float(repeated_scores.mean()),
            "repeated_cv_std": float(repeated_scores.std()),
            "repeated_cv_scores": repeated_scores.tolist(),
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "total_cv_iterations": len(repeated_scores),
        },
        "full_dataset_training": {
            "training_accuracy": float(model.score(X_enhanced, y_true)),
            "total_samples": len(X_enhanced),
            "samples_per_class": len(X_enhanced) // 3,  # Assuming equal distribution
        },
        # Add classification results from LOOCV evaluation
        "per_class_accuracy": loocv_eval_results["per_class_accuracy"],
        "confusion_matrix": loocv_eval_results["confusion_matrix"],
        "classification_report": loocv_eval_results["classification_report"],
    }

    # Add performance metrics section
    results["performance_metrics"] = {
        "overall_accuracy": loocv_eval_results["overall_accuracy"],
        "total_samples": loocv_eval_results["total_samples"],
        "correct_predictions": loocv_eval_results["correct_predictions"],
        "misclassification_count": len(loocv_eval_results["misclassifications"]),
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
    experiment_name = "decision_tree_comprehensive"
    log_performance_summary(results, experiment_name)

    # Log validation results
    val_results = results["validation_results"]
    logger.info(
        "Comprehensive validation completed",
        loocv_accuracy=val_results["loocv_accuracy"],
        repeated_cv_accuracy=val_results["repeated_cv_accuracy"],
        training_accuracy=results["full_dataset_training"]["training_accuracy"],
    )

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
    results["model_path"] = model_path

    # Save results
    results_path = os.path.join(results_dir, results_filename)
    import json

    # Convert any numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
        f"Decision Tree Comprehensive Experiment: {results['performance_metrics']['overall_accuracy']:.1%} LOOCV accuracy. "
        f"Training accuracy: {results['full_dataset_training']['training_accuracy']:.1%}. "
        f"Repeated CV: {val_results['repeated_cv_accuracy']:.1%}. "
        f"Top features: {', '.join(list(sorted_importance.keys())[:3])}"
    )

    logger.info(
        "Decision tree comprehensive experiment completed successfully",
        loocv_accuracy=f"{results['performance_metrics']['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
