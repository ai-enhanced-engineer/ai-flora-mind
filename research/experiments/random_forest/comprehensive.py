"""
Random Forest Iris Classifier - Comprehensive Experiment

This module implements the comprehensive validation experiment for Random Forest on the Iris dataset.
Uses Leave-One-Out Cross-Validation (LOOCV) and repeated k-fold with OOB scoring.

Configuration:
- 300 trees for comprehensive validation
- Out-of-bag (OOB) scoring enabled
- Full dataset utilization with multiple validation strategies
"""

import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, cross_val_score

from ml_production_service.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run Random Forest comprehensive experiment with full dataset validation."""
    logger.info("Starting Random Forest Comprehensive Experiment")

    # Load Iris dataset
    logger.info("Loading Iris dataset")
    X_original, y, iris_data = load_iris_data()
    feature_names_original = iris_data.feature_names
    logger.info(
        "Dataset loaded successfully",
        features=feature_names_original,
        target_classes=list(set(y)),
        total_samples=len(X_original),
    )

    # Log petal feature statistics for context
    petal_length_idx = feature_names_original.index("petal length (cm)")
    petal_width_idx = feature_names_original.index("petal width (cm)")
    petal_length_range = f"{X_original[:, petal_length_idx].min():.2f}-{X_original[:, petal_width_idx].max():.2f}"
    petal_width_range = f"{X_original[:, petal_width_idx].min():.2f}-{X_original[:, petal_width_idx].max():.2f}"
    logger.info("Petal feature statistics", petal_length_range=petal_length_range, petal_width_range=petal_width_range)

    # Engineer features for Random Forest (all 14 features)
    logger.info(
        "Engineering features for Random Forest",
        n_samples=len(X_original),
        original_features=len(feature_names_original),
    )
    X_engineered, all_feature_names = engineer_features(X_original, feature_names_original, ModelType.RANDOM_FOREST)
    logger.info("Feature engineering completed", total_features=X_engineered.shape[1])

    # Configuration for comprehensive validation
    cv_folds = 10
    cv_repeats = 10
    logger.info("Running comprehensive validation experiment", cv_folds=cv_folds, cv_repeats=cv_repeats)

    # Step 1: Train model on full dataset
    logger.info("Step 1: Training model on full dataset", total_samples=len(X_engineered))
    n_estimators = 300  # More trees for comprehensive validation
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        oob_score=True,  # Enable out-of-bag scoring for additional validation
    )

    logger.info(
        "Training Random Forest classifier",
        features=len(all_feature_names),
        n_estimators=n_estimators,
        samples=len(X_engineered),
    )

    model.fit(X_engineered, y)

    # Log training completion with OOB score
    training_accuracy = model.score(X_engineered, y)
    oob_score = model.oob_score_
    logger.info(
        "Random Forest training completed",
        training_accuracy=training_accuracy,
        oob_score=oob_score,
        n_estimators=model.n_estimators,
    )

    # Step 2: Leave-One-Out Cross-Validation with prediction tracking
    logger.info("Step 2: Performing Leave-One-Out Cross-Validation with prediction tracking")
    loocv = LeaveOneOut()
    loocv_predictions = []
    loocv_true_labels = []
    loocv_indices = []

    # Manually perform LOOCV to track predictions
    for train_idx, test_idx in loocv.split(X_engineered):
        # Create and train model for this fold
        fold_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
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

    logger.info(
        "LOOCV completed",
        loocv_accuracy=float(loocv_scores.mean()),
        loocv_std=float(loocv_scores.std()),
        total_iterations=len(loocv_scores),
        correct_predictions=int(loocv_scores.sum()),
    )

    # Step 3: Repeated Stratified K-Fold Cross-Validation
    logger.info("Step 3: Performing Repeated Stratified K-Fold Cross-Validation")
    rskf = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)
    repeated_scores = cross_val_score(
        RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        X_engineered,
        y,
        cv=rskf,
        scoring="accuracy",
    )
    repeated_scores = np.array(repeated_scores)  # Ensure it's a numpy array
    logger.info(
        "Repeated K-Fold completed",
        repeated_cv_accuracy=float(repeated_scores.mean()),
        repeated_cv_std=float(repeated_scores.std()),
        total_iterations=len(repeated_scores),
    )

    # Extract feature importance for EDA validation
    importance_scores = model.feature_importances_
    feature_importance = dict(zip(all_feature_names, importance_scores))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    top_features = list(sorted_importance.keys())[:5]
    top_feature = top_features[0]
    top_importance = list(sorted_importance.values())[0]
    logger.info("Top 5 most important features", features=top_features)
    logger.info("Feature importance extracted", top_feature=top_feature, top_importance=top_importance)

    # Validate EDA findings: petal features should dominate
    petal_features = [f for f in top_features if "petal" in f.lower()]
    if len(petal_features) >= 2:
        logger.info("EDA validation successful", insight="Petal features dominate as expected")
    else:
        logger.warning(
            "EDA validation concern", insight="Petal features not dominating as expected", top_features=top_features
        )

    # Use LOOCV predictions to generate proper evaluation metrics
    loocv_eval_results = evaluate_model(loocv_true_labels_array, loocv_predictions_array, iris_data, X_engineered)

    # Build comprehensive results
    results = {
        "predictions": model.predict(X_engineered).tolist(),
        "true_labels": y.tolist(),
        "algorithm_details": {
            "experiment_type": "comprehensive",
            "n_estimators": n_estimators,
            "features_used": all_feature_names,
            "feature_importance": sorted_importance,
            "training_samples": len(X_engineered),
            "training_accuracy": float(training_accuracy),
            "oob_score": float(oob_score),
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
            "training_accuracy": float(training_accuracy),
            "oob_score": float(oob_score),
            "total_samples": len(X_engineered),
            "samples_per_class": len(X_engineered) // 3,  # Assuming equal distribution
        },
        # Add classification results
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

    # Performance validation against EDA expectations
    loocv_accuracy = float(loocv_scores.mean())
    if loocv_accuracy >= 0.98:
        logger.info(
            "Target accuracy achieved",
            achieved_accuracy=loocv_accuracy,
            insight=f"Excellent accuracy: {loocv_accuracy:.3f}",
            target_range="0.98-0.99",
        )
    elif loocv_accuracy >= 0.95:
        logger.warning(
            "Target accuracy close but not met",
            achieved_accuracy=loocv_accuracy,
            target_range="0.98-0.99",
            warning=f"Accuracy {loocv_accuracy:.3f} below target",
        )
    else:
        logger.error(
            "Target accuracy not achieved",
            achieved_accuracy=loocv_accuracy,
            target_range="0.98-0.99",
            error=f"Significantly below target: {loocv_accuracy:.3f}",
        )

    # Log performance summary
    experiment_name = "random_forest_comprehensive"
    logger.info("Step 4: Evaluating model performance")
    log_performance_summary(results, experiment_name)

    # Final validation summary
    logger.info(
        "Comprehensive validation completed",
        loocv_accuracy=results["validation_results"]["loocv_accuracy"],
        repeated_cv_accuracy=results["validation_results"]["repeated_cv_accuracy"],
        training_accuracy=results["algorithm_details"]["training_accuracy"],
        oob_score=results["algorithm_details"]["oob_score"],
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

    # Log model details
    model_size = os.path.getsize(model_path)
    logger.info("Model saved", path=model_path, size_bytes=model_size, n_estimators=model.n_estimators)

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
        f"Random Forest Comprehensive Experiment: {results['performance_metrics']['overall_accuracy']:.1%} LOOCV accuracy. "
        f"Training accuracy: {training_accuracy:.1%}. "
        f"OOB score: {oob_score:.1%}. "
        f"Top features: {', '.join(top_features[:3])}"
    )

    logger.info(
        "Random Forest comprehensive experiment completed successfully",
        loocv_accuracy=f"{results['performance_metrics']['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
