"""
Random Forest Iris Classifier - Split Experiment

This module implements the split experiment for Random Forest on the Iris dataset.
Uses traditional 70/30 train/test split methodology with ensemble learning.

Configuration:
- 200 trees for moderate ensemble size
- All 14 features (4 original + 10 engineered)
- Expected accuracy: 98-99%
"""

import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from ai_flora_mind.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run Random Forest split experiment with traditional train/test methodology."""
    logger.info("Starting Random Forest Split Experiment")

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
    petal_length_range = f"{X_original[:, petal_length_idx].min():.2f}-{X_original[:, petal_length_idx].max():.2f}"
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

    # Split data for training
    test_size = 0.3
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Splitting data for training", test_size=test_size, random_state=random_state)

    # Train Random Forest classifier
    logger.info("Training Random Forest classifier")
    n_estimators = 200  # Moderate number based on EDA recommendations
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all cores for faster training
    )

    logger.info(
        "Training Random Forest classifier",
        features=len(all_feature_names),
        n_estimators=n_estimators,
        samples=len(X_train),
    )

    model.fit(X_train, y_train)

    # Log training completion
    training_accuracy = model.score(X_train, y_train)
    logger.info(
        "Random Forest training completed",
        training_accuracy=training_accuracy,
        n_estimators=model.n_estimators,
        oob_score=getattr(model, "oob_score_", None),
    )

    # Make predictions on test set
    logger.info("Making predictions on test set")
    y_pred = model.predict(X_test)

    # Use evaluate_model to get proper metrics including confusion matrix
    results = evaluate_model(y_test, y_pred, iris_data, X_test)

    # Extract and rank feature importance
    importance_scores = model.feature_importances_
    feature_importance = dict(zip(all_feature_names, importance_scores))

    # Sort by importance (descending)
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Log top features for validation
    top_features = list(sorted_importance.keys())[:5]
    logger.info("Top 5 most important features", features=top_features)

    # Validate EDA findings: petal features should dominate
    petal_features = [f for f in top_features if "petal" in f.lower()]
    if len(petal_features) >= 2:
        logger.info("EDA validation successful", insight="Petal features dominate as expected")
    else:
        logger.warning(
            "EDA validation concern", insight="Petal features not dominating as expected", top_features=top_features
        )

    # Add metadata sections to match original structure
    results["experiment_metadata"] = {
        "experiment_name": "random_forest_split",
        "timestamp": datetime.now().isoformat(),
        "model_type": "iris_classifier",
        "approach": "baseline",
    }

    results["performance_metrics"] = {
        "overall_accuracy": results["overall_accuracy"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "misclassification_count": len(results["misclassifications"]),
    }

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "experiment_type": "split",
        "n_estimators": n_estimators,
        "features_used": all_feature_names,
        "feature_importance": sorted_importance,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "training_accuracy": float(training_accuracy),
    }

    # Cross-validation on full dataset for comparison
    logger.info("Performing cross-validation on full dataset")
    cv_scores = cross_val_score(model, X_engineered, y, cv=5, scoring="accuracy")
    cv_scores = np.array(cv_scores)  # Ensure it's a numpy array
    results["cross_validation"] = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }

    # Log performance summary
    experiment_name = "random_forest_split"
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
        f"Random Forest Split Experiment: {results['overall_accuracy']:.1%} test accuracy "
        f"({results['correct_predictions']}/{results['total_samples']} correct). "
        f"Training accuracy: {results['algorithm_details']['training_accuracy']:.1%}. "
        f"CV mean: {results['cross_validation']['cv_mean']:.1%}. "
        f"Top features: {', '.join(top_features[:3])}"
    )

    logger.info(
        "Random Forest split experiment completed successfully",
        accuracy=f"{results['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
