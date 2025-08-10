"""
Random Forest Iris Classifier - Split Experiment

This module implements the split experiment for Random Forest on the Iris dataset.
Uses traditional 70/30 train/test split methodology with ensemble learning.

Configuration:
- 200 trees for moderate ensemble size
- All 14 features (4 original + 10 engineered)
- Expected accuracy: 98-99%
"""

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

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
    """Run Random Forest split experiment with traditional train/test methodology."""
    logger.info("Starting Random Forest Split Experiment")

    # Load and prepare data
    X_engineered, y, iris_data, feature_names = load_and_prepare_data(ModelType.RANDOM_FOREST)

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.3, random_state=42, stratify=y)

    # Train Random Forest classifier
    n_estimators = 200  # Moderate number based on EDA recommendations
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,  # Use all cores for faster training
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    results = evaluate_model(y_test, y_pred, iris_data, X_test)

    # Calculate training accuracy to check for overfitting
    training_accuracy = model.score(X_train, y_train)

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=10)

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "random_forest",
        "experiment_type": "split",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
        },
        "feature_importance": sorted_importance,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "training_accuracy": training_accuracy,
        "overfitting_gap": abs(training_accuracy - results["overall_accuracy"]),
        "oob_score": getattr(model, "oob_score_", None),
    }

    # Cross-validation on training set for additional validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    results["cross_validation"] = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "random_forest_split",
        "timestamp": datetime.now().isoformat(),
        "model_type": "iris_classifier",
        "approach": "ensemble_learning",
    }

    results["performance_metrics"] = {
        "overall_accuracy": results["overall_accuracy"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "misclassification_count": len(results["misclassifications"]),
    }

    # Log performance and save results
    log_performance_summary(results, "random_forest_split")
    save_experiment_results(results, "random_forest", "split", model)

    logger.info(
        "Experiment completed",
        test_accuracy=results["performance_metrics"]["overall_accuracy"],
        training_accuracy=training_accuracy,
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
