"""
Random Forest Iris Classifier - Regularized Experiment

This module implements the regularized experiment for Random Forest on the Iris dataset.
Uses regularization techniques to reduce overfitting and optimize performance.

Configuration:
- 100 trees (reduced for efficiency)
- Max depth: 5, min_samples_split: 5, min_samples_leaf: 2
- Max features: sqrt (feature subsampling)
"""

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

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
    """Run Random Forest regularized experiment with optimal hyperparameters."""
    logger.info("Starting Random Forest Regularized Experiment")

    # Load and prepare data
    X_engineered, y, iris_data, feature_names = load_and_prepare_data(ModelType.RANDOM_FOREST)

    # Train regularized model on full dataset
    model = RandomForestClassifier(
        n_estimators=100,  # Reduced from 300
        max_depth=5,  # Limited depth to prevent overfitting
        min_samples_split=5,  # Increased from default 2
        min_samples_leaf=2,  # Increased from default 1
        max_features="sqrt",  # Feature subsampling for diversity
        random_state=42,
        n_jobs=-1,
        oob_score=True,  # Enable OOB validation
    )
    model.fit(X_engineered, y)

    # Make predictions
    y_pred = model.predict(X_engineered)

    # Evaluate performance
    results = evaluate_model(y, y_pred, iris_data, X_engineered)

    # Calculate training accuracy and OOB score
    training_accuracy = model.score(X_engineered, y)
    oob_score = model.oob_score_

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=10)

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "random_forest",
        "experiment_type": "regularized",
        "hyperparameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features,
        },
        "feature_importance": sorted_importance,
        "training_accuracy": training_accuracy,
        "oob_score": oob_score,
        "overfitting_gap": abs(training_accuracy - oob_score),
        "regularization_impact": {
            "tree_reduction": "67% (300 → 100)",
            "depth_limit": "max_depth=5",
            "sample_constraints": "min_samples_split=5, min_samples_leaf=2",
            "feature_sampling": "sqrt",
        },
    }

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "random_forest_regularized",
        "timestamp": datetime.now().isoformat(),
        "model_type": "iris_classifier",
        "approach": "regularized_ensemble",
        "validation_method": "OOB",
    }

    results["performance_metrics"] = {
        "overall_accuracy": results["overall_accuracy"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "misclassification_count": len(results["misclassifications"]),
    }

    # Log performance and save results
    log_performance_summary(results, "random_forest_regularized")
    save_experiment_results(results, "random_forest", "regularized", model)

    logger.info(
        "Experiment completed",
        oob_score=oob_score,
        training_accuracy=training_accuracy,
        overfitting_gap=abs(training_accuracy - oob_score),
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
