"""
Decision Tree Iris Classifier - Split Experiment

This module implements the split experiment for decision tree on the Iris dataset.
Uses traditional 70/30 train/test split methodology with a shallow tree for interpretability.

Configuration:
- Max depth: 3 (maintains human readability)
- Features: Original 4 + petal_area (engineered feature)
- Expected accuracy: 96-98%
"""

import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from ml_production_service.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run decision tree split experiment with traditional train/test methodology."""
    logger.info("Starting Decision Tree Split Experiment")

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

    # Split data for training
    logger.info("Splitting data for training", test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_true, test_size=0.3, random_state=42, stratify=y_true
    )

    logger.info(
        "Data split completed",
        training_samples=len(X_train),
        test_samples=len(X_test),
        samples_per_class_train=[len(y_train[y_train == species]) for species in iris_data.target_names],
        samples_per_class_test=[len(y_test[y_test == species]) for species in iris_data.target_names],
    )

    # Configure decision tree for interpretability and EDA insights
    logger.info("Training decision tree classifier", max_depth=3)
    model = DecisionTreeClassifier(
        max_depth=3,  # Shallow tree for interpretability
        random_state=42,  # Reproducible results
        min_samples_split=2,  # Allow smaller splits for better patterns
        min_samples_leaf=1,  # Allow single-sample leaves for precision
        criterion="gini",  # Standard impurity measure
    )

    # Train the model
    model.fit(X_train, y_train)

    logger.info(
        "Decision tree training completed",
        tree_depth=model.get_depth(),
        n_leaves=model.get_n_leaves(),
        training_accuracy=model.score(X_train, y_train),
    )

    # Make predictions
    logger.info("Making predictions on test set")
    y_pred = model.predict(X_test)

    # Evaluate performance - pass X_test for correct misclassification analysis
    results = evaluate_model(y_test, y_pred, iris_data, X_test)

    # Calculate training accuracy to check for overfitting
    training_accuracy = model.score(X_train, y_train)

    # Extract feature importance
    feature_importance = {
        name: float(importance) for name, importance in zip(feature_names, model.feature_importances_)
    }
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "decision_tree",
        "experiment_type": "split",
        "max_depth": 3,
        "features_used": feature_names,
        "feature_importance": sorted_importance,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "training_accuracy": training_accuracy,
        "overfitting_gap": abs(training_accuracy - results["overall_accuracy"]),
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "decision_rules": export_text(model, feature_names=feature_names),
    }

    # Cross-validation on training set for additional validation
    logger.info("Performing 5-fold cross-validation on training set")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores = np.array(cv_scores)  # Ensure it's a numpy array
    results["cross_validation"] = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "decision_tree_split",
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

    # Log performance summary
    experiment_name = "decision_tree_split"
    log_performance_summary(results, experiment_name)

    # Validate performance against targets
    target_min, target_max = 0.96, 0.98
    test_acc = results["overall_accuracy"]

    if target_min <= test_acc <= target_max:
        logger.info(
            "Target accuracy achieved",
            insight=f"Test accuracy {test_acc:.3f} within target range",
            target_range=f"{target_min}-{target_max}",
            achieved_accuracy=test_acc,
        )
    else:
        logger.warning(
            "Target accuracy not met",
            warning=f"Test accuracy {test_acc:.3f} outside target range",
            target_range=f"{target_min}-{target_max}",
            achieved_accuracy=test_acc,
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
        f"Decision Tree Split Experiment: {results['performance_metrics']['overall_accuracy']:.1%} test accuracy "
        f"({results['performance_metrics']['correct_predictions']}/{results['performance_metrics']['total_samples']} correct). "
        f"Training accuracy: {results['algorithm_details']['training_accuracy']:.1%}. "
        f"CV mean: {results['cross_validation']['cv_mean']:.1%}. "
        f"Top features: {', '.join(list(sorted_importance.keys())[:3])}"
    )

    logger.info(
        "Decision tree split experiment completed successfully",
        accuracy=f"{results['performance_metrics']['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
