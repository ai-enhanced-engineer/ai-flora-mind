"""
Decision Tree Iris Classifier - Split Experiment

This module implements the split experiment for decision tree on the Iris dataset.
Uses traditional 70/30 train/test split methodology with a shallow tree for interpretability.

Configuration:
- Max depth: 3 (maintains human readability)
- Features: Original 4 + petal_area (engineered feature)
- Expected accuracy: 96-98%
"""

from datetime import datetime

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

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
    """Run decision tree split experiment with traditional train/test methodology."""
    logger.info("Starting Decision Tree Split Experiment")

    # Load and prepare data
    X_enhanced, y_true, iris_data, feature_names = load_and_prepare_data(ModelType.DECISION_TREE)

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y_true, test_size=0.3, random_state=42, stratify=y_true
    )

    # Configure and train decision tree
    model = DecisionTreeClassifier(
        max_depth=3,  # Shallow tree for interpretability
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance - pass X_test for correct misclassification analysis
    results = evaluate_model(y_test, y_pred, iris_data, X_test)

    # Calculate training accuracy to check for overfitting
    training_accuracy = model.score(X_train, y_train)

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=len(feature_names))

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "decision_tree",
        "experiment_type": "split",
        "hyperparameters": {
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "criterion": model.criterion,
        },
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
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
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

    # Log performance and save results
    log_performance_summary(results, "decision_tree_split")
    save_experiment_results(results, "decision_tree", "split", model)

    logger.info(
        "Experiment completed",
        test_accuracy=results["performance_metrics"]["overall_accuracy"],
        training_accuracy=training_accuracy,
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
