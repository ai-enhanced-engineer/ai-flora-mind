"""
XGBoost Iris Classifier - Split Experiment

This module implements the split experiment for XGBoost on the Iris dataset.
Uses traditional 70/30 train/test split methodology to establish baseline performance.

Configuration:
- Conservative hyperparameters (max_depth=3, learning_rate=0.1)
- 100 trees with subsampling to prevent overfitting
- Targeted feature engineering for gradient boosting
"""

import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from ai_flora_mind.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run XGBoost split experiment with traditional train/test methodology."""
    logger.info("Starting XGBoost Split Experiment")

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

    # Split data (70/30 split following other experiments)
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.3, random_state=42, stratify=y)

    logger.info(
        "Data split completed",
        training_samples=len(X_train),
        test_samples=len(X_test),
        samples_per_class_train=[len(y_train[y_train == i]) for i in range(3)],
        samples_per_class_test=[len(y_test[y_test == i]) for i in range(3)],
    )

    # Create XGBoost classifier with conservative hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=100,  # Conservative tree count
        max_depth=3,  # Shallow trees to prevent overfitting
        learning_rate=0.1,  # Conservative learning rate
        subsample=0.8,  # Feature subsampling
        colsample_bytree=0.8,  # Column subsampling
        random_state=42,
        eval_metric="mlogloss",  # Multi-class log loss
    )

    # Train model
    logger.info("Training XGBoost model with conservative hyperparameters")
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Evaluate on test set
    results = evaluate_model(y_test, y_pred, iris_data, X_test)

    # Calculate training accuracy to check for overfitting
    training_accuracy = model.score(X_train, y_train)

    # Add required metadata structure
    results["experiment_metadata"] = {
        "experiment_name": "xgboost_split",
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

    # Feature importance analysis - convert to native float for JSON serialization
    feature_importance = dict(zip(engineered_feature_names, [float(imp) for imp in model.feature_importances_]))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Add XGBoost-specific metadata
    results["algorithm_details"] = {
        "experiment_type": "split",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "learning_rate": model.learning_rate,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
        "features_used": engineered_feature_names,
        "feature_importance": sorted_importance,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "training_accuracy": training_accuracy,
        "overfitting_gap": abs(training_accuracy - results["overall_accuracy"]),
    }

    # Log performance summary
    experiment_name = "xgboost_split"
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
        f"XGBoost Split Experiment: {results['performance_metrics']['overall_accuracy']:.1%} test accuracy "
        f"({results['performance_metrics']['correct_predictions']}/{results['performance_metrics']['total_samples']} correct). "
        f"Training accuracy: {results['algorithm_details']['training_accuracy']:.1%}. "
        f"Overfitting gap: {results['algorithm_details']['overfitting_gap']:.1%}. "
        f"Top features: {', '.join(list(results['algorithm_details']['feature_importance'].keys())[:3])}"
    )

    logger.info(
        "XGBoost split experiment completed successfully",
        accuracy=f"{results['performance_metrics']['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
