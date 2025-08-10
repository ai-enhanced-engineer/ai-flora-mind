"""
XGBoost Iris Classifier - Split Experiment

This module implements the split experiment for XGBoost on the Iris dataset.
Uses traditional 70/30 train/test split methodology to establish baseline performance.

Configuration:
- Conservative hyperparameters (max_depth=3, learning_rate=0.1)
- 100 trees with subsampling to prevent overfitting
- Targeted feature engineering for gradient boosting
"""

from datetime import datetime

import numpy as np
import xgboost as xgb
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
    """Run XGBoost split experiment with traditional train/test methodology."""
    logger.info("Starting XGBoost Split Experiment")

    # Load and prepare data (XGBoost needs numeric labels)
    X_engineered, y, iris_data, feature_names = load_and_prepare_data(ModelType.XGBOOST, numeric_labels=True)

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.3, random_state=42, stratify=y)

    # Configure XGBoost with conservative parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Convert predictions back to string labels for evaluation
    y_test_names = np.array([iris_data.target_names[i] for i in y_test])
    y_pred_names = np.array([iris_data.target_names[i] for i in y_pred])

    # Evaluate performance
    results = evaluate_model(y_test_names, y_pred_names, iris_data, X_test)

    # Calculate training accuracy
    training_accuracy = model.score(X_train, y_train)

    # Extract feature importance
    sorted_importance = extract_feature_importance(model, feature_names, top_n=10)

    # Add algorithm-specific details
    results["algorithm_details"] = {
        "algorithm_type": "xgboost",
        "experiment_type": "split",
        "hyperparameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
        },
        "feature_importance": sorted_importance,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "training_accuracy": training_accuracy,
        "overfitting_gap": abs(training_accuracy - results["overall_accuracy"]),
    }

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    results["cross_validation"] = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }

    # Add metadata
    results["experiment_metadata"] = {
        "experiment_name": "xgboost_split",
        "timestamp": datetime.now().isoformat(),
        "model_type": "iris_classifier",
        "approach": "gradient_boosting",
    }

    results["performance_metrics"] = {
        "overall_accuracy": results["overall_accuracy"],
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "misclassification_count": len(results["misclassifications"]),
    }

    # Log performance and save results
    log_performance_summary(results, "xgboost_split")
    save_experiment_results(results, "xgboost", "split", model)

    logger.info(
        "Experiment completed",
        test_accuracy=results["performance_metrics"]["overall_accuracy"],
        training_accuracy=training_accuracy,
        top_features=list(sorted_importance.keys())[:3],
    )


if __name__ == "__main__":
    main()
