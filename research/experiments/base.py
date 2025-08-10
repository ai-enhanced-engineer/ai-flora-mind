"""
Base utilities for research experiments.

Provides common functionality used across all experiments to reduce duplication.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, cross_val_score

from ml_production_service.logging import get_logger
from research.data import load_iris_data
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def load_and_prepare_data(
    model_type: ModelType, numeric_labels: bool = False
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Any, List[str]]:
    """Load iris data and engineer features for the specified model type.

    Args:
        model_type: Type of model to prepare data for
        numeric_labels: If True, return numeric labels (for XGBoost)
    """
    X, y_names, iris_data = load_iris_data()

    # XGBoost requires numeric targets
    if numeric_labels:
        y = iris_data.target
    else:
        y = y_names

    X_enhanced, feature_names = engineer_features(X, list(iris_data.feature_names), model_type)

    logger.info("Data prepared", samples=len(X), features=len(feature_names), model_type=model_type.value)

    return X_enhanced, y, iris_data, feature_names


def save_experiment_results(
    results: Dict[str, Any], algorithm_type: str, experiment_type: str, model: Optional[Any] = None
) -> str:
    """Save experiment results and optionally the trained model."""

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        return obj

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # Save results JSON
    results_filename = f"{algorithm_type}_{experiment_type}_{timestamp}.json"
    results_path = os.path.join(RESULTS_DIR, results_filename)

    json_safe_results = convert_numpy_to_list(results)

    with open(results_path, "w") as f:
        json.dump(json_safe_results, f, indent=2)

    logger.info("Results saved", path=results_path)

    # Save model if provided
    if model is not None:
        model_filename = f"{algorithm_type}_{experiment_type}_{timestamp}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        logger.info("Model saved", path=model_path)

    return results_path


def perform_comprehensive_validation(
    model_class: Any,
    model_params: Dict[str, Any],
    X: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    cv_folds: int = 10,
    cv_repeats: int = 10,
) -> Tuple[Any, Dict[str, Any]]:
    """Perform comprehensive validation including LOOCV and repeated k-fold."""
    validation_results: Dict[str, Any] = {}

    # Train on full dataset
    model = model_class(**model_params)
    model.fit(X, y)
    training_accuracy = model.score(X, y)

    # Get OOB score if available (RandomForest)
    oob_score = getattr(model, "oob_score_", None)

    validation_results["training_accuracy"] = float(training_accuracy)
    if oob_score is not None:
        validation_results["oob_score"] = float(oob_score)

    # Leave-One-Out Cross-Validation
    loocv = LeaveOneOut()
    loocv_predictions: List[Any] = []
    loocv_true_labels: List[Any] = []

    for train_idx, test_idx in loocv.split(X):
        fold_model = model_class(**model_params)
        fold_model.fit(X[train_idx], y[train_idx])
        pred = fold_model.predict(X[test_idx])[0]
        loocv_predictions.append(pred)
        loocv_true_labels.append(y[test_idx][0])

    loocv_predictions_array = np.array(loocv_predictions)
    loocv_true_labels_array = np.array(loocv_true_labels)
    loocv_scores = (loocv_predictions_array == loocv_true_labels_array).astype(float)

    validation_results["loocv"] = {
        "accuracy": float(loocv_scores.mean()),
        "std": float(loocv_scores.std()),
        "predictions": loocv_predictions_array.tolist(),
        "true_labels": loocv_true_labels_array.tolist(),
        "scores": loocv_scores.tolist(),
    }

    # Repeated Stratified K-Fold
    rskf = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)
    repeated_scores = cross_val_score(model_class(**model_params), X, y, cv=rskf, scoring="accuracy")

    validation_results["repeated_kfold"] = {
        "accuracy": float(repeated_scores.mean()),
        "std": float(repeated_scores.std()),
        "scores": repeated_scores.tolist(),
    }

    logger.info(
        "Comprehensive validation complete",
        loocv_acc=validation_results["loocv"]["accuracy"],
        repeated_cv_acc=validation_results["repeated_kfold"]["accuracy"],
    )

    return model, validation_results


def extract_feature_importance(model: Any, feature_names: List[str], top_n: int = 5) -> Dict[str, float]:
    """Extract and return top feature importances from a trained model."""
    if hasattr(model, "feature_importances_"):
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, model.feature_importances_)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
        return sorted_importance
    return {}
