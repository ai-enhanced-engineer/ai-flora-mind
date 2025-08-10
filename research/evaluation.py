"""
Evaluation Module for Rule-Based Heuristic Iris Classifier

This module provides evaluation and performance summary functionality
for the rule-based heuristic classifier.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import Bunch

from ml_production_service.logging import get_logger
from research.experiments.constants import RESULTS_DIR

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    iris_data: Bunch,
    X_subset: np.ndarray[Any, Any] | None = None,
) -> Dict[str, Any]:
    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Generate detailed classification report
    class_report = classification_report(y_true, y_pred, target_names=iris_data.target_names, output_dict=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate per-class accuracy
    per_class_accuracy = {}

    # Check if labels are numeric or string
    is_numeric = isinstance(y_true[0], (int, np.integer))

    for i, species in enumerate(iris_data.target_names):
        if is_numeric:
            species_mask = y_true == i  # Use numeric index
            species_predictions = y_pred[species_mask]
            species_accuracy = accuracy_score([i] * len(species_predictions), species_predictions)
        else:
            # Handle string labels
            species_mask = y_true == species
            species_predictions = y_pred[species_mask]
            species_accuracy = accuracy_score([species] * len(species_predictions), species_predictions)
        per_class_accuracy[species] = species_accuracy

    # Analyze misclassifications
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]

    misclassifications = []
    # Use subset features if provided (for split experiments), otherwise use full dataset
    X = X_subset if X_subset is not None else iris_data.data
    for idx in misclassified_indices:
        # Handle both numeric and string labels
        if is_numeric:
            true_species = iris_data.target_names[y_true[idx]]
            pred_species = iris_data.target_names[y_pred[idx]]
        else:
            true_species = y_true[idx]
            pred_species = y_pred[idx]

        misc_data = {
            "index": idx,
            "true_species": true_species,
            "predicted_species": pred_species,
            "petal_length": X[idx, 2],
            "petal_width": X[idx, 3],
            "sepal_length": X[idx, 0],
            "sepal_width": X[idx, 1],
        }
        misclassifications.append(misc_data)

        logger.debug(
            "Misclassification detected",
            sample_index=idx,
            true_label=true_species if not is_numeric else y_true[idx],
            predicted_label=pred_species if not is_numeric else y_pred[idx],
            petal_length=X[idx, 2],
            petal_width=X[idx, 3],
        )

    results = {
        "overall_accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "misclassifications": misclassifications,
        "total_samples": len(y_true),
        "correct_predictions": len(y_true) - len(misclassified_indices),
    }

    # Log final evaluation summary
    logger.info(
        "Evaluation completed",
        accuracy=f"{accuracy:.3f}",
        misclassifications=len(misclassifications),
    )

    return results


def log_performance_summary(results: Dict[str, Any], experiment_name: str) -> None:
    """Log performance summary with EDA validation and target accuracy assessment."""

    # Extract overall accuracy from either location
    if "performance_metrics" in results:
        overall_accuracy = results["performance_metrics"]["overall_accuracy"]
    else:
        overall_accuracy = results["overall_accuracy"]

    # Log essential performance metrics in single statement
    per_class = results["per_class_accuracy"]
    logger.info(
        "Performance",
        overall=f"{overall_accuracy:.1%}",
        setosa=f"{per_class['setosa']:.1%}",
        versicolor=f"{per_class['versicolor']:.1%}",
        virginica=f"{per_class['virginica']:.1%}",
    )

    # Skip verbose confusion matrix and algorithm details logging

    # Log misclassification summary only
    misc_count = len(results.get("misclassifications", []))
    if misc_count > 0:
        logger.info(f"Misclassifications: {misc_count} samples")

    # Log key insights and validation
    setosa_acc = results["per_class_accuracy"]["setosa"]
    overall_acc = overall_accuracy  # Use the variable we already resolved above

    if setosa_acc == 1.0:
        logger.info(
            "EDA validation successful",
            insight="Perfect Setosa classification validates EDA finding",
            setosa_accuracy=setosa_acc,
        )
    else:
        logger.warning(
            "EDA validation issue",
            warning=f"Setosa accuracy: {setosa_acc:.3f} (expected: 1.0)",
            setosa_accuracy=setosa_acc,
        )

    if overall_acc >= 0.95:
        logger.info(
            "Target accuracy achieved",
            insight=f"Excellent overall accuracy: {overall_acc:.3f}",
            target_range="0.95-0.97",
            achieved_accuracy=overall_acc,
        )
    else:
        logger.warning(
            "Target accuracy not met",
            warning=f"Below target accuracy: {overall_acc:.3f}",
            target_range="0.95-0.97",
            achieved_accuracy=overall_acc,
        )

    # Save results to JSON file
    _save_results_to_json(results, experiment_name)


def _save_results_to_json(results: Dict[str, Any], experiment_name: str) -> None:
    # Use centralized results directory
    results_dir = Path(RESULTS_DIR)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = results_dir / filename

    # Prepare JSON-serializable results
    json_results = {
        "experiment_metadata": {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": "iris_classifier",
            "approach": "baseline",
        },
        "performance_metrics": {
            "overall_accuracy": float(results["performance_metrics"]["overall_accuracy"])
            if "performance_metrics" in results
            else float(results["overall_accuracy"]),
            "total_samples": int(results["performance_metrics"]["total_samples"])
            if "performance_metrics" in results
            else int(results["total_samples"]),
            "correct_predictions": int(results["performance_metrics"]["correct_predictions"])
            if "performance_metrics" in results
            else int(results["correct_predictions"]),
            "misclassification_count": int(results["performance_metrics"]["misclassification_count"])
            if "performance_metrics" in results and "misclassification_count" in results["performance_metrics"]
            else len(results["misclassifications"]),
        },
        "per_class_accuracy": {species: float(accuracy) for species, accuracy in results["per_class_accuracy"].items()},
        "confusion_matrix": results["confusion_matrix"].tolist()
        if hasattr(results["confusion_matrix"], "tolist")
        else results["confusion_matrix"],
        "classification_report": results["classification_report"],
        "algorithm_details": results.get("algorithm_details", {}),
        "misclassifications": results["misclassifications"],
    }

    # Save to JSON file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        logger.info(
            "Results saved to JSON file", filepath=str(filepath), filename=filename, results_directory=str(results_dir)
        )

    except Exception as e:
        logger.error("Failed to save results to JSON", error=str(e), filepath=str(filepath))
        raise
