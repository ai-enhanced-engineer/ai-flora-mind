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
    logger.info("Step 4: Evaluating model performance")

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)

    logger.info(
        "Performance metrics calculated",
        overall_accuracy=accuracy,
        correct_predictions=int(accuracy * len(y_true)),
        total_samples=len(y_true),
    )

    # Generate detailed classification report
    class_report = classification_report(y_true, y_pred, target_names=iris_data.target_names, output_dict=True)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, species in enumerate(iris_data.target_names):
        species_mask = y_true == i  # Use numeric index
        species_predictions = y_pred[species_mask]
        species_accuracy = accuracy_score([i] * len(species_predictions), species_predictions)
        per_class_accuracy[species] = species_accuracy

    # Analyze misclassifications
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]

    logger.info(
        "Analyzing misclassifications",
        misclassified_count=len(misclassified_indices),
        misclassification_rate=len(misclassified_indices) / len(y_true),
    )

    misclassifications = []
    # Use subset features if provided (for split experiments), otherwise use full dataset
    X = X_subset if X_subset is not None else iris_data.data
    for idx in misclassified_indices:
        misc_data = {
            "index": idx,
            "true_species": iris_data.target_names[y_true[idx]],
            "predicted_species": iris_data.target_names[y_pred[idx]],
            "petal_length": X[idx, 2],
            "petal_width": X[idx, 3],
            "sepal_length": X[idx, 0],
            "sepal_width": X[idx, 1],
        }
        misclassifications.append(misc_data)

        logger.debug(
            "Misclassification detected",
            sample_index=idx,
            true_label=y_true[idx],
            predicted_label=y_pred[idx],
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

    logger.info(
        "Evaluation completed successfully",
        overall_accuracy=accuracy,
        setosa_accuracy=per_class_accuracy.get("setosa", 0),
        versicolor_accuracy=per_class_accuracy.get("versicolor", 0),
        virginica_accuracy=per_class_accuracy.get("virginica", 0),
        total_misclassifications=len(misclassifications),
    )

    return results


def log_performance_summary(results: Dict[str, Any], experiment_name: str) -> None:
    """Comprehensive performance logging with EDA validation and target accuracy assessment."""
    logger.info("Generating performance summary")

    # Log overall performance metrics - check both new and legacy locations
    if "performance_metrics" in results:
        total_samples = results["performance_metrics"]["total_samples"]
        correct_predictions = results["performance_metrics"]["correct_predictions"]
        overall_accuracy = results["performance_metrics"]["overall_accuracy"]
    else:
        total_samples = results["total_samples"]
        correct_predictions = results["correct_predictions"]
        overall_accuracy = results["overall_accuracy"]

    logger.info(
        "Overall performance summary",
        total_samples=total_samples,
        correct_predictions=correct_predictions,
        overall_accuracy=f"{overall_accuracy:.3f}",
        accuracy_percentage=f"{overall_accuracy * 100:.1f}%",
    )

    # Log per-class accuracy
    for species, accuracy in results["per_class_accuracy"].items():
        logger.info(
            "Per-class accuracy",
            species=species.capitalize(),
            accuracy=f"{accuracy:.3f}",
            accuracy_percentage=f"{accuracy * 100:.1f}%",
        )

    # Log confusion matrix data
    confusion_data = {}
    species_names = ["setosa", "versicolor", "virginica"]
    for i, true_species in enumerate(species_names):
        row_data = {}
        for j, pred_species in enumerate(species_names):
            row_data[f"predicted_{pred_species}"] = int(results["confusion_matrix"][i][j])
        confusion_data[f"true_{true_species}"] = row_data

    logger.info("Confusion matrix", confusion_matrix=confusion_data)

    # Log algorithm details
    algorithm_details = results.get("algorithm_details", {})
    if algorithm_details:
        logger.info(
            "Algorithm details",
            algorithm_type=algorithm_details.get("algorithm_type", "unknown"),
            features_used=algorithm_details.get("features_used", []),
            training_required=algorithm_details.get("training_required", True),
        )

        if "rules" in algorithm_details:
            logger.info("Algorithm rules applied", rules=algorithm_details["rules"])

    # Log misclassifications
    if results["misclassifications"]:
        logger.info("Misclassifications detected", total_misclassifications=len(results["misclassifications"]))

        for misc in results["misclassifications"]:
            logger.warning(
                "Misclassification detail",
                sample_index=misc["index"],
                true_species=misc["true_species"],
                predicted_species=misc["predicted_species"],
                petal_length=f"{misc['petal_length']:.1f}",
                petal_width=f"{misc['petal_width']:.1f}",
                sepal_length=f"{misc['sepal_length']:.1f}",
                sepal_width=f"{misc['sepal_width']:.1f}",
            )
    else:
        logger.info("Perfect classification - no misclassifications detected")

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

    logger.info("Performance summary completed")


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
