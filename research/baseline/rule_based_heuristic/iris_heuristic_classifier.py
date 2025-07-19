"""
Rule-Based Heuristic Iris Classifier

This module implements a simple rule-based classifier for the Iris dataset based on
comprehensive EDA findings. The classifier leverages perfect Setosa separation via
petal features and achieves ~97% accuracy with zero training time.

Algorithm:
- if petal_length < 2.0: setosa
- elif petal_width < 1.7: versicolor
- else: virginica

Key Benefits:
- Perfect interpretability (transparent decision rules)
- Zero training time (instant deployment)
- High accuracy baseline (95-97%)
- Educational value for understanding data patterns
"""

from typing import Any, List, Union

import numpy as np

from ai_flora_mind.logging import get_logger

from ..data import load_iris_data
from ..evaluation import evaluate_model, log_performance_summary

logger = get_logger(__name__)


def classify_iris_heuristic(petal_length: float, petal_width: float) -> str:
    """
    Classify iris species using rule-based heuristic.

    Based on EDA findings that show perfect Setosa separation via petal features
    and clear thresholds for distinguishing between Versicolor and Virginica.

    Args:
        petal_length: Petal length measurement in cm
        petal_width: Petal width measurement in cm

    Returns:
        Species prediction: 'setosa', 'versicolor', or 'virginica'

    Raises:
        ValueError: If measurements are negative or unrealistic

    Examples:
        >>> classify_iris_heuristic(1.4, 0.2)
        'setosa'
        >>> classify_iris_heuristic(4.5, 1.5)
        'versicolor'
        >>> classify_iris_heuristic(6.0, 2.0)
        'virginica'
    """
    logger.debug("Classifying iris sample", petal_length=petal_length, petal_width=petal_width)

    # Input validation
    if petal_length < 0 or petal_width < 0:
        error_msg = "Petal measurements must be non-negative"
        logger.error(
            "Invalid input validation failed", error=error_msg, petal_length=petal_length, petal_width=petal_width
        )
        raise ValueError(error_msg)

    if petal_length > 10 or petal_width > 5:
        error_msg = "Unrealistic petal measurements provided"
        logger.warning(
            "Unrealistic measurements detected", warning=error_msg, petal_length=petal_length, petal_width=petal_width
        )
        raise ValueError(error_msg)

    # Rule-based classification logic
    # Rule 1: Perfect Setosa separation (EDA finding: petal_length < 2.0)
    if petal_length < 2.0:
        prediction = "setosa"
        logger.debug("Applied rule 1 - Setosa separation", rule="petal_length < 2.0", prediction=prediction)
        return prediction

    # Rule 2: Versicolor vs Virginica separation (EDA finding: petal_width threshold)
    elif petal_width < 1.7:
        prediction = "versicolor"
        logger.debug("Applied rule 2 - Versicolor threshold", rule="petal_width < 1.7", prediction=prediction)
        return prediction

    # Rule 3: Default to Virginica for large petal measurements
    else:
        prediction = "virginica"
        logger.debug("Applied rule 3 - Virginica default", rule="else (large petals)", prediction=prediction)
        return prediction


def classify_batch(
    petal_lengths: Union[List[float], np.ndarray[Any, Any]], petal_widths: Union[List[float], np.ndarray[Any, Any]]
) -> List[str]:
    """
    Classify multiple iris samples using heuristic rules.

    Args:
        petal_lengths: Array of petal length measurements
        petal_widths: Array of petal width measurements

    Returns:
        List of species predictions

    Raises:
        ValueError: If input arrays have different lengths
    """
    petal_lengths = np.array(petal_lengths)
    petal_widths = np.array(petal_widths)

    logger.info("Starting batch classification", sample_count=len(petal_lengths))

    if len(petal_lengths) != len(petal_widths):
        error_msg = "Petal length and width arrays must have same length"
        logger.error(
            "Batch validation failed",
            error=error_msg,
            petal_lengths_count=len(petal_lengths),
            petal_widths_count=len(petal_widths),
        )
        raise ValueError(error_msg)

    predictions = []
    for i, (pl, pw) in enumerate(zip(petal_lengths, petal_widths)):
        predictions.append(classify_iris_heuristic(pl, pw))
        if (i + 1) % 50 == 0:  # Log progress every 50 samples
            logger.debug("Batch classification progress", processed=i + 1, total=len(petal_lengths))

    logger.info(
        "Batch classification completed", total_samples=len(predictions), predictions_generated=len(predictions)
    )

    return predictions


if __name__ == "__main__":
    logger.info("Rule-based heuristic classifier evaluation started")

    try:
        # Step 1: Load data
        X, y_true, iris_data = load_iris_data()

        # Step 2: Train (no training needed for rule-based approach)
        logger.info("Training phase - no training required for rule-based heuristic")

        # Step 3: Predict using heuristic rules
        logger.info("Making predictions using rule-based heuristic")
        petal_lengths = X[:, 2]  # petal length (column 2)
        petal_widths = X[:, 3]  # petal width (column 3)
        y_pred = classify_batch(petal_lengths, petal_widths)

        # Step 4: Evaluate performance
        results = evaluate_model(y_true, np.array(y_pred), iris_data)

        # Add algorithm-specific details to results
        results["algorithm_details"] = {
            "algorithm_type": "rule_based_heuristic",
            "rules": {
                "rule_1": "if petal_length < 2.0: setosa",
                "rule_2": "elif petal_width < 1.7: versicolor",
                "rule_3": "else: virginica",
            },
            "features_used": ["petal_length", "petal_width"],
            "training_required": False,
        }

        # Step 5: Generate comprehensive performance summary
        experiment_name = "rule_based_heuristic"
        log_performance_summary(results, experiment_name)

    except Exception as e:
        logger.error("Script execution failed", error=str(e), error_type=type(e).__name__)
        print(f"Error: {e}")
        exit(1)

    logger.info("Rule-based heuristic classifier evaluation completed successfully")
