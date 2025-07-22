"""
Data Loading Module for Rule-Based Heuristic Iris Classifier

This module provides data loading functionality for the Iris dataset.
"""

from typing import Any, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

from ai_flora_mind.logging import get_logger

logger = get_logger(__name__)


def load_iris_data() -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Bunch]:
    logger.info("Loading Iris dataset")

    iris = load_iris()
    X = iris.data
    y = iris.target  # Keep numeric targets (0, 1, 2)

    logger.info(
        "Dataset loaded successfully",
        total_samples=len(X),
        features=iris.feature_names,
        target_classes=iris.target_names.tolist(),
    )

    # Log feature statistics relevant to our model
    petal_lengths = X[:, 2]
    petal_widths = X[:, 3]

    logger.info(
        "Petal feature statistics",
        petal_length_range=f"{petal_lengths.min():.2f}-{petal_lengths.max():.2f}",
        petal_width_range=f"{petal_widths.min():.2f}-{petal_widths.max():.2f}",
    )

    return X, y, iris
