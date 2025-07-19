"""
Feature engineering for Decision Tree Iris Classifier.

This module provides feature engineering functions specifically for the decision tree model,
including the creation of the petal_area feature which helps improve model performance.
"""

from typing import Any, List, Tuple

import numpy as np

from ai_flora_mind.logging import get_logger

logger = get_logger(__name__)


def create_petal_area_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Create petal area feature from petal length and width.

    This engineered feature helps the decision tree better separate species
    by combining the two most discriminative features identified in EDA.

    Args:
        X: Feature matrix with petal length in column 2 and petal width in column 3

    Returns:
        Array of petal area values
    """
    petal_length = X[:, 2]
    petal_width = X[:, 3]
    petal_area: np.ndarray[Any, Any] = petal_length * petal_width

    logger.debug(
        "Created petal area feature",
        min_area=float(np.min(petal_area)),
        max_area=float(np.max(petal_area)),
        mean_area=float(np.mean(petal_area)),
    )

    return petal_area


def engineer_features(X: np.ndarray[Any, Any], feature_names: List[str]) -> Tuple[np.ndarray[Any, Any], List[str]]:
    """
    Engineer features for the decision tree model.

    Currently adds petal_area as an additional feature to improve separability.

    Args:
        X: Original feature matrix (n_samples, 4)
        feature_names: Original feature names

    Returns:
        Tuple of (enhanced feature matrix, updated feature names)
    """
    logger.info("Engineering features for decision tree", original_features=len(feature_names), n_samples=len(X))

    # Create petal area feature
    petal_area = create_petal_area_feature(X).reshape(-1, 1)

    # Combine original features with engineered features
    X_enhanced = np.hstack([X, petal_area])
    enhanced_feature_names = feature_names + ["petal_area"]

    logger.info("Feature engineering completed", total_features=X_enhanced.shape[1], engineered_features=["petal_area"])

    return X_enhanced, enhanced_feature_names
