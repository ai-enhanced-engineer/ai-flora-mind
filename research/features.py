"""
Feature engineering for Iris Classifiers.

This module provides comprehensive feature engineering functions for different model types,
including all 10 engineered features identified in EDA analysis for maximum accuracy.
"""

from enum import Enum
from typing import Any, List, Tuple

import numpy as np

from ml_production_service.logging import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    DECISION_TREE = "decision_tree"  # Uses basic features for interpretability
    RANDOM_FOREST = "random_forest"  # Uses all engineered features for maximum accuracy
    HEURISTIC = "heuristic"  # Uses only original features
    XGBOOST = "xgboost"  # Uses targeted high-discriminative features for gradient boosting


def create_petal_area_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Combines the two most discriminative features identified in EDA analysis.

    Critical for decision tree species separation based on research findings.
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


def create_sepal_area_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    sepal_length = X[:, 0]
    sepal_width = X[:, 1]
    sepal_area: np.ndarray[Any, Any] = sepal_length * sepal_width
    return sepal_area


def create_petal_aspect_ratio_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    petal_length = X[:, 2]
    petal_width = X[:, 3]
    # Avoid division by zero
    return np.where(petal_width > 0, petal_length / petal_width, 0)


def create_sepal_aspect_ratio_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    sepal_length = X[:, 0]
    sepal_width = X[:, 1]
    # Avoid division by zero
    return np.where(sepal_width > 0, sepal_length / sepal_width, 0)


def create_total_area_feature(
    petal_area: np.ndarray[Any, Any], sepal_area: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    total_area: np.ndarray[Any, Any] = petal_area + sepal_area
    return total_area


def create_area_ratio_feature(
    petal_area: np.ndarray[Any, Any], sepal_area: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    # Avoid division by zero
    return np.where(sepal_area > 0, petal_area / sepal_area, 0)


def create_is_likely_setosa_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """EDA heuristic rule: petal_length < 2.0 achieves perfect Setosa separation."""
    petal_length = X[:, 2]
    # Rule from heuristic classifier: petal_length < 2.0 indicates Setosa
    return (petal_length < 2.0).astype(float)


def create_petal_to_sepal_length_ratio_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    petal_length = X[:, 2]
    sepal_length = X[:, 0]
    return np.where(sepal_length > 0, petal_length / sepal_length, 0)


def create_petal_to_sepal_width_ratio_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    petal_width = X[:, 3]
    sepal_width = X[:, 1]
    return np.where(sepal_width > 0, petal_width / sepal_width, 0)


def create_size_index_feature(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    size_index: np.ndarray[Any, Any] = np.mean(X, axis=1)
    return size_index


def create_versicolor_vs_virginica_interaction(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Interaction term targeting Versicolor/Virginica boundary - the main classification challenge."""
    petal_length = X[:, 2]
    petal_width = X[:, 3]
    # Interaction combining petal features for non-Setosa discrimination
    # Based on EDA insight that Versicolor/Virginica separation is the main challenge
    interaction: np.ndarray[Any, Any] = petal_length * petal_width * (petal_length / petal_width)
    return interaction


def engineer_features(
    X: np.ndarray[Any, Any], feature_names: List[str], model_type: ModelType = ModelType.DECISION_TREE
) -> Tuple[np.ndarray[Any, Any], List[str]]:
    """
    Model-specific feature engineering based on EDA analysis and performance requirements.

    DECISION_TREE: 5 features (interpretability focus)
    RANDOM_FOREST: 14 features (maximum accuracy)
    XGBOOST: 9 features (high-discriminative + interaction terms)
    HEURISTIC: 4 features (original only)
    """
    logger.info(f"Engineering features for {model_type.value}", original_features=len(feature_names), n_samples=len(X))

    if model_type == ModelType.HEURISTIC:
        # Heuristic model uses only original features
        logger.info("Feature engineering completed", total_features=X.shape[1], engineered_features=[])
        return X, feature_names

    elif model_type == ModelType.DECISION_TREE:
        # Decision tree uses basic engineered features for interpretability
        petal_area = create_petal_area_feature(X).reshape(-1, 1)

        X_enhanced = np.hstack([X, petal_area])
        enhanced_feature_names = feature_names + ["petal_area"]

        logger.info(
            "Feature engineering completed", total_features=X_enhanced.shape[1], engineered_features=["petal_area"]
        )
        return X_enhanced, enhanced_feature_names

    elif model_type == ModelType.RANDOM_FOREST:
        # Random Forest uses all 10 engineered features for maximum accuracy

        # Create all engineered features
        petal_area = create_petal_area_feature(X)
        sepal_area = create_sepal_area_feature(X)
        petal_aspect_ratio = create_petal_aspect_ratio_feature(X)
        sepal_aspect_ratio = create_sepal_aspect_ratio_feature(X)
        total_area = create_total_area_feature(petal_area, sepal_area)
        area_ratio = create_area_ratio_feature(petal_area, sepal_area)
        is_likely_setosa = create_is_likely_setosa_feature(X)
        petal_to_sepal_length_ratio = create_petal_to_sepal_length_ratio_feature(X)
        petal_to_sepal_width_ratio = create_petal_to_sepal_width_ratio_feature(X)
        size_index = create_size_index_feature(X)

        # Stack all engineered features
        engineered_features = np.column_stack(
            [
                petal_area,
                sepal_area,
                petal_aspect_ratio,
                sepal_aspect_ratio,
                total_area,
                area_ratio,
                is_likely_setosa,
                petal_to_sepal_length_ratio,
                petal_to_sepal_width_ratio,
                size_index,
            ]
        )

        # Combine original features with all engineered features
        X_enhanced = np.hstack([X, engineered_features])

        # Create feature names for all engineered features
        engineered_feature_names = [
            "petal_area",
            "sepal_area",
            "petal_aspect_ratio",
            "sepal_aspect_ratio",
            "total_area",
            "area_ratio",
            "is_likely_setosa",
            "petal_to_sepal_length_ratio",
            "petal_to_sepal_width_ratio",
            "size_index",
        ]

        enhanced_feature_names = feature_names + engineered_feature_names

        logger.info(
            "Feature engineering completed",
            total_features=X_enhanced.shape[1],
            engineered_features=engineered_feature_names,
        )

        # Debug log for Random Forest feature validation
        logger.debug(
            "Random Forest feature stats",
            original_features=len(feature_names),
            engineered_features=len(engineered_feature_names),
            total_features=len(enhanced_feature_names),
            petal_area_range=f"{petal_area.min():.2f}-{petal_area.max():.2f}",
            is_likely_setosa_sum=int(is_likely_setosa.sum()),
        )

        return X_enhanced, enhanced_feature_names

    elif model_type == ModelType.XGBOOST:
        # XGBoost uses targeted high-discriminative features based on EDA analysis
        # Strategy: High CV features + perfect separability indicator + interaction terms

        # High CV (coefficient of variation) features from EDA
        petal_area = create_petal_area_feature(X)  # CV: 0.813 (highest discriminative power)
        area_ratio = create_area_ratio_feature(petal_area, create_sepal_area_feature(X))  # Key ratio feature

        # Perfect separability indicator from EDA heuristic analysis
        is_likely_setosa = create_is_likely_setosa_feature(X)  # Binary flag for perfect Setosa separation

        # Interaction terms specifically for Versicolor/Virginica boundary challenge
        versicolor_virginica_interaction = create_versicolor_vs_virginica_interaction(X)
        petal_to_sepal_width_ratio = create_petal_to_sepal_width_ratio_feature(X)  # Key discriminative ratio

        # Stack targeted engineered features (8 total: 4 original + 4 engineered)
        engineered_features = np.column_stack(
            [
                petal_area,
                area_ratio,
                is_likely_setosa,
                versicolor_virginica_interaction,
                petal_to_sepal_width_ratio,
            ]
        )

        # Combine original features with targeted engineered features
        X_enhanced = np.hstack([X, engineered_features])

        # Create feature names for targeted engineered features
        engineered_feature_names = [
            "petal_area",
            "area_ratio",
            "is_likely_setosa",
            "versicolor_virginica_interaction",
            "petal_to_sepal_width_ratio",
        ]

        enhanced_feature_names = feature_names + engineered_feature_names

        logger.info(
            "Feature engineering completed",
            total_features=X_enhanced.shape[1],
            engineered_features=engineered_feature_names,
        )

        # Debug log for XGBoost targeted feature validation
        logger.debug(
            "XGBoost targeted feature stats",
            original_features=len(feature_names),
            engineered_features=len(engineered_feature_names),
            total_features=len(enhanced_feature_names),
            petal_area_range=f"{petal_area.min():.2f}-{petal_area.max():.2f}",
            setosa_samples=int(is_likely_setosa.sum()),
            interaction_range=f"{versicolor_virginica_interaction.min():.2f}-{versicolor_virginica_interaction.max():.2f}",
        )

        return X_enhanced, enhanced_feature_names

    else:
        raise ValueError(f"Unknown model type: {model_type}")
