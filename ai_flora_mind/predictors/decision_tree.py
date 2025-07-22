"""
Decision Tree predictor for iris species classification.

Implements a machine learning predictor using pre-trained Decision Tree models
with minimal feature engineering for maximum interpretability.
"""

from typing import Any

import numpy as np
from pydantic import validate_call

from ai_flora_mind.configs import IrisMeasurements, ModelType
from ai_flora_mind.features import engineer_features, get_feature_names
from ai_flora_mind.logging import get_logger
from ai_flora_mind.predictors.base import BasePredictor

logger = get_logger(__name__)


class DecisionTreePredictor(BasePredictor):
    """
    Decision Tree predictor for iris species classification.

    Loads a pre-trained Decision Tree model and applies minimal feature engineering
    for maximum interpretability. Uses 5 features (4 original + petal_area) optimized
    for decision tree performance.
    """

    model_path: str
    model: Any = None  # Will hold the loaded sklearn model

    def __init__(self, model_path: str = "registry/prd/decision_tree.joblib"):
        super().__init__(model_path=model_path)
        self.model = self._load_model(model_path)

        logger.info(
            "DecisionTreePredictor initialized",
            model_path=self.model_path,
            model_type="DecisionTreeClassifier",
            max_depth=getattr(self.model, "max_depth", "unknown"),
            features_expected=5,  # 4 original + petal_area
        )

    def _prepare_features(self, measurements: IrisMeasurements) -> np.ndarray[Any, Any]:
        """
        Applies Decision Tree feature engineering to create 5-feature vector
        (4 original + petal_area).
        """
        # Convert single measurement to array format
        X = measurements.to_array().reshape(1, -1)

        # Apply Decision Tree feature engineering (5 features)
        feature_names = get_feature_names()
        X_engineered, feature_names_enhanced = engineer_features(X, feature_names, ModelType.DECISION_TREE)

        logger.debug(
            "Features prepared for Decision Tree",
            original_features=len(feature_names),
            engineered_features=len(feature_names_enhanced),
            shape=X_engineered.shape,
        )

        return X_engineered

    @validate_call
    def predict(self, measurements: IrisMeasurements) -> str:
        """Decision Tree implementation with minimal feature engineering for interpretability."""
        logger.debug(
            "Single prediction request",
            sepal_length=measurements.sepal_length,
            sepal_width=measurements.sepal_width,
            petal_length=measurements.petal_length,
            petal_width=measurements.petal_width,
        )

        # Prepare features
        X_features = self._prepare_features(measurements)

        # Make prediction
        prediction_array = self.model.predict(X_features)
        prediction = str(prediction_array[0])

        logger.debug(
            "Decision Tree prediction completed",
            prediction=prediction,
            features_used=X_features.shape[1],
            model_interpretability="high_with_decision_paths",
        )

        return prediction
