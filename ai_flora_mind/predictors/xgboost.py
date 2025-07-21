"""
XGBoost predictor for iris species classification.

Implements a gradient boosting predictor using pre-trained XGBoost models
with targeted feature engineering for maximum accuracy.
"""

from typing import Any

import numpy as np
from pydantic import validate_call

from ai_flora_mind.configs import IrisMeasurements, ModelType
from ai_flora_mind.features import engineer_features, get_feature_names
from ai_flora_mind.logging import get_logger
from ai_flora_mind.predictors.base import BasePredictor

logger = get_logger(__name__)


class XGBoostPredictor(BasePredictor):
    """
    XGBoost predictor for iris species classification.

    Loads a pre-trained XGBoost model and applies targeted feature engineering
    for maximum accuracy. Uses 9 features (4 original + 5 engineered) optimized
    for gradient boosting performance while preventing overfitting.
    """

    model_path: str
    model: Any = None  # Will hold the loaded XGBoost model

    def __init__(self, model_path: str = "research/models/xgboost_optimized_2025_07_20_005952.joblib"):
        super().__init__(model_path=model_path)
        self.model = self._load_model(model_path)

        logger.info(
            "XGBoostPredictor initialized",
            model_path=self.model_path,
            model_type="XGBClassifier",
            n_estimators=getattr(self.model, "n_estimators", "unknown"),
            max_depth=getattr(self.model, "max_depth", "unknown"),
            features_expected=9,  # 4 original + 5 engineered
        )

    def _prepare_features(self, measurements: IrisMeasurements) -> np.ndarray[Any, Any]:
        """
        Applies XGBoost feature engineering to create 9-feature vector
        (4 original + 5 high-discriminative engineered features).
        """
        # Convert single measurement to array format
        X = measurements.to_array().reshape(1, -1)

        # Apply XGBoost feature engineering (9 features)
        feature_names = get_feature_names()
        X_engineered, feature_names_enhanced = engineer_features(X, feature_names, ModelType.XGBOOST)

        logger.debug(
            "Features prepared for XGBoost",
            original_features=len(feature_names),
            engineered_features=len(feature_names_enhanced),
            shape=X_engineered.shape,
        )

        return X_engineered

    @validate_call
    def predict(self, measurements: IrisMeasurements) -> str:
        """XGBoost implementation with targeted feature engineering for maximum accuracy."""
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

        # Convert numeric prediction to class name
        class_names = ["setosa", "versicolor", "virginica"]
        prediction_idx = int(prediction_array[0])
        prediction = class_names[prediction_idx]

        logger.debug(
            "XGBoost prediction completed",
            prediction=prediction,
            features_used=X_features.shape[1],
            model_performance="theoretical_maximum_98_99_percent",
        )

        return prediction
