"""Unified ML model predictor base class."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from ml_production_service.configs import IrisMeasurements, ModelType
from ml_production_service.features import engineer_features, get_feature_names
from ml_production_service.logging import get_logger
from ml_production_service.predictors.base import BasePredictor

logger = get_logger(__name__)


class MLModelPredictor(BasePredictor):
    """Base class for all ML model predictors with shared functionality."""

    model_path: str
    model: Any = None
    model_type: ModelType

    def __init__(self, model_path: str, model_type: ModelType):
        super().__init__(model_path=model_path, model_type=model_type)
        self.model = self._load_model(model_path)

        # Log once at initialization
        logger.info(
            "ML predictor initialized",
            model_type=model_type.value,
            model_path=model_path,
            model_class=self.model.__class__.__name__,
        )

    def _prepare_features(self, measurements: IrisMeasurements) -> NDArray[np.floating[Any]]:
        """Apply feature engineering based on model type."""
        X = measurements.to_array().reshape(1, -1)
        feature_names = get_feature_names()
        X_engineered, _ = engineer_features(X, feature_names, self.model_type)
        return X_engineered

    @validate_call
    def predict(self, measurements: IrisMeasurements) -> str:
        """Make prediction using the loaded model."""
        logger.debug(
            "Prediction request",
            model=self.model_type.value,
            petal_length=measurements.petal_length,
            petal_width=measurements.petal_width,
        )

        X_features = self._prepare_features(measurements)
        prediction_array = self.model.predict(X_features)
        prediction_numeric = int(prediction_array[0])
        prediction = self.SPECIES_MAP[prediction_numeric]

        logger.debug("Prediction completed", model=self.model_type.value, prediction=prediction)
        return prediction
