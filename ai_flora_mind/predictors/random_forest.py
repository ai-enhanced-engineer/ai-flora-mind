"""
Random Forest predictor for iris species classification.

Implements a machine learning predictor using pre-trained Random Forest models
with comprehensive feature engineering for maximum accuracy.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import validate_call

from ai_flora_mind.configs import IrisMeasurements, ModelType
from ai_flora_mind.features import engineer_features, get_feature_names
from ai_flora_mind.logging import get_logger
from ai_flora_mind.predictors.base import BasePredictor

logger = get_logger(__name__)


class RandomForestPredictor(BasePredictor):
    """
    Random Forest predictor for iris species classification.

    Loads a pre-trained Random Forest model and applies comprehensive feature engineering
    for maximum accuracy. Uses all 14 features (4 original + 10 engineered) as identified
    in EDA analysis.
    """

    model_path: str
    model: Any = None  # Will hold the loaded sklearn model

    def __init__(self, model_path: str = "research/models/random_forest_regularized_2025_07_19_234849.joblib"):
        super().__init__(model_path=model_path)
        self.model = self._load_model()

        logger.info(
            "RandomForestPredictor initialized",
            model_path=self.model_path,
            model_type="RandomForestClassifier",
            n_estimators=getattr(self.model, "n_estimators", "unknown"),
            features_expected=14,  # 4 original + 10 engineered
        )

    def _load_model(self) -> Any:
        model_path = Path(self.model_path)

        if not model_path.exists():
            error_msg = f"Model file not found: {self.model_path}"
            logger.error("Model loading failed", error=error_msg, model_path=str(model_path))
            raise FileNotFoundError(error_msg)

        try:
            model = joblib.load(model_path)
            logger.info(
                "Model loaded successfully",
                model_path=str(model_path),
                file_size=model_path.stat().st_size,
                model_class=model.__class__.__name__,
            )
            return model

        except Exception as e:
            error_msg = f"Failed to load model from {self.model_path}: {str(e)}"
            logger.error("Model loading error", error=error_msg, exception_type=type(e).__name__)
            raise RuntimeError(error_msg) from e

    def _prepare_features(self, measurements: IrisMeasurements) -> np.ndarray[Any, Any]:
        """
        Applies Random Forest feature engineering to create 14-feature vector
        (4 original + 10 engineered features).
        """
        # Convert single measurement to array format
        X = measurements.to_array().reshape(1, -1)

        # Apply Random Forest feature engineering (all 14 features)
        feature_names = get_feature_names()
        X_engineered, feature_names_enhanced = engineer_features(X, feature_names, ModelType.RANDOM_FOREST)

        logger.debug(
            "Features prepared for Random Forest",
            original_features=len(feature_names),
            engineered_features=len(feature_names_enhanced),
            shape=X_engineered.shape,
        )

        return X_engineered

    @validate_call
    def predict(self, measurements: IrisMeasurements) -> str:
        """Random Forest implementation with comprehensive feature engineering."""
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
            "Random Forest prediction completed",
            prediction=prediction,
            features_used=X_features.shape[1],
            model_confidence="available_via_predict_proba",
        )

        return prediction
