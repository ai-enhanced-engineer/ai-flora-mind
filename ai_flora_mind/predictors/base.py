"""
Base predictor interface for iris species classification.

Defines the abstract BasePredictor class that all concrete predictor
implementations must inherit from.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel

from ai_flora_mind.configs import IrisMeasurements
from ai_flora_mind.logging import get_logger

logger = get_logger(__name__)


class BasePredictor(BaseModel, ABC):
    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def predict(self, measurements: IrisMeasurements) -> str:
        """
        Abstract method defining the prediction interface.
        All predictors must return one of: 'setosa', 'versicolor', or 'virginica'.
        """
        pass

    def _load_model(self, model_path: str) -> Any:
        """Common model loading functionality for ML predictors."""
        path = Path(model_path)

        if not path.exists():
            error_msg = f"Model file not found: {model_path}"
            logger.error("Model loading failed", error=error_msg, model_path=str(path))
            raise FileNotFoundError(error_msg)

        try:
            model = joblib.load(path)
            logger.info(
                "Model loaded successfully",
                model_path=str(path),
                file_size=path.stat().st_size,
                model_class=model.__class__.__name__,
            )
            return model

        except Exception as e:
            error_msg = f"Failed to load model from {model_path}: {str(e)}"
            logger.error("Model loading error", error=error_msg, exception_type=type(e).__name__)
            raise RuntimeError(error_msg) from e
