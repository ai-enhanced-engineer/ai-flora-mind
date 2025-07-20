"""
Base predictor interface for iris species classification.

Defines the abstract BasePredictor class that all concrete predictor
implementations must inherit from.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ai_flora_mind.configs import IrisMeasurements


class BasePredictor(BaseModel, ABC):
    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def predict(self, measurements: IrisMeasurements) -> str:
        """
        Abstract method defining the prediction interface.
        All predictors must return one of: 'setosa', 'versicolor', or 'virginica'.
        """
        pass
