"""
Configuration models for ML Production Service.

This module contains Pydantic models for configuration and data validation
across the application.
"""

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class IrisMeasurements(BaseModel):
    """
    Pydantic model representing the four standard iris flower measurements.

    This model encapsulates all required measurements for iris species classification
    and provides validation for input data.
    """

    sepal_length: float = Field(..., description="Length of the sepal in centimeters", gt=0)
    sepal_width: float = Field(..., description="Width of the sepal in centimeters", gt=0)
    petal_length: float = Field(..., description="Length of the petal in centimeters", gt=0)
    petal_width: float = Field(..., description="Width of the petal in centimeters", gt=0)

    def to_array(self) -> np.ndarray[Any, Any]:
        return np.array([self.sepal_length, self.sepal_width, self.petal_length, self.petal_width])


class ModelType(Enum):
    DECISION_TREE = "decision_tree"  # Uses basic features for interpretability
    RANDOM_FOREST = "random_forest"  # Uses all engineered features for maximum accuracy
    HEURISTIC = "heuristic"  # Uses only original features
    XGBOOST = "xgboost"  # Uses targeted high-discriminative features for gradient boosting


class ServiceConfig(BaseSettings):
    """
    Service configuration for ML Production Service.

    Reads configuration from environment variables to determine which
    predictor model to use for iris species classification.
    """

    model_type: ModelType = Field(
        default=ModelType.HEURISTIC,
        description="Type of predictor model to use for iris classification",
        alias="MPS_MODEL_TYPE",
    )

    model_config = {"env_prefix": "MPS_", "case_sensitive": False, "extra": "ignore"}

    def get_model_path(self) -> str | None:
        """Get the model path based on model type and environment.

        Returns None for models that don't require file loading (e.g., heuristic).
        Raises ValueError for models that should have files but aren't configured.
        """
        # Use consistent registry path in both local and Docker environments
        base_path = "registry/prd"

        match self.model_type:
            case ModelType.HEURISTIC:
                # Heuristic model doesn't require file loading
                return None
            case ModelType.RANDOM_FOREST:
                return f"{base_path}/random_forest.joblib"
            case ModelType.DECISION_TREE:
                return f"{base_path}/decision_tree.joblib"
            case ModelType.XGBOOST:
                return f"{base_path}/xgboost.joblib"
            case _:
                raise ValueError(f"Unknown model type: {self.model_type.value}")
