"""Configuration models for ML Production Service."""

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class IrisMeasurements(BaseModel):
    sepal_length: float = Field(..., description="Length of the sepal in centimeters", gt=0)
    sepal_width: float = Field(..., description="Width of the sepal in centimeters", gt=0)
    petal_length: float = Field(..., description="Length of the petal in centimeters", gt=0)
    petal_width: float = Field(..., description="Width of the petal in centimeters", gt=0)

    def to_array(self) -> NDArray[np.floating[Any]]:
        return np.array([self.sepal_length, self.sepal_width, self.petal_length, self.petal_width])


class ModelType(Enum):
    DECISION_TREE = "decision_tree"  # Uses basic features for interpretability
    RANDOM_FOREST = "random_forest"  # Uses all engineered features for maximum accuracy
    HEURISTIC = "heuristic"  # Uses only original features
    XGBOOST = "xgboost"  # Uses targeted high-discriminative features for gradient boosting


class ServiceConfig(BaseSettings):
    model_type: ModelType = Field(
        default=ModelType.HEURISTIC,
        description="Type of predictor model to use for iris classification",
        alias="MPS_MODEL_TYPE",
    )

    model_config = {"env_prefix": "MPS_", "case_sensitive": False, "extra": "ignore"}

    def get_model_path(self) -> str | None:
        """Get the model path based on model type."""
        base_path = "registry/prd"

        match self.model_type:
            case ModelType.HEURISTIC:
                return None
            case ModelType.RANDOM_FOREST:
                return f"{base_path}/random_forest.joblib"
            case ModelType.DECISION_TREE:
                return f"{base_path}/decision_tree.joblib"
            case ModelType.XGBOOST:
                return f"{base_path}/xgboost.joblib"
            case _:
                raise ValueError(f"Unknown model type: {self.model_type.value}")
