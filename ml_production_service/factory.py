"""Factory for creating predictor instances."""

from ml_production_service.configs import ModelType, ServiceConfig
from ml_production_service.logging import get_logger
from ml_production_service.predictors import BasePredictor, HeuristicPredictor, MLModelPredictor

logger = get_logger(__name__)

# Model types that require file-based models
ML_MODEL_TYPES = {
    ModelType.DECISION_TREE,
    ModelType.RANDOM_FOREST,
    ModelType.XGBOOST,
}


def get_predictor(config: ServiceConfig) -> BasePredictor:
    """Create predictor instance based on configuration."""
    if config.model_type == ModelType.HEURISTIC:
        return HeuristicPredictor()

    # Validate that the model type requires a file-based model
    if config.model_type not in ML_MODEL_TYPES:
        model_type_value = config.model_type.value if hasattr(config.model_type, "value") else str(config.model_type)
        raise ValueError(
            f"Unknown model type '{model_type_value}'. "
            f"Supported types: {', '.join(sorted([mt.value for mt in ML_MODEL_TYPES | {ModelType.HEURISTIC}]))}"
        )

    # All ML models use the unified MLModelPredictor
    model_path = config.get_model_path()
    if not model_path:
        raise ValueError(
            f"Model type '{config.model_type.value}' requires a model file path. "
            "Ensure the model file exists in the registry."
        )

    return MLModelPredictor(model_path=model_path, model_type=config.model_type)
