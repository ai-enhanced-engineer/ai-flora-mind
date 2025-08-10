"""Factory for creating predictor instances."""

from ml_production_service.configs import ModelType, ServiceConfig
from ml_production_service.logging import get_logger
from ml_production_service.predictors import BasePredictor, HeuristicPredictor, MLModelPredictor

logger = get_logger(__name__)


def get_predictor(config: ServiceConfig) -> BasePredictor:
    """Create predictor instance based on configuration."""
    if config.model_type == ModelType.HEURISTIC:
        return HeuristicPredictor()

    # All ML models use the unified MLModelPredictor
    model_path = config.get_model_path()
    if not model_path:
        raise ValueError(f"{config.model_type.value} requires a model file")

    return MLModelPredictor(model_path=model_path, model_type=config.model_type)
