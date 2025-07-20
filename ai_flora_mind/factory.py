"""
Factory for creating predictor instances.

This module provides factory functions to create the appropriate predictor
instance based on configuration settings.
"""

from ai_flora_mind.configs import ModelType, ServiceConfig
from ai_flora_mind.logging import get_logger
from ai_flora_mind.predictors import BasePredictor, DecisionTreePredictor, HeuristicPredictor, RandomForestPredictor

logger = get_logger(__name__)


def get_predictor(config: ServiceConfig) -> BasePredictor:
    """
    Factory function that instantiates the correct predictor implementation
    based on the model type specified in configuration.
    """
    logger.info("Creating predictor instance", model_type=config.model_type.value)

    match config.model_type:
        case ModelType.HEURISTIC:
            predictor = HeuristicPredictor()
            logger.info("Heuristic predictor created successfully")
            return predictor

        case ModelType.RANDOM_FOREST:
            model_path = config.get_model_path()
            if not model_path:
                raise ValueError("Random Forest model requires a file path")
            rf_predictor = RandomForestPredictor(model_path=model_path)
            logger.info(
                "Random Forest predictor created successfully",
                model_path=model_path,
                n_estimators=getattr(rf_predictor.model, "n_estimators", "unknown"),
            )
            return rf_predictor

        case ModelType.DECISION_TREE:
            model_path = config.get_model_path()
            if not model_path:
                raise ValueError("Decision Tree model requires a file path")
            dt_predictor = DecisionTreePredictor(model_path=model_path)
            logger.info(
                "Decision Tree predictor created successfully",
                model_path=model_path,
                max_depth=getattr(dt_predictor.model, "max_depth", "unknown"),
            )
            return dt_predictor

        case ModelType.XGBOOST:
            # TODO: Implement XGBoostPredictor when available
            logger.warning("XGBoost predictor not yet implemented, falling back to Heuristic")
            predictor = HeuristicPredictor()
            return predictor

        case _:
            error_msg = f"Unsupported model type: {config.model_type}"
            logger.error("Predictor creation failed", error=error_msg, model_type=config.model_type.value)
            raise ValueError(error_msg)
