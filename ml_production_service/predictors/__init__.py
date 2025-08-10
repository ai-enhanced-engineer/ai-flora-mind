"""
Iris Species Predictors

Unified interface for iris prediction algorithms with abstract BasePredictor
class and concrete implementations for heuristic and machine learning models.
"""

from ml_production_service.predictors.base import BasePredictor
from ml_production_service.predictors.decision_tree import DecisionTreePredictor
from ml_production_service.predictors.heuristic import HeuristicPredictor
from ml_production_service.predictors.random_forest import RandomForestPredictor
from ml_production_service.predictors.xgboost import XGBoostPredictor

__all__ = [
    "BasePredictor",
    "DecisionTreePredictor",
    "HeuristicPredictor",
    "RandomForestPredictor",
    "XGBoostPredictor",
]
