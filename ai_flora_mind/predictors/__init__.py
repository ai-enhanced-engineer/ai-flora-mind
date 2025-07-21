"""
Iris Species Predictors

Unified interface for iris prediction algorithms with abstract BasePredictor
class and concrete implementations for heuristic and machine learning models.
"""

from ai_flora_mind.predictors.base import BasePredictor
from ai_flora_mind.predictors.decision_tree import DecisionTreePredictor
from ai_flora_mind.predictors.heuristic import HeuristicPredictor
from ai_flora_mind.predictors.random_forest import RandomForestPredictor
from ai_flora_mind.predictors.xgboost import XGBoostPredictor

__all__ = [
    "BasePredictor",
    "DecisionTreePredictor",
    "HeuristicPredictor",
    "RandomForestPredictor",
    "XGBoostPredictor",
]
