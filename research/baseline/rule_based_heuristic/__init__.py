"""
Rule-Based Heuristic Iris Classifier Package

This package implements a simple rule-based classifier for the Iris dataset
based on comprehensive EDA findings.
"""

from .data import load_iris_data
from .evaluation import evaluate_model, log_performance_summary
from .iris_heuristic_classifier import classify_batch, classify_iris_heuristic

__all__ = [
    'classify_iris_heuristic',
    'classify_batch', 
    'load_iris_data',
    'evaluate_model',
    'log_performance_summary'
]