"""
Rule-Based Heuristic Iris Classifier Package

This package implements a simple rule-based classifier for the Iris dataset
based on comprehensive EDA findings.
"""

from .iris_heuristic_classifier import classify_batch, classify_iris_heuristic

__all__ = ["classify_iris_heuristic", "classify_batch"]
