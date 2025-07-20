"""
Iris Species Predictors

Unified interface for iris prediction algorithms with abstract BasePredictor
class and concrete HeuristicPredictor implementation.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pydantic import BaseModel, validate_call

from ai_flora_mind.logging import get_logger

logger = get_logger(__name__)


class BasePredictor(BaseModel, ABC):
    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def predict(self, petal_length: float, petal_width: float) -> str:
        pass

    @abstractmethod
    def predict_batch(self, petal_lengths: List[float], petal_widths: List[float]) -> List[str]:
        pass


class HeuristicPredictor(BasePredictor):
    setosa_threshold: float = 2.0
    versicolor_threshold: float = 1.7

    @validate_call
    def predict(self, petal_length: float, petal_width: float) -> str:
        logger.debug("Classifying iris sample", petal_length=petal_length, petal_width=petal_width)

        # Rule 1: Perfect Setosa separation
        if petal_length < self.setosa_threshold:
            prediction = "setosa"
            logger.debug(
                "Applied rule 1 - Setosa separation",
                rule=f"petal_length < {self.setosa_threshold}",
                prediction=prediction,
            )
            return prediction

        # Rule 2: Versicolor vs Virginica separation
        elif petal_width < self.versicolor_threshold:
            prediction = "versicolor"
            logger.debug(
                "Applied rule 2 - Versicolor threshold",
                rule=f"petal_width < {self.versicolor_threshold}",
                prediction=prediction,
            )
            return prediction

        # Rule 3: Default to Virginica
        else:
            prediction = "virginica"
            logger.debug("Applied rule 3 - Virginica default", rule="else (large petals)", prediction=prediction)
            return prediction

    @validate_call
    def predict_batch(self, petal_lengths: List[float], petal_widths: List[float]) -> List[str]:
        petal_lengths_array = np.array(petal_lengths)
        petal_widths_array = np.array(petal_widths)

        logger.info("Starting batch classification", sample_count=len(petal_lengths))

        if len(petal_lengths) != len(petal_widths):
            error_msg = "Petal length and width arrays must have same length"
            logger.error(
                "Batch validation failed",
                error=error_msg,
                petal_lengths_count=len(petal_lengths),
                petal_widths_count=len(petal_widths),
            )
            raise ValueError(error_msg)

        predictions = []
        for i, (pl, pw) in enumerate(zip(petal_lengths_array, petal_widths_array)):
            predictions.append(self.predict(float(pl), float(pw)))
            if (i + 1) % 50 == 0:
                logger.debug("Batch classification progress", processed=i + 1, total=len(petal_lengths))

        logger.info(
            "Batch classification completed", total_samples=len(predictions), predictions_generated=len(predictions)
        )

        return predictions
