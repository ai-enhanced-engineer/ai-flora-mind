"""
Heuristic predictor for iris species classification.

Implements a simple rule-based classifier using only petal measurements
for species prediction with perfect Setosa separation.
"""

from pydantic import validate_call

from ml_production_service.configs import IrisMeasurements
from ml_production_service.logging import get_logger
from ml_production_service.predictors.base import BasePredictor

logger = get_logger(__name__)


class HeuristicPredictor(BasePredictor):
    """
    Heuristic predictor using simple decision rules.

    Based on EDA analysis showing perfect Setosa separation using petal_length < 2.0
    and reasonable Versicolor/Virginica separation using petal_width < 1.7.
    """

    setosa_threshold: float = 2.0
    versicolor_threshold: float = 1.7

    @validate_call
    def predict(self, measurements: IrisMeasurements) -> str:
        """
        Predict using measurements (heuristic only uses petal measurements).

        Decision Rules:
        1. If petal_length < 2.0 → setosa (perfect separation)
        2. If petal_width < 1.7 → versicolor
        3. Otherwise → virginica

        Args:
            measurements: Complete iris measurements

        Returns:
            Predicted species: 'setosa', 'versicolor', or 'virginica'
        """
        logger.debug(
            "Classifying iris sample", petal_length=measurements.petal_length, petal_width=measurements.petal_width
        )

        # Rule 1: Perfect Setosa separation
        if measurements.petal_length < self.setosa_threshold:
            prediction = "setosa"
            logger.debug(
                "Applied rule 1 - Setosa separation",
                rule=f"petal_length < {self.setosa_threshold}",
                prediction=prediction,
            )
            return prediction

        # Rule 2: Versicolor vs Virginica separation
        elif measurements.petal_width < self.versicolor_threshold:
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
