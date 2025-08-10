"""Rule-based iris classifier with perfect Setosa separation."""

from pydantic import validate_call

from ml_production_service.configs import IrisMeasurements
from ml_production_service.logging import get_logger
from ml_production_service.predictors.base import BasePredictor

logger = get_logger(__name__)


class HeuristicPredictor(BasePredictor):
    """EDA-based thresholds: petal_length < 2.0 for Setosa, petal_width < 1.7 for Versicolor."""

    setosa_threshold: float = 2.0
    versicolor_threshold: float = 1.7

    @validate_call
    def predict(self, measurements: IrisMeasurements) -> str:
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
