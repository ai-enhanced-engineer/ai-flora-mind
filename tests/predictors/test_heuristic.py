"""
Tests for HeuristicPredictor.

Tests the rule-based iris species classification predictor including
initialization, prediction logic, algorithm behavior, and integration tests.
"""

from typing import List, Tuple

import pytest

from ml_production_service.configs import IrisMeasurements
from ml_production_service.predictors import BasePredictor, HeuristicPredictor

# ----------------------
# Initialization Tests
# ----------------------


@pytest.mark.unit
def test__heuristic_predictor__initializes_with_correct_thresholds(heuristic_predictor: HeuristicPredictor) -> None:
    assert heuristic_predictor.setosa_threshold == 2.0
    assert heuristic_predictor.versicolor_threshold == 1.7


# ----------------------
# Single Prediction Tests
# ----------------------


@pytest.mark.unit
def test__predict__species_classification(heuristic_predictor: HeuristicPredictor) -> None:
    # Create test measurements
    setosa_measurements = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=1.4, petal_width=0.2)
    versicolor_measurements = IrisMeasurements(sepal_length=6.0, sepal_width=3.0, petal_length=4.5, petal_width=1.5)
    virginica_measurements = IrisMeasurements(sepal_length=7.0, sepal_width=3.0, petal_length=6.0, petal_width=2.0)

    assert heuristic_predictor.predict(setosa_measurements) == "setosa"
    assert heuristic_predictor.predict(versicolor_measurements) == "versicolor"
    assert heuristic_predictor.predict(virginica_measurements) == "virginica"


@pytest.mark.unit
def test__predict__boundary_cases(heuristic_predictor: HeuristicPredictor) -> None:
    # Test exact threshold boundaries
    measurements_1 = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=1.9, petal_width=0.5)
    measurements_2 = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=2.0, petal_width=0.5)
    measurements_3 = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=3.0, petal_width=1.6)
    measurements_4 = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=3.0, petal_width=1.7)

    assert heuristic_predictor.predict(measurements_1) == "setosa"  # Just below setosa threshold
    assert heuristic_predictor.predict(measurements_2) == "versicolor"  # At setosa threshold
    assert heuristic_predictor.predict(measurements_3) == "versicolor"  # Just below versicolor threshold
    assert heuristic_predictor.predict(measurements_4) == "virginica"  # At versicolor threshold


# ----------------------
# Input Validation Tests (Pydantic)
# ----------------------


@pytest.mark.unit
def test__predict__invalid_types_raise_pydantic_error(heuristic_predictor: HeuristicPredictor) -> None:
    from pydantic import ValidationError

    # Test invalid IrisMeasurements
    with pytest.raises(ValidationError):
        heuristic_predictor.predict("invalid")  # Wrong type entirely
    with pytest.raises(ValidationError):
        # Create invalid measurements with string values
        IrisMeasurements(sepal_length="invalid", sepal_width=3.0, petal_length=1.0, petal_width=0.5)


@pytest.mark.unit
def test__predict__small_values_accepted(heuristic_predictor: HeuristicPredictor) -> None:
    # Test with very small but positive values (IrisMeasurements requires gt=0)
    measurements = IrisMeasurements(sepal_length=0.1, sepal_width=0.1, petal_length=0.1, petal_width=0.1)
    result = heuristic_predictor.predict(measurements)
    assert result == "setosa"  # Should follow Rule 1


# ----------------------
# Algorithm Logic Tests
# ----------------------


@pytest.mark.unit
def test__predict__algorithm_logic(heuristic_predictor: HeuristicPredictor) -> None:
    # Rule precedence: Rule 1 (setosa) takes precedence
    measurements = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=1.5, petal_width=0.5)
    assert heuristic_predictor.predict(measurements) == "setosa"  # Both conditions could apply, but Rule 1 wins

    # Multiple setosa samples
    setosa_samples = [(1.0, 0.1), (1.5, 0.2), (1.9, 0.3), (0.5, 1.0)]
    for petal_length, petal_width in setosa_samples:
        measurements = IrisMeasurements(
            sepal_length=5.0, sepal_width=3.0, petal_length=petal_length, petal_width=petal_width
        )
        assert heuristic_predictor.predict(measurements) == "setosa"

    # Multiple versicolor samples
    versicolor_samples = [(4.0, 1.3), (3.5, 1.0), (5.0, 1.6), (2.5, 0.8)]
    for petal_length, petal_width in versicolor_samples:
        measurements = IrisMeasurements(
            sepal_length=5.0, sepal_width=3.0, petal_length=petal_length, petal_width=petal_width
        )
        assert heuristic_predictor.predict(measurements) == "versicolor"

    # Multiple virginica samples
    virginica_samples = [(6.0, 2.0), (5.5, 1.8), (4.0, 2.5), (2.1, 1.7)]
    for petal_length, petal_width in virginica_samples:
        measurements = IrisMeasurements(
            sepal_length=5.0, sepal_width=3.0, petal_length=petal_length, petal_width=petal_width
        )
        assert heuristic_predictor.predict(measurements) == "virginica"


# ----------------------
# Interface Compliance Tests
# ----------------------


@pytest.mark.unit
def test__heuristic_predictor__interface_compliance() -> None:
    heuristic_predictor = HeuristicPredictor()

    # Implements base interface
    assert isinstance(heuristic_predictor, BasePredictor)
    assert hasattr(heuristic_predictor, "predict")

    # Return types match interface
    measurements = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=1.4, petal_width=0.2)
    result = heuristic_predictor.predict(measurements)
    assert isinstance(result, str)


# ----------------------
# Integration Tests
# ----------------------


@pytest.mark.integration
def test__heuristic_predictor__sample_dataset_performance(
    heuristic_predictor: HeuristicPredictor, iris_dataset: Tuple[List[IrisMeasurements], List[str]]
) -> None:
    """Integration test: validate predictor performance on sample of iris dataset."""
    measurements, true_species = iris_dataset

    # Make individual predictions on sample of dataset (first 30 samples)
    test_measurements = measurements[:30]
    test_species = true_species[:30]
    predictions = [heuristic_predictor.predict(m) for m in test_measurements]

    # Calculate accuracy metrics
    correct_predictions = sum(1 for pred, true in zip(predictions, test_species) if pred == true)
    total_samples = len(test_species)
    accuracy = correct_predictions / total_samples

    # Assert expected performance on sample
    assert accuracy >= 0.90, f"Sample accuracy {accuracy:.3f} below expected threshold (0.90)"

    # Verify we got valid predictions
    assert all(pred in ["setosa", "versicolor", "virginica"] for pred in predictions)
    assert len(predictions) == 30
