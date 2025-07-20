from typing import List, Tuple

import numpy as np
import pytest

from ai_flora_mind.predictors import BasePredictor, HeuristicPredictor

# ----------------------
# Fixture Setup
# ----------------------


@pytest.fixture
def predictor() -> HeuristicPredictor:
    return HeuristicPredictor()


@pytest.fixture
def iris_dataset() -> Tuple[List[float], List[float], List[str]]:
    """Load iris dataset directly from sklearn for integration testing."""
    from sklearn.datasets import load_iris

    # Load the iris dataset
    iris = load_iris()

    # Extract petal features (columns 2 and 3)
    petal_lengths = iris.data[:, 2].tolist()  # petal length
    petal_widths = iris.data[:, 3].tolist()  # petal width

    # Convert numeric targets to species names
    species = [iris.target_names[target] for target in iris.target]

    return petal_lengths, petal_widths, species


# ----------------------
# Abstract Base Class Tests
# ----------------------


@pytest.mark.unit
def test__base_predictor__cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="Can't instantiate abstract class BasePredictor"):
        BasePredictor()


# ----------------------
# HeuristicPredictor Initialization Tests
# ----------------------


@pytest.mark.unit
def test__heuristic_predictor__initializes_with_correct_thresholds(predictor: HeuristicPredictor) -> None:
    assert predictor.setosa_threshold == 2.0
    assert predictor.versicolor_threshold == 1.7


# ----------------------
# Single Prediction Tests
# ----------------------


@pytest.mark.unit
def test__predict__species_classification(predictor: HeuristicPredictor) -> None:
    assert predictor.predict(1.4, 0.2) == "setosa"
    assert predictor.predict(4.5, 1.5) == "versicolor"
    assert predictor.predict(6.0, 2.0) == "virginica"


@pytest.mark.unit
def test__predict__boundary_cases(predictor: HeuristicPredictor) -> None:
    # Test exact threshold boundaries
    assert predictor.predict(1.9, 0.5) == "setosa"  # Just below setosa threshold
    assert predictor.predict(2.0, 0.5) == "versicolor"  # At setosa threshold
    assert predictor.predict(3.0, 1.6) == "versicolor"  # Just below versicolor threshold
    assert predictor.predict(3.0, 1.7) == "virginica"  # At versicolor threshold


# ----------------------
# Input Validation Tests (Pydantic)
# ----------------------


@pytest.mark.unit
def test__predict__invalid_types_raise_pydantic_error(predictor: HeuristicPredictor) -> None:
    from pydantic import ValidationError

    # Test non-numeric types
    with pytest.raises(ValidationError):
        predictor.predict("invalid", 0.5)
    with pytest.raises(ValidationError):
        predictor.predict(1.0, "invalid")
    with pytest.raises(ValidationError):
        predictor.predict(None, 0.5)
    with pytest.raises(ValidationError):
        predictor.predict(1.0, None)


@pytest.mark.unit
def test__predict__zero_values_accepted(predictor: HeuristicPredictor) -> None:
    result = predictor.predict(0.0, 0.0)
    assert result == "setosa"  # Should follow Rule 1


@pytest.mark.unit
def test__predict_batch__invalid_types_raise_pydantic_error(predictor: HeuristicPredictor) -> None:
    from pydantic import ValidationError

    # Test non-list types
    with pytest.raises(ValidationError):
        predictor.predict_batch("invalid", [0.5])
    with pytest.raises(ValidationError):
        predictor.predict_batch([1.0], "invalid")
    with pytest.raises(ValidationError):
        predictor.predict_batch(None, [0.5])
    with pytest.raises(ValidationError):
        predictor.predict_batch([1.0], None)


# ----------------------
# Batch Prediction Tests
# ----------------------


@pytest.mark.unit
def test__predict_batch__multiple_input_types(predictor: HeuristicPredictor) -> None:
    expected = ["setosa", "versicolor", "virginica"]

    # Test list input
    results_list = predictor.predict_batch([1.4, 4.5, 6.0], [0.2, 1.5, 2.0])
    assert results_list == expected

    # Test numpy array input
    results_numpy = predictor.predict_batch(np.array([1.4, 4.5, 6.0]).tolist(), np.array([0.2, 1.5, 2.0]).tolist())
    assert results_numpy == expected


@pytest.mark.unit
def test__predict_batch__edge_cases(predictor: HeuristicPredictor) -> None:
    # Single sample
    assert predictor.predict_batch([1.4], [0.2]) == ["setosa"]

    # Empty arrays
    assert predictor.predict_batch([], []) == []


@pytest.mark.unit
def test__predict_batch__mismatched_array_lengths_raises_error(predictor: HeuristicPredictor) -> None:
    with pytest.raises(ValueError, match="Petal length and width arrays must have same length"):
        predictor.predict_batch([1.4, 4.5], [0.2])


# ----------------------
# Algorithm Logic Tests
# ----------------------


@pytest.mark.unit
def test__predict__algorithm_logic(predictor: HeuristicPredictor) -> None:
    # Rule precedence: Rule 1 (setosa) takes precedence
    assert predictor.predict(1.5, 0.5) == "setosa"  # Both conditions could apply, but Rule 1 wins

    # Multiple setosa samples
    setosa_samples = [(1.0, 0.1), (1.5, 0.2), (1.9, 0.3), (0.5, 1.0)]
    for length, width in setosa_samples:
        assert predictor.predict(length, width) == "setosa"

    # Multiple versicolor samples
    versicolor_samples = [(4.0, 1.3), (3.5, 1.0), (5.0, 1.6), (2.5, 0.8)]
    for length, width in versicolor_samples:
        assert predictor.predict(length, width) == "versicolor"

    # Multiple virginica samples
    virginica_samples = [(6.0, 2.0), (5.5, 1.8), (4.0, 2.5), (2.1, 1.7)]
    for length, width in virginica_samples:
        assert predictor.predict(length, width) == "virginica"


# ----------------------
# Integration with Base Class Tests
# ----------------------


@pytest.mark.unit
def test__heuristic_predictor__interface_compliance() -> None:
    predictor = HeuristicPredictor()

    # Implements base interface
    assert isinstance(predictor, BasePredictor)
    assert hasattr(predictor, "predict")
    assert hasattr(predictor, "predict_batch")

    # Return types match interface
    result = predictor.predict(1.4, 0.2)
    assert isinstance(result, str)

    batch_result = predictor.predict_batch([1.4], [0.2])
    assert isinstance(batch_result, list)
    assert all(isinstance(pred, str) for pred in batch_result)


# ----------------------
# Integration Tests
# ----------------------


@pytest.mark.integration
def test__heuristic_predictor__full_dataset_performance(
    predictor: HeuristicPredictor, iris_dataset: Tuple[List[float], List[float], List[str]]
) -> None:
    """Integration test: validate predictor performance on complete iris dataset."""
    petal_lengths, petal_widths, true_species = iris_dataset

    # Make predictions on the entire dataset
    predictions = predictor.predict_batch(petal_lengths, petal_widths)

    # Calculate accuracy metrics
    correct_predictions = sum(1 for pred, true in zip(predictions, true_species) if pred == true)
    total_samples = len(true_species)
    accuracy = correct_predictions / total_samples

    # Calculate per-species accuracy
    species_counts = {"setosa": 0, "versicolor": 0, "virginica": 0}
    species_correct = {"setosa": 0, "versicolor": 0, "virginica": 0}

    for pred, true in zip(predictions, true_species):
        species_counts[true] += 1
        if pred == true:
            species_correct[true] += 1

    species_accuracy = {species: species_correct[species] / species_counts[species] for species in species_counts}

    # Assert expected performance based on research findings
    # From research: heuristic achieves ~96% accuracy
    assert accuracy >= 0.95, f"Overall accuracy {accuracy:.3f} below expected threshold (0.95)"

    # Setosa should have perfect separation (Rule 1)
    assert species_accuracy["setosa"] == 1.0, f"Setosa accuracy {species_accuracy['setosa']:.3f} not perfect"

    # Versicolor and Virginica should have reasonable accuracy
    assert species_accuracy["versicolor"] >= 0.90, f"Versicolor accuracy {species_accuracy['versicolor']:.3f} too low"
    assert species_accuracy["virginica"] >= 0.90, f"Virginica accuracy {species_accuracy['virginica']:.3f} too low"

    # Validate dataset size and distribution
    assert total_samples == 150, f"Expected 150 samples, got {total_samples}"
    assert all(count == 50 for count in species_counts.values()), "Unequal species distribution"

    # Log performance for visibility
    print("\nDataset Performance Summary:")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{total_samples})")
    print(f"Setosa Accuracy: {species_accuracy['setosa']:.3f}")
    print(f"Versicolor Accuracy: {species_accuracy['versicolor']:.3f}")
    print(f"Virginica Accuracy: {species_accuracy['virginica']:.3f}")
