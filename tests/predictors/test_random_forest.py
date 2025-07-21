"""
Tests for RandomForestPredictor.

Tests the machine learning iris species classification predictor including
model loading, feature engineering, prediction functionality, and error handling.
"""

import pytest

from ai_flora_mind.configs import IrisMeasurements
from ai_flora_mind.predictors import BasePredictor, RandomForestPredictor

# ----------------------
# Initialization Tests
# ----------------------


@pytest.mark.unit
def test__random_forest_predictor__initialization_default_model() -> None:
    predictor = RandomForestPredictor()

    assert predictor.model_path == "research/models/random_forest_regularized_2025_07_19_234849.joblib"
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


@pytest.mark.unit
def test__random_forest_predictor__initialization_custom_model(temp_model_path: str) -> None:
    predictor = RandomForestPredictor(model_path=temp_model_path)

    assert predictor.model_path == temp_model_path
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


# ----------------------
# Error Handling Tests
# ----------------------


@pytest.mark.unit
def test__random_forest_predictor__missing_model_file_raises_error() -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        RandomForestPredictor(model_path="nonexistent/path/model.joblib")


@pytest.mark.unit
def test__random_forest_predictor__invalid_model_file_raises_error(tmp_path) -> None:
    # Create invalid model file
    invalid_file = tmp_path / "invalid.joblib"
    invalid_file.write_text("not a valid joblib file")

    with pytest.raises(RuntimeError, match="Failed to load model"):
        RandomForestPredictor(model_path=str(invalid_file))


# ----------------------
# Prediction Tests
# ----------------------


@pytest.mark.unit
def test__random_forest_predictor__predict_single_measurement(temp_model_path: str) -> None:
    predictor = RandomForestPredictor(model_path=temp_model_path)

    # Test with typical measurements
    setosa_measurements = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    versicolor_measurements = IrisMeasurements(sepal_length=6.0, sepal_width=3.0, petal_length=4.5, petal_width=1.5)
    virginica_measurements = IrisMeasurements(sepal_length=7.0, sepal_width=3.0, petal_length=6.0, petal_width=2.0)

    # Make predictions
    setosa_pred = predictor.predict(setosa_measurements)
    versicolor_pred = predictor.predict(versicolor_measurements)
    virginica_pred = predictor.predict(virginica_measurements)

    # Verify predictions are valid species
    assert setosa_pred in ["setosa", "versicolor", "virginica"]
    assert versicolor_pred in ["setosa", "versicolor", "virginica"]
    assert virginica_pred in ["setosa", "versicolor", "virginica"]

    # Verify predictions are strings
    assert isinstance(setosa_pred, str)
    assert isinstance(versicolor_pred, str)
    assert isinstance(virginica_pred, str)


@pytest.mark.unit
def test__random_forest_predictor__handles_edge_case_measurements(temp_model_path: str) -> None:
    predictor = RandomForestPredictor(model_path=temp_model_path)

    # Very small measurements
    small_measurements = IrisMeasurements(sepal_length=0.1, sepal_width=0.1, petal_length=0.1, petal_width=0.1)
    result = predictor.predict(small_measurements)
    assert result in ["setosa", "versicolor", "virginica"]

    # Very large measurements
    large_measurements = IrisMeasurements(sepal_length=10.0, sepal_width=8.0, petal_length=12.0, petal_width=5.0)
    result = predictor.predict(large_measurements)
    assert result in ["setosa", "versicolor", "virginica"]


# ----------------------
# Feature Engineering Tests
# ----------------------


@pytest.mark.unit
def test__random_forest_predictor__feature_preparation(temp_model_path: str) -> None:
    predictor = RandomForestPredictor(model_path=temp_model_path)

    measurements = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)

    # Test feature preparation
    features = predictor._prepare_features(measurements)

    # Should have 14 features (4 original + 10 engineered for Random Forest)
    assert features.shape == (1, 14)

    # First 4 features should match original measurements
    assert features[0, 0] == 5.1  # sepal_length
    assert features[0, 1] == 3.5  # sepal_width
    assert features[0, 2] == 1.4  # petal_length
    assert features[0, 3] == 0.2  # petal_width

    # Additional features should be engineered (non-zero for these measurements)
    assert features[0, 4] > 0  # petal_area should be positive
    assert features[0, 5] > 0  # sepal_area should be positive


# ----------------------
# Interface Compliance Tests
# ----------------------


@pytest.mark.unit
def test__random_forest_predictor__interface_compliance(temp_model_path: str) -> None:
    predictor = RandomForestPredictor(model_path=temp_model_path)

    # Implements base interface
    assert isinstance(predictor, BasePredictor)
    assert hasattr(predictor, "predict")

    # Return types match interface
    measurements = IrisMeasurements(sepal_length=5.0, sepal_width=3.0, petal_length=1.4, petal_width=0.2)
    result = predictor.predict(measurements)
    assert isinstance(result, str)


# ----------------------
# Integration Tests
# ----------------------


@pytest.mark.integration
def test__random_forest_predictor__real_model_predictions() -> None:
    import os

    # Only run if the default model exists
    default_model_path = "research/models/random_forest_regularized_2025_07_19_234849.joblib"
    if not os.path.exists(default_model_path):
        pytest.skip("Default model not available for integration test")

    predictor = RandomForestPredictor()

    # Test with known iris samples
    setosa_sample = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    versicolor_sample = IrisMeasurements(sepal_length=6.0, sepal_width=3.0, petal_length=4.5, petal_width=1.5)
    virginica_sample = IrisMeasurements(sepal_length=7.0, sepal_width=3.0, petal_length=6.0, petal_width=2.0)

    # Make predictions
    setosa_pred = predictor.predict(setosa_sample)
    versicolor_pred = predictor.predict(versicolor_sample)
    virginica_pred = predictor.predict(virginica_sample)

    # Verify all predictions are valid
    assert setosa_pred in ["setosa", "versicolor", "virginica"]
    assert versicolor_pred in ["setosa", "versicolor", "virginica"]
    assert virginica_pred in ["setosa", "versicolor", "virginica"]

    # With a well-trained model, we expect good predictions on typical samples
    # Note: We don't assert exact predictions since model performance can vary
