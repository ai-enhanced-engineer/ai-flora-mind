"""
Tests for DecisionTreePredictor.

Tests the machine learning iris species classification predictor including
model loading, feature engineering, prediction functionality, and error handling.
"""

import pytest

from ml_production_service.configs import IrisMeasurements
from ml_production_service.predictors import BasePredictor, DecisionTreePredictor

# ----------------------
# Initialization Tests
# ----------------------


@pytest.mark.unit
def test__decision_tree_predictor__initialization_default_model() -> None:
    predictor = DecisionTreePredictor()

    assert predictor.model_path == "registry/prd/decision_tree.joblib"
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


@pytest.mark.unit
def test__decision_tree_predictor__initialization_custom_model(temp_decision_tree_model_path: str) -> None:
    predictor = DecisionTreePredictor(model_path=temp_decision_tree_model_path)

    assert predictor.model_path == temp_decision_tree_model_path
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


# ----------------------
# Error Handling Tests
# ----------------------


@pytest.mark.unit
def test__decision_tree_predictor__missing_model_file_raises_error() -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        DecisionTreePredictor(model_path="nonexistent/path/model.joblib")


@pytest.mark.unit
def test__decision_tree_predictor__invalid_model_file_raises_error(tmp_path) -> None:
    # Create invalid model file
    invalid_file = tmp_path / "invalid.joblib"
    invalid_file.write_text("not a valid joblib file")

    with pytest.raises(RuntimeError, match="Failed to load model"):
        DecisionTreePredictor(model_path=str(invalid_file))


# ----------------------
# Prediction Tests
# ----------------------


@pytest.mark.unit
def test__decision_tree_predictor__predict_single_measurement(temp_decision_tree_model_path: str) -> None:
    predictor = DecisionTreePredictor(model_path=temp_decision_tree_model_path)

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
def test__decision_tree_predictor__handles_edge_case_measurements(temp_decision_tree_model_path: str) -> None:
    predictor = DecisionTreePredictor(model_path=temp_decision_tree_model_path)

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
def test__decision_tree_predictor__feature_preparation(temp_decision_tree_model_path: str) -> None:
    predictor = DecisionTreePredictor(model_path=temp_decision_tree_model_path)

    measurements = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)

    # Test feature preparation
    features = predictor._prepare_features(measurements)

    # Should have 5 features (4 original + 1 engineered for Decision Tree)
    assert features.shape == (1, 5)

    # First 4 features should match original measurements
    assert features[0, 0] == 5.1  # sepal_length
    assert features[0, 1] == 3.5  # sepal_width
    assert features[0, 2] == 1.4  # petal_length
    assert features[0, 3] == 0.2  # petal_width

    # 5th feature should be petal_area (1.4 * 0.2 = 0.28)
    assert features[0, 4] == pytest.approx(0.28, rel=1e-2)


# ----------------------
# Interface Compliance Tests
# ----------------------


@pytest.mark.unit
def test__decision_tree_predictor__interface_compliance(temp_decision_tree_model_path: str) -> None:
    predictor = DecisionTreePredictor(model_path=temp_decision_tree_model_path)

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
def test__decision_tree_predictor__production_model_predictions() -> None:
    import os

    # Only run if the production model exists
    production_model_path = "registry/prd/decision_tree.joblib"
    if not os.path.exists(production_model_path):
        pytest.skip("Production model not available for integration test")

    predictor = DecisionTreePredictor(model_path=production_model_path)

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
