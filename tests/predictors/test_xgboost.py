"""
Tests for XGBoostPredictor.

Tests the gradient boosting iris species classification predictor including
model loading, feature engineering, prediction functionality, and error handling.
"""

import os

import pytest

from ai_flora_mind.configs import IrisMeasurements
from ai_flora_mind.predictors import BasePredictor, XGBoostPredictor

# ----------------------
# Initialization Tests
# ----------------------


@pytest.mark.unit
def test__xgboost_predictor__initialization_default_model() -> None:
    predictor = XGBoostPredictor()

    assert predictor.model_path == "registry/prd/xgboost.joblib"
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


@pytest.mark.unit
def test__xgboost_predictor__initialization_custom_model(temp_xgboost_model_path: str) -> None:
    predictor = XGBoostPredictor(model_path=temp_xgboost_model_path)

    assert predictor.model_path == temp_xgboost_model_path
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


# ----------------------
# Error Handling Tests
# ----------------------


@pytest.mark.unit
def test__xgboost_predictor__initialization_file_not_found() -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        XGBoostPredictor(model_path="nonexistent/model.joblib")

    assert "Model file not found" in str(exc_info.value)


@pytest.mark.unit
def test__xgboost_predictor__initialization_invalid_model_file(tmp_path) -> None:
    invalid_model_path = tmp_path / "invalid_model.joblib"
    invalid_model_path.write_text("invalid content")

    with pytest.raises(RuntimeError) as exc_info:
        XGBoostPredictor(model_path=str(invalid_model_path))

    assert "Failed to load model" in str(exc_info.value)


# ----------------------
# Interface Tests
# ----------------------


@pytest.mark.unit
def test__xgboost_predictor__implements_base_predictor() -> None:
    predictor = XGBoostPredictor()
    assert isinstance(predictor, BasePredictor)


@pytest.mark.unit
def test__xgboost_predictor__has_predict_method() -> None:
    predictor = XGBoostPredictor()
    assert hasattr(predictor, "predict")
    assert callable(predictor.predict)


# ----------------------
# Prediction Tests
# ----------------------


@pytest.mark.unit
def test__xgboost_predictor__predict_setosa(xgboost_predictor: XGBoostPredictor) -> None:
    setosa_iris = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)

    prediction = xgboost_predictor.predict(setosa_iris)

    assert isinstance(prediction, str)
    assert prediction in ["setosa", "versicolor", "virginica"]


@pytest.mark.unit
def test__xgboost_predictor__predict_versicolor(xgboost_predictor: XGBoostPredictor) -> None:
    versicolor_iris = IrisMeasurements(sepal_length=5.9, sepal_width=3.0, petal_length=4.2, petal_width=1.5)

    prediction = xgboost_predictor.predict(versicolor_iris)

    assert isinstance(prediction, str)
    assert prediction in ["setosa", "versicolor", "virginica"]


@pytest.mark.unit
def test__xgboost_predictor__predict_virginica(xgboost_predictor: XGBoostPredictor) -> None:
    virginica_iris = IrisMeasurements(sepal_length=6.3, sepal_width=3.3, petal_length=6.0, petal_width=2.5)

    prediction = xgboost_predictor.predict(virginica_iris)

    assert isinstance(prediction, str)
    assert prediction in ["setosa", "versicolor", "virginica"]


# ----------------------
# Edge Case Tests
# ----------------------


@pytest.mark.unit
def test__xgboost_predictor__predict_minimum_values(xgboost_predictor: XGBoostPredictor) -> None:
    min_iris = IrisMeasurements(sepal_length=0.1, sepal_width=0.1, petal_length=0.1, petal_width=0.1)

    prediction = xgboost_predictor.predict(min_iris)

    assert isinstance(prediction, str)
    assert prediction in ["setosa", "versicolor", "virginica"]


@pytest.mark.unit
def test__xgboost_predictor__predict_maximum_values(xgboost_predictor: XGBoostPredictor) -> None:
    max_iris = IrisMeasurements(sepal_length=10.0, sepal_width=10.0, petal_length=10.0, petal_width=10.0)

    prediction = xgboost_predictor.predict(max_iris)

    assert isinstance(prediction, str)
    assert prediction in ["setosa", "versicolor", "virginica"]


# ----------------------
# Input Validation Tests
# ----------------------


@pytest.mark.unit
def test__xgboost_predictor__validates_input() -> None:
    predictor = XGBoostPredictor()

    with pytest.raises(Exception):  # Pydantic validation error
        predictor.predict({"invalid": "input"})


# ----------------------
# Integration Tests
# ----------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists("registry/prd/xgboost.joblib"), reason="Production XGBoost model not available")
def test__xgboost_predictor__production_model_predictions() -> None:
    predictor = XGBoostPredictor(model_path="registry/prd/xgboost.joblib")

    # Test known Iris samples
    setosa_sample = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    versicolor_sample = IrisMeasurements(sepal_length=5.9, sepal_width=3.0, petal_length=4.2, petal_width=1.5)
    virginica_sample = IrisMeasurements(sepal_length=6.3, sepal_width=3.3, petal_length=6.0, petal_width=2.5)

    # Make predictions
    setosa_pred = predictor.predict(setosa_sample)
    versicolor_pred = predictor.predict(versicolor_sample)
    virginica_pred = predictor.predict(virginica_sample)

    # Verify all predictions are valid
    assert setosa_pred in ["setosa", "versicolor", "virginica"]
    assert versicolor_pred in ["setosa", "versicolor", "virginica"]
    assert virginica_pred in ["setosa", "versicolor", "virginica"]


# ----------------------
# Feature Engineering Tests
# ----------------------


@pytest.mark.unit
def test__xgboost_predictor__uses_9_features(xgboost_predictor: XGBoostPredictor) -> None:
    measurements = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)

    # Access internal method to check feature engineering
    features = xgboost_predictor._prepare_features(measurements)

    # XGBoost uses 9 features (4 original + 5 engineered)
    assert features.shape == (1, 9)
