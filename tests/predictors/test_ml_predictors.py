"""Consolidated tests for all ML predictors."""

import joblib
import numpy as np
import pytest

from ml_production_service.configs import IrisMeasurements, ModelType
from ml_production_service.predictors import BasePredictor, MLModelPredictor

# Parametrize tests to run for all ML model types
ML_MODEL_CONFIGS = [
    (ModelType.DECISION_TREE, "registry/prd/decision_tree.joblib", 5),  # 5 features for decision tree
    (ModelType.RANDOM_FOREST, "registry/prd/random_forest.joblib", 14),  # 14 features for random forest
    (ModelType.XGBOOST, "registry/prd/xgboost.joblib", 9),  # 9 features for xgboost
]


@pytest.fixture
def temp_ml_model_path(request, tmp_path):
    """Create a temporary model file with the right number of features for each model type."""
    model_type, _, n_features = request.param

    if model_type == ModelType.DECISION_TREE:
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_type == ModelType.RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
    elif model_type == ModelType.XGBOOST:
        try:
            import xgboost as xgb

            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        except ImportError:
            from sklearn.tree import DecisionTreeClassifier

            model = DecisionTreeClassifier(max_depth=3, random_state=42)

    X_dummy = np.random.rand(10, n_features)
    y_dummy = [0] * 3 + [1] * 3 + [2] * 4  # Numeric labels
    model.fit(X_dummy, y_dummy)

    temp_model_file = tmp_path / "test_model.joblib"
    joblib.dump(model, temp_model_file)

    return str(temp_model_file)


# ----------------------
# Initialization Tests
# ----------------------
@pytest.mark.unit
@pytest.mark.parametrize("model_type,default_path,n_features", ML_MODEL_CONFIGS)
def test__ml_predictor__initialization_default_model(model_type, default_path, n_features) -> None:
    predictor = MLModelPredictor(model_path=default_path, model_type=model_type)
    assert predictor.model_path == default_path
    assert predictor.model_type == model_type
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


@pytest.mark.unit
@pytest.mark.parametrize("temp_ml_model_path", ML_MODEL_CONFIGS, indirect=True)
def test__ml_predictor__initialization_custom_model(temp_ml_model_path, request) -> None:
    model_type, _, _ = request.node.callspec.params["temp_ml_model_path"]
    predictor = MLModelPredictor(model_path=temp_ml_model_path, model_type=model_type)
    assert predictor.model_path == temp_ml_model_path
    assert predictor.model_type == model_type
    assert predictor.model is not None
    assert hasattr(predictor.model, "predict")


# ----------------------
# Error Handling Tests
# ----------------------
@pytest.mark.unit
@pytest.mark.parametrize("model_type,_,__", ML_MODEL_CONFIGS)
def test__ml_predictor__missing_model_file_raises_error(model_type, _, __) -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        MLModelPredictor(model_path="nonexistent/path/model.joblib", model_type=model_type)


@pytest.mark.unit
@pytest.mark.parametrize("model_type,_,__", ML_MODEL_CONFIGS)
def test__ml_predictor__invalid_model_file_raises_error(model_type, _, __, tmp_path) -> None:
    invalid_file = tmp_path / "invalid.joblib"
    invalid_file.write_text("invalid model content")

    with pytest.raises(RuntimeError, match="Failed to load model"):
        MLModelPredictor(model_path=str(invalid_file), model_type=model_type)


# ----------------------
# Interface Tests
# ----------------------
@pytest.mark.unit
def test__ml_predictor__implements_base_predictor_interface() -> None:
    assert issubclass(MLModelPredictor, BasePredictor)
    assert hasattr(MLModelPredictor, "predict")


# ----------------------
# Prediction Tests
# ----------------------
@pytest.mark.unit
@pytest.mark.parametrize("temp_ml_model_path", ML_MODEL_CONFIGS, indirect=True)
def test__ml_predictor__prediction_returns_valid_species(temp_ml_model_path, request) -> None:
    """Test that predictors return valid species names (dummy models won't have accurate predictions)."""
    model_type, _, _ = request.node.callspec.params["temp_ml_model_path"]
    predictor = MLModelPredictor(model_path=temp_ml_model_path, model_type=model_type)
    measurements = IrisMeasurements(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)

    prediction = predictor.predict(measurements)

    assert prediction in ["setosa", "versicolor", "virginica"]


@pytest.mark.unit
@pytest.mark.parametrize("temp_ml_model_path", ML_MODEL_CONFIGS, indirect=True)
def test__ml_predictor__edge_case_small_values(temp_ml_model_path, request) -> None:
    model_type, _, _ = request.node.callspec.params["temp_ml_model_path"]
    predictor = MLModelPredictor(model_path=temp_ml_model_path, model_type=model_type)
    measurements = IrisMeasurements(sepal_length=0.1, sepal_width=0.1, petal_length=0.1, petal_width=0.1)

    prediction = predictor.predict(measurements)

    assert prediction in ["setosa", "versicolor", "virginica"]


@pytest.mark.unit
@pytest.mark.parametrize("temp_ml_model_path", ML_MODEL_CONFIGS, indirect=True)
def test__ml_predictor__edge_case_large_values(temp_ml_model_path, request) -> None:
    model_type, _, _ = request.node.callspec.params["temp_ml_model_path"]
    predictor = MLModelPredictor(model_path=temp_ml_model_path, model_type=model_type)
    measurements = IrisMeasurements(sepal_length=10.0, sepal_width=10.0, petal_length=10.0, petal_width=10.0)

    prediction = predictor.predict(measurements)

    assert prediction in ["setosa", "versicolor", "virginica"]
