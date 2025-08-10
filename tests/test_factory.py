"""Factory pattern tests for predictor creation."""

import os
import tempfile
from contextlib import contextmanager
from typing import Generator

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ml_production_service.configs import ModelType, ServiceConfig
from ml_production_service.factory import get_predictor
from ml_production_service.predictors import (
    BasePredictor,
    HeuristicPredictor,
    MLModelPredictor,
)


@contextmanager
def temporary_model_file(model_type: ModelType) -> Generator[str, None, None]:
    """Create a temporary model file for testing."""
    # Create appropriate model based on type
    if model_type == ModelType.RANDOM_FOREST:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
    elif model_type == ModelType.DECISION_TREE:
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == ModelType.XGBOOST:
        # For XGBoost, we'll use RandomForest as a placeholder since it's just for factory testing
        model = RandomForestClassifier(n_estimators=5, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Fit with dummy data to make it a valid model
    import numpy as np

    X_dummy = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
        joblib.dump(model, temp_file.name)
        try:
            yield temp_file.name
        finally:
            os.unlink(temp_file.name)


@pytest.mark.unit
def test__get_predictor__heuristic_model_creates_heuristic_predictor() -> None:
    config = ServiceConfig()  # Defaults to heuristic

    predictor = get_predictor(config)

    assert isinstance(predictor, HeuristicPredictor)
    assert isinstance(predictor, BasePredictor)


@pytest.mark.unit
def test__get_predictor__xgboost_model_creates_xgboost_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    with temporary_model_file(ModelType.XGBOOST) as temp_model_path:
        monkeypatch.setenv("MPS_MODEL_TYPE", "xgboost")

        # Mock get_model_path to return our temporary file
        def mock_get_model_path(self) -> str:
            return temp_model_path

        monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

        config = ServiceConfig()
        predictor = get_predictor(config)

        assert isinstance(predictor, MLModelPredictor)
        assert isinstance(predictor, BasePredictor)


@pytest.mark.unit
def test__get_predictor__logging_behavior_for_heuristic_model(caplog: pytest.LogCaptureFixture) -> None:
    config = ServiceConfig()  # Defaults to heuristic

    with caplog.at_level("INFO"):
        predictor = get_predictor(config)

    assert isinstance(predictor, HeuristicPredictor)

    # Factory doesn't log anything for heuristic predictor (no model loading)
    log_messages = [record.message for record in caplog.records]
    assert len(log_messages) == 0  # No logging expected for heuristic


@pytest.mark.unit
def test__get_predictor__decision_tree_model_creates_decision_tree_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    with temporary_model_file(ModelType.DECISION_TREE) as temp_model_path:
        monkeypatch.setenv("MPS_MODEL_TYPE", "decision_tree")

        # Mock get_model_path to return our temporary file
        def mock_get_model_path(self) -> str:
            return temp_model_path

        monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

        config = ServiceConfig()
        predictor = get_predictor(config)

        assert isinstance(predictor, MLModelPredictor)
        assert isinstance(predictor, BasePredictor)


@pytest.mark.unit
def test__get_predictor__all_models_work_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that all models create correct predictor types

    # Heuristic - always works
    monkeypatch.setenv("MPS_MODEL_TYPE", "heuristic")
    config_heuristic = ServiceConfig()
    predictor_heuristic = get_predictor(config_heuristic)
    assert isinstance(predictor_heuristic, HeuristicPredictor)

    # Test all ML model types with temporary files
    ml_model_types = [ModelType.RANDOM_FOREST, ModelType.DECISION_TREE, ModelType.XGBOOST]

    for model_type in ml_model_types:
        with temporary_model_file(model_type) as temp_model_path:
            monkeypatch.setenv("MPS_MODEL_TYPE", model_type.value)

            # Mock get_model_path to return our temporary file
            def mock_get_model_path(self) -> str:
                return temp_model_path

            monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

            config = ServiceConfig()
            predictor = get_predictor(config)
            assert isinstance(predictor, MLModelPredictor)


@pytest.mark.unit
def test__get_predictor__random_forest_model_creates_random_forest_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    with temporary_model_file(ModelType.RANDOM_FOREST) as temp_model_path:
        monkeypatch.setenv("MPS_MODEL_TYPE", "random_forest")

        # Mock get_model_path to return our temporary file
        def mock_get_model_path(self) -> str:
            return temp_model_path

        monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

        config = ServiceConfig()
        predictor = get_predictor(config)

        assert isinstance(predictor, MLModelPredictor)
        assert isinstance(predictor, BasePredictor)


@pytest.mark.unit
def test__get_predictor__random_forest_with_temporary_model() -> None:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        joblib.dump(model, temp_file.name)

        try:
            predictor = MLModelPredictor(model_path=temp_file.name, model_type=ModelType.RANDOM_FOREST)
            assert isinstance(predictor, MLModelPredictor)
            assert isinstance(predictor, BasePredictor)
        finally:
            os.unlink(temp_file.name)


@pytest.mark.unit
def test__get_predictor__random_forest_with_invalid_path_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPS_MODEL_TYPE", "random_forest")

    def mock_get_model_path(self) -> str:
        return "/nonexistent/path/model.joblib"

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    config = ServiceConfig()

    with pytest.raises(FileNotFoundError):
        get_predictor(config)


@pytest.mark.unit
def test__get_predictor__environment_integration_with_is_container(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a temporary model file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        joblib.dump(model, temp_file.name)

        try:
            # Test in container environment
            monkeypatch.setenv("IS_CONTAINER", "true")
            monkeypatch.setenv("MPS_MODEL_TYPE", "random_forest")

            # Mock get_model_path to simulate container path
            def mock_get_model_path(self) -> str:
                return temp_file.name  # Use temp file but simulate container path logic

            monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

            config = ServiceConfig()
            assert config.model_type == ModelType.RANDOM_FOREST

            predictor = get_predictor(config)
            assert isinstance(predictor, MLModelPredictor)
        finally:
            os.unlink(temp_file.name)


@pytest.mark.unit
def test__get_predictor__error_propagation_from_predictor_initialization() -> None:
    # Create an invalid model file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"not a valid joblib file")
        temp_file.flush()

        try:
            # Should raise an exception from joblib loading
            with pytest.raises(Exception):  # Could be various exception types from joblib
                MLModelPredictor(model_path=temp_file.name, model_type=ModelType.RANDOM_FOREST)
        finally:
            os.unlink(temp_file.name)


@pytest.mark.unit
def test__get_predictor__random_forest_model_attributes_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that the factory successfully creates a random forest predictor
    # Logging functionality is tested in other integration tests
    with temporary_model_file(ModelType.RANDOM_FOREST) as temp_model_path:
        monkeypatch.setenv("MPS_MODEL_TYPE", "random_forest")

        # Mock get_model_path to return our temporary file
        def mock_get_model_path(self) -> str:
            return temp_model_path

        monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

        config = ServiceConfig()
        predictor = get_predictor(config)

        assert isinstance(predictor, MLModelPredictor)
        assert predictor.model_type == ModelType.RANDOM_FOREST


@pytest.mark.unit
def test__get_predictor__value_error_from_random_forest_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPS_MODEL_TYPE", "random_forest")
    config = ServiceConfig()

    # Mock get_model_path to return None to trigger ValueError
    def mock_get_model_path(self) -> str:
        return None

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    with pytest.raises(ValueError, match="Model type 'random_forest' requires a model file path"):
        get_predictor(config)


@pytest.mark.unit
def test__get_predictor__value_error_from_decision_tree_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPS_MODEL_TYPE", "decision_tree")
    config = ServiceConfig()

    # Mock get_model_path to return None to trigger ValueError
    def mock_get_model_path(self) -> str:
        return None

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    with pytest.raises(ValueError, match="Model type 'decision_tree' requires a model file path"):
        get_predictor(config)


@pytest.mark.unit
def test__get_predictor__value_error_from_xgboost_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPS_MODEL_TYPE", "xgboost")
    config = ServiceConfig()

    # Mock get_model_path to return None to trigger ValueError
    def mock_get_model_path(self) -> str:
        return None

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    with pytest.raises(ValueError, match="Model type 'xgboost' requires a model file path"):
        get_predictor(config)


@pytest.mark.unit
def test__get_predictor__unknown_model_type_validation() -> None:
    # Create a mock config with an invalid model type
    # This test verifies the new model type validation logic
    config = ServiceConfig()
    config.model_type = "invalid_model_type"  # type: ignore

    with pytest.raises(ValueError, match="Unknown model type 'invalid_model_type'"):
        get_predictor(config)


@pytest.mark.unit
def test__get_predictor__improved_error_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPS_MODEL_TYPE", "random_forest")
    config = ServiceConfig()

    # Mock get_model_path to return None
    def mock_get_model_path(self) -> str:
        return None

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    with pytest.raises(ValueError) as exc_info:
        get_predictor(config)

    error_message = str(exc_info.value)
    assert "Model type 'random_forest' requires a model file path" in error_message
    assert "Ensure the model file exists in the registry" in error_message
