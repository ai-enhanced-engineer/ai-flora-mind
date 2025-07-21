"""
Unit tests for the factory pattern and predictor creation.

Tests the get_predictor factory function including model instantiation,
error handling, logging behavior, and match/case pattern coverage.
"""

import os
import tempfile

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from ai_flora_mind.configs import ModelType, ServiceConfig
from ai_flora_mind.factory import get_predictor
from ai_flora_mind.predictors import (
    BasePredictor,
    DecisionTreePredictor,
    HeuristicPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)


@pytest.mark.unit
def test__get_predictor__heuristic_model_creates_heuristic_predictor() -> None:
    config = ServiceConfig()  # Defaults to heuristic

    predictor = get_predictor(config)

    assert isinstance(predictor, HeuristicPredictor)
    assert isinstance(predictor, BasePredictor)


@pytest.mark.unit
def test__get_predictor__xgboost_model_creates_xgboost_predictor() -> None:
    if not os.path.exists("registry/prd/xgboost.joblib"):
        pytest.skip("XGBoost model not available for testing")

    try:
        os.environ["FLORA_CLASSIFIER_TYPE"] = "xgboost"
        config = ServiceConfig()
        predictor = get_predictor(config)

        assert isinstance(predictor, XGBoostPredictor)
        assert isinstance(predictor, BasePredictor)
    finally:
        os.environ.pop("FLORA_CLASSIFIER_TYPE", None)


@pytest.mark.unit
def test__get_predictor__logging_behavior_for_heuristic_model(caplog: pytest.LogCaptureFixture) -> None:
    config = ServiceConfig()  # Defaults to heuristic

    with caplog.at_level("INFO"):
        predictor = get_predictor(config)

    assert isinstance(predictor, HeuristicPredictor)

    # Check log messages
    log_messages = [record.message for record in caplog.records]
    assert any("Creating predictor instance" in msg for msg in log_messages)
    assert any("Heuristic predictor created successfully" in msg for msg in log_messages)


@pytest.mark.unit
def test__get_predictor__decision_tree_model_creates_decision_tree_predictor() -> None:
    if not os.path.exists("registry/prd/decision_tree.joblib"):
        pytest.skip("Production decision tree model not available")

    os.environ["FLORA_CLASSIFIER_TYPE"] = "decision_tree"
    try:
        config = ServiceConfig()
        predictor = get_predictor(config)

        assert isinstance(predictor, DecisionTreePredictor)
        assert isinstance(predictor, BasePredictor)
    finally:
        os.environ.pop("FLORA_CLASSIFIER_TYPE", None)


@pytest.mark.unit
def test__get_predictor__all_models_work_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that all models create correct predictor types

    # Heuristic - always works
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "heuristic")
    config_heuristic = ServiceConfig()
    predictor_heuristic = get_predictor(config_heuristic)
    assert isinstance(predictor_heuristic, HeuristicPredictor)

    # Only test models if their files exist
    if os.path.exists("registry/prd/random_forest.joblib"):
        monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "random_forest")
        config_rf = ServiceConfig()
        predictor_rf = get_predictor(config_rf)
        assert isinstance(predictor_rf, RandomForestPredictor)

    if os.path.exists("registry/prd/decision_tree.joblib"):
        monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "decision_tree")
        config_dt = ServiceConfig()
        predictor_dt = get_predictor(config_dt)
        assert isinstance(predictor_dt, DecisionTreePredictor)

    if os.path.exists("registry/prd/xgboost.joblib"):
        monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "xgboost")
        config_xgb = ServiceConfig()
        predictor_xgb = get_predictor(config_xgb)
        assert isinstance(predictor_xgb, XGBoostPredictor)


@pytest.mark.unit
def test__get_predictor__random_forest_model_creates_random_forest_predictor() -> None:
    if not os.path.exists("registry/prd/random_forest.joblib"):
        pytest.skip("Production random forest model not available")

    os.environ["FLORA_CLASSIFIER_TYPE"] = "random_forest"
    try:
        config = ServiceConfig()
        predictor = get_predictor(config)

        assert isinstance(predictor, RandomForestPredictor)
        assert isinstance(predictor, BasePredictor)
    finally:
        os.environ.pop("FLORA_CLASSIFIER_TYPE", None)


@pytest.mark.unit
def test__get_predictor__random_forest_with_temporary_model() -> None:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        joblib.dump(model, temp_file.name)

        try:
            predictor = RandomForestPredictor(model_path=temp_file.name)
            assert isinstance(predictor, RandomForestPredictor)
            assert isinstance(predictor, BasePredictor)
        finally:
            os.unlink(temp_file.name)


@pytest.mark.unit
def test__get_predictor__random_forest_with_invalid_path_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "random_forest")

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
            monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "random_forest")

            # Mock get_model_path to simulate container path
            def mock_get_model_path(self) -> str:
                return temp_file.name  # Use temp file but simulate container path logic

            monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

            config = ServiceConfig()
            assert config.model_type == ModelType.RANDOM_FOREST

            predictor = get_predictor(config)
            assert isinstance(predictor, RandomForestPredictor)
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
                RandomForestPredictor(model_path=temp_file.name)
        finally:
            os.unlink(temp_file.name)


@pytest.mark.unit
def test__get_predictor__random_forest_model_attributes_logging(caplog: pytest.LogCaptureFixture) -> None:
    # Skip if production model not available
    if not os.path.exists("registry/prd/random_forest.joblib"):
        pytest.skip("Production random forest model not available")

    # Use environment variable to set model type
    os.environ["FLORA_CLASSIFIER_TYPE"] = "random_forest"
    try:
        config = ServiceConfig()

        with caplog.at_level("INFO"):
            predictor = get_predictor(config)

        assert isinstance(predictor, RandomForestPredictor)

        # Check that model attributes are logged
        log_records = [record for record in caplog.records if record.levelname == "INFO"]
        rf_success_logs = [
            record for record in log_records if "Random Forest predictor created successfully" in record.message
        ]

        assert len(rf_success_logs) > 0
    finally:
        # Clean up environment
        os.environ.pop("FLORA_CLASSIFIER_TYPE", None)


@pytest.mark.unit
def test__get_predictor__value_error_from_random_forest_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "random_forest")
    config = ServiceConfig()

    # Mock get_model_path to return None to trigger ValueError
    def mock_get_model_path(self) -> str:
        return None

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    with pytest.raises(ValueError, match="Random Forest model requires a file path"):
        get_predictor(config)


@pytest.mark.unit
def test__get_predictor__value_error_from_decision_tree_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "decision_tree")
    config = ServiceConfig()

    # Mock get_model_path to return None to trigger ValueError
    def mock_get_model_path(self) -> str:
        return None

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    with pytest.raises(ValueError, match="Decision Tree model requires a file path"):
        get_predictor(config)


@pytest.mark.unit
def test__get_predictor__value_error_from_xgboost_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "xgboost")
    config = ServiceConfig()

    # Mock get_model_path to return None to trigger ValueError
    def mock_get_model_path(self) -> str:
        return None

    monkeypatch.setattr(ServiceConfig, "get_model_path", mock_get_model_path)

    with pytest.raises(ValueError, match="XGBoost model requires a file path"):
        get_predictor(config)
