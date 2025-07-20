"""
Unit tests for configuration models and ServiceConfig.

Tests the ServiceConfig class, environment variable handling, and model path resolution.
"""

import pytest

from ai_flora_mind.configs import ModelType, ServiceConfig


@pytest.mark.unit
def test__service_config__default_model_type_is_heuristic() -> None:
    config = ServiceConfig()
    assert config.model_type == ModelType.HEURISTIC


@pytest.mark.unit
def test__service_config__environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_MODEL_TYPE", "random_forest")
    config = ServiceConfig()
    assert config.model_type == ModelType.RANDOM_FOREST


@pytest.mark.unit
def test__service_config__case_insensitive_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables are case insensitive."""
    monkeypatch.setenv("flora_model_type", "random_forest")
    config = ServiceConfig()
    assert config.model_type == ModelType.RANDOM_FOREST


@pytest.mark.unit
def test__service_config__extra_environment_variables_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that extra environment variables are ignored."""
    monkeypatch.setenv("FLORA_MODEL_TYPE", "heuristic")
    monkeypatch.setenv("FLORA_UNKNOWN_VAR", "should_be_ignored")
    config = ServiceConfig()
    assert config.model_type == ModelType.HEURISTIC


@pytest.mark.unit
def test__service_config__get_model_path_heuristic_returns_none() -> None:
    config = ServiceConfig(model_type=ModelType.HEURISTIC)
    assert config.get_model_path() is None


@pytest.mark.unit
def test__service_config__get_model_path_random_forest_returns_consistent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_MODEL_TYPE", "random_forest")
    config = ServiceConfig()

    model_path = config.get_model_path()
    assert model_path == "registry/prd/random_forest.joblib"


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type,expected_error",
    [
        ("xgboost", "Model file not configured for model type: xgboost"),
    ],
)
def test__service_config__get_model_path_unimplemented_models_raise_error(
    monkeypatch: pytest.MonkeyPatch, model_type: str, expected_error: str
) -> None:
    monkeypatch.setenv("FLORA_MODEL_TYPE", model_type)
    config = ServiceConfig()

    with pytest.raises(ValueError, match=expected_error):
        config.get_model_path()


@pytest.mark.unit
def test__service_config__environment_prefix_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that FLORA_ prefix is handled correctly."""
    monkeypatch.setenv("FLORA_MODEL_TYPE", "random_forest")
    monkeypatch.setenv("OTHER_MODEL_TYPE", "should_be_ignored")

    config = ServiceConfig()
    assert config.model_type == ModelType.RANDOM_FOREST


@pytest.mark.unit
def test__service_config__get_model_path_decision_tree_returns_consistent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that decision tree model path is correctly configured."""
    monkeypatch.setenv("FLORA_MODEL_TYPE", "decision_tree")
    config = ServiceConfig()

    model_path = config.get_model_path()
    assert model_path == "registry/prd/decision_tree.joblib"


@pytest.mark.unit
def test__service_config__error_message_includes_model_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that error messages include the specific model type for debugging."""
    monkeypatch.setenv("FLORA_MODEL_TYPE", "xgboost")
    config = ServiceConfig()

    with pytest.raises(ValueError) as exc_info:
        config.get_model_path()

    error_message = str(exc_info.value)
    assert "xgboost" in error_message
    assert "Model file not configured" in error_message
