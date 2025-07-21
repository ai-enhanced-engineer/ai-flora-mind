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
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "random_forest")
    config = ServiceConfig()
    assert config.model_type == ModelType.RANDOM_FOREST


@pytest.mark.unit
def test__service_config__case_insensitive_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("flora_classifier_type", "random_forest")
    config = ServiceConfig()
    assert config.model_type == ModelType.RANDOM_FOREST


@pytest.mark.unit
def test__service_config__extra_environment_variables_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "heuristic")
    monkeypatch.setenv("FLORA_UNKNOWN_VAR", "should_be_ignored")
    config = ServiceConfig()
    assert config.model_type == ModelType.HEURISTIC


@pytest.mark.unit
def test__service_config__get_model_path_heuristic_returns_none() -> None:
    config = ServiceConfig(model_type=ModelType.HEURISTIC)
    assert config.get_model_path() is None


@pytest.mark.unit
def test__service_config__get_model_path_random_forest_returns_consistent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "random_forest")
    config = ServiceConfig()

    model_path = config.get_model_path()
    assert model_path == "registry/prd/random_forest.joblib"


@pytest.mark.unit
def test__service_config__environment_prefix_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "random_forest")
    monkeypatch.setenv("OTHER_CLASSIFIER_TYPE", "should_be_ignored")

    config = ServiceConfig()
    assert config.model_type == ModelType.RANDOM_FOREST


@pytest.mark.unit
def test__service_config__get_model_path_decision_tree_returns_consistent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "decision_tree")
    config = ServiceConfig()

    model_path = config.get_model_path()
    assert model_path == "registry/prd/decision_tree.joblib"


@pytest.mark.unit
def test__service_config__xgboost_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLORA_CLASSIFIER_TYPE", "xgboost")
    config = ServiceConfig()

    model_path = config.get_model_path()
    assert model_path == "registry/prd/xgboost.joblib"
