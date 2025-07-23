"""
Shared test fixtures for predictor tests.

Contains common fixtures used across multiple predictor test modules.
"""

from typing import List, Tuple

import pytest

from ai_flora_mind.configs import IrisMeasurements
from ai_flora_mind.predictors import DecisionTreePredictor, HeuristicPredictor, RandomForestPredictor, XGBoostPredictor


@pytest.fixture
def heuristic_predictor() -> HeuristicPredictor:
    return HeuristicPredictor()


@pytest.fixture
def random_forest_predictor() -> RandomForestPredictor:
    return RandomForestPredictor()


@pytest.fixture
def decision_tree_predictor() -> DecisionTreePredictor:
    return DecisionTreePredictor()


@pytest.fixture
def xgboost_predictor() -> XGBoostPredictor:
    return XGBoostPredictor()


@pytest.fixture
def temp_model_path(tmp_path):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # 14 features to match Random Forest feature engineering
    X_dummy = np.random.rand(10, 14)
    y_dummy = [0] * 3 + [1] * 3 + [2] * 4  # Numeric labels: 0=setosa, 1=versicolor, 2=virginica
    model.fit(X_dummy, y_dummy)

    temp_model_file = tmp_path / "test_model.joblib"
    joblib.dump(model, temp_model_file)

    return str(temp_model_file)


@pytest.fixture
def temp_decision_tree_model_path(tmp_path):
    import joblib
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    # 5 features to match Decision Tree feature engineering (4 original + petal_area)
    X_dummy = np.random.rand(10, 5)
    y_dummy = [0] * 3 + [1] * 3 + [2] * 4  # Numeric labels: 0=setosa, 1=versicolor, 2=virginica
    model.fit(X_dummy, y_dummy)

    temp_model_file = tmp_path / "test_decision_tree_model.joblib"
    joblib.dump(model, temp_model_file)

    return str(temp_model_file)


@pytest.fixture
def temp_xgboost_model_path(tmp_path):
    import joblib
    import numpy as np

    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        # 9 features to match XGBoost feature engineering
        X_dummy = np.random.rand(10, 9)
        y_dummy = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # Numeric labels for XGBoost
        model.fit(X_dummy, y_dummy)

        temp_model_file = tmp_path / "test_xgboost_model.joblib"
        joblib.dump(model, temp_model_file)

        return str(temp_model_file)
    except ImportError:
        # If xgboost not installed, create a simple mock that behaves like XGBoost
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        X_dummy = np.random.rand(10, 9)
        y_dummy = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        model.fit(X_dummy, y_dummy)

        temp_model_file = tmp_path / "test_xgboost_model.joblib"
        joblib.dump(model, temp_model_file)

        return str(temp_model_file)


@pytest.fixture
def iris_dataset() -> Tuple[List[IrisMeasurements], List[str]]:
    from sklearn.datasets import load_iris

    iris = load_iris()

    measurements = []
    for sample in iris.data:
        measurements.append(
            IrisMeasurements(
                sepal_length=sample[0], sepal_width=sample[1], petal_length=sample[2], petal_width=sample[3]
            )
        )

    species = [iris.target_names[target] for target in iris.target]

    return measurements, species
