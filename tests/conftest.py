"""
Shared test fixtures for predictor tests.

Contains common fixtures used across multiple predictor test modules.
"""

from typing import List, Tuple

import pytest

from ai_flora_mind.configs import IrisMeasurements
from ai_flora_mind.predictors import HeuristicPredictor, RandomForestPredictor


@pytest.fixture
def heuristic_predictor() -> HeuristicPredictor:
    return HeuristicPredictor()


@pytest.fixture
def random_forest_predictor() -> RandomForestPredictor:
    return RandomForestPredictor()


@pytest.fixture
def temp_model_path(tmp_path):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # 14 features to match Random Forest feature engineering
    X_dummy = np.random.rand(10, 14)
    y_dummy = ["setosa"] * 3 + ["versicolor"] * 3 + ["virginica"] * 4
    model.fit(X_dummy, y_dummy)

    temp_model_file = tmp_path / "test_model.joblib"
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
