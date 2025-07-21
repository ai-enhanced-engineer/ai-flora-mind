"""
Comprehensive tests for ai_flora_mind.features module.

Tests all feature engineering functions, edge cases, and model-specific feature generation
to ensure robust and accurate feature creation for iris classification.
"""

from typing import List

import numpy as np
import pytest

from ai_flora_mind.configs import ModelType
from ai_flora_mind.features import (
    create_area_ratio_feature,
    create_is_likely_setosa_feature,
    create_petal_area_feature,
    create_petal_aspect_ratio_feature,
    create_petal_to_sepal_length_ratio_feature,
    create_petal_to_sepal_width_ratio_feature,
    create_sepal_area_feature,
    create_sepal_aspect_ratio_feature,
    create_size_index_feature,
    create_total_area_feature,
    create_versicolor_vs_virginica_interaction,
    engineer_features,
    get_feature_names,
)

# ----------------------
# Test Data Fixtures
# ----------------------


@pytest.fixture
def sample_iris_data() -> np.ndarray:
    return np.array(
        [
            [5.1, 3.5, 1.4, 0.2],  # Typical setosa
            [7.0, 3.2, 4.7, 1.4],  # Typical versicolor
            [6.3, 3.3, 6.0, 2.5],  # Typical virginica
        ]
    )


@pytest.fixture
def edge_case_data() -> np.ndarray:
    return np.array(
        [
            [0.1, 0.1, 0.1, 0.1],  # Very small values
            [10.0, 5.0, 8.0, 3.0],  # Very large values
            [5.0, 0.0, 2.0, 1.0],  # Zero sepal width (division test)
            [5.0, 2.0, 2.0, 0.0],  # Zero petal width (division test)
        ]
    )


@pytest.fixture
def boundary_setosa_data() -> np.ndarray:
    return np.array(
        [
            [5.0, 3.0, 1.9, 0.3],  # Just below setosa threshold
            [5.0, 3.0, 2.0, 0.3],  # At setosa threshold
            [5.0, 3.0, 2.1, 0.3],  # Just above setosa threshold
        ]
    )


@pytest.fixture
def feature_names() -> List[str]:
    return ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


# ----------------------
# Individual Feature Creation Tests
# ----------------------


@pytest.mark.unit
def test__create_petal_area_feature__basic_calculation(sample_iris_data: np.ndarray) -> None:
    result = create_petal_area_feature(sample_iris_data)

    expected = np.array(
        [
            1.4 * 0.2,  # setosa: 0.28
            4.7 * 1.4,  # versicolor: 6.58
            6.0 * 2.5,  # virginica: 15.0
        ]
    )

    np.testing.assert_array_almost_equal(result, expected, decimal=2)
    assert result.shape == (3,)


@pytest.mark.unit
def test__create_sepal_area_feature__basic_calculation(sample_iris_data: np.ndarray) -> None:
    result = create_sepal_area_feature(sample_iris_data)

    expected = np.array(
        [
            5.1 * 3.5,  # setosa: 17.85
            7.0 * 3.2,  # versicolor: 22.4
            6.3 * 3.3,  # virginica: 20.79
        ]
    )

    np.testing.assert_array_almost_equal(result, expected, decimal=2)
    assert result.shape == (3,)


@pytest.mark.unit
def test__create_petal_aspect_ratio_feature__normal_cases(sample_iris_data: np.ndarray) -> None:
    result = create_petal_aspect_ratio_feature(sample_iris_data)

    expected = np.array(
        [
            1.4 / 0.2,  # setosa: 7.0
            4.7 / 1.4,  # versicolor: 3.357
            6.0 / 2.5,  # virginica: 2.4
        ]
    )

    np.testing.assert_array_almost_equal(result, expected, decimal=2)


@pytest.mark.unit
def test__create_petal_aspect_ratio_feature__zero_width_handling(edge_case_data: np.ndarray) -> None:
    result = create_petal_aspect_ratio_feature(edge_case_data)

    # Check that zero petal width (last sample) results in 0, not division error
    assert result[3] == 0.0  # Zero petal width case
    assert not np.isnan(result[3])
    assert not np.isinf(result[3])


@pytest.mark.unit
def test__create_sepal_aspect_ratio_feature__zero_width_handling(edge_case_data: np.ndarray) -> None:
    result = create_sepal_aspect_ratio_feature(edge_case_data)

    # Check that zero sepal width (third sample) results in 0, not division error
    assert result[2] == 0.0  # Zero sepal width case
    assert not np.isnan(result[2])
    assert not np.isinf(result[2])


@pytest.mark.unit
def test__create_total_area_feature__combination(sample_iris_data: np.ndarray) -> None:
    petal_area = create_petal_area_feature(sample_iris_data)
    sepal_area = create_sepal_area_feature(sample_iris_data)
    result = create_total_area_feature(petal_area, sepal_area)

    expected = petal_area + sepal_area
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


@pytest.mark.unit
def test__create_area_ratio_feature__normal_cases(sample_iris_data: np.ndarray) -> None:
    petal_area = create_petal_area_feature(sample_iris_data)
    sepal_area = create_sepal_area_feature(sample_iris_data)
    result = create_area_ratio_feature(petal_area, sepal_area)

    expected = petal_area / sepal_area
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


@pytest.mark.unit
def test__create_area_ratio_feature__zero_sepal_area_handling() -> None:
    petal_area = np.array([1.0, 2.0])
    sepal_area = np.array([0.0, 4.0])  # First element is zero

    result = create_area_ratio_feature(petal_area, sepal_area)

    assert result[0] == 0.0  # Zero sepal area case
    assert result[1] == 0.5  # Normal case
    assert not np.isnan(result[0])
    assert not np.isinf(result[0])


@pytest.mark.unit
def test__create_is_likely_setosa_feature__boundary_detection(boundary_setosa_data: np.ndarray) -> None:
    result = create_is_likely_setosa_feature(boundary_setosa_data)

    expected = np.array([1.0, 0.0, 0.0])  # Below, at, above threshold
    np.testing.assert_array_equal(result, expected)

    # Ensure binary output
    assert all(val in [0.0, 1.0] for val in result)


@pytest.mark.unit
def test__create_petal_to_sepal_length_ratio_feature__calculation(sample_iris_data: np.ndarray) -> None:
    result = create_petal_to_sepal_length_ratio_feature(sample_iris_data)

    expected = np.array(
        [
            1.4 / 5.1,  # setosa
            4.7 / 7.0,  # versicolor
            6.0 / 6.3,  # virginica
        ]
    )

    np.testing.assert_array_almost_equal(result, expected, decimal=6)


@pytest.mark.unit
def test__create_petal_to_sepal_width_ratio_feature__calculation(sample_iris_data: np.ndarray) -> None:
    result = create_petal_to_sepal_width_ratio_feature(sample_iris_data)

    expected = np.array(
        [
            0.2 / 3.5,  # setosa
            1.4 / 3.2,  # versicolor
            2.5 / 3.3,  # virginica
        ]
    )

    np.testing.assert_array_almost_equal(result, expected, decimal=6)


@pytest.mark.unit
def test__create_size_index_feature__average_calculation(sample_iris_data: np.ndarray) -> None:
    result = create_size_index_feature(sample_iris_data)

    expected = np.array(
        [
            np.mean([5.1, 3.5, 1.4, 0.2]),  # setosa
            np.mean([7.0, 3.2, 4.7, 1.4]),  # versicolor
            np.mean([6.3, 3.3, 6.0, 2.5]),  # virginica
        ]
    )

    np.testing.assert_array_almost_equal(result, expected, decimal=6)


@pytest.mark.unit
def test__create_versicolor_vs_virginica_interaction__calculation(sample_iris_data: np.ndarray) -> None:
    result = create_versicolor_vs_virginica_interaction(sample_iris_data)

    # Manual calculation for verification
    petal_lengths = sample_iris_data[:, 2]
    petal_widths = sample_iris_data[:, 3]
    expected = petal_lengths * petal_widths * (petal_lengths / petal_widths)

    np.testing.assert_array_almost_equal(result, expected, decimal=6)
    assert result.shape == (3,)


# ----------------------
# Edge Case and Error Handling Tests
# ----------------------


@pytest.mark.unit
def test__feature_functions__handle_empty_arrays() -> None:
    empty_data = np.array([]).reshape(0, 4)

    # Test functions that should work with empty arrays
    assert create_petal_area_feature(empty_data).shape == (0,)
    assert create_sepal_area_feature(empty_data).shape == (0,)
    assert create_size_index_feature(empty_data).shape == (0,)
    assert create_is_likely_setosa_feature(empty_data).shape == (0,)


@pytest.mark.unit
def test__feature_functions__handle_single_sample() -> None:
    single_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

    petal_area = create_petal_area_feature(single_sample)
    assert petal_area.shape == (1,)
    assert petal_area[0] == 1.4 * 0.2

    is_setosa = create_is_likely_setosa_feature(single_sample)
    assert is_setosa.shape == (1,)
    assert is_setosa[0] == 1.0  # petal_length < 2.0


@pytest.mark.unit
def test__aspect_ratio_functions__extreme_values() -> None:
    extreme_data = np.array(
        [
            [1.0, 1000.0, 1.0, 1000.0],  # Very wide sepals/petals
            [1000.0, 1.0, 1000.0, 1.0],  # Very long sepals/petals
        ]
    )

    sepal_ratios = create_sepal_aspect_ratio_feature(extreme_data)
    petal_ratios = create_petal_aspect_ratio_feature(extreme_data)

    # Check calculations are correct
    assert sepal_ratios[0] == 1.0 / 1000.0  # Very small ratio
    assert sepal_ratios[1] == 1000.0 / 1.0  # Very large ratio
    assert petal_ratios[0] == 1.0 / 1000.0  # Very small ratio
    assert petal_ratios[1] == 1000.0 / 1.0  # Very large ratio

    # Ensure no overflow/underflow issues
    assert not np.isnan(sepal_ratios).any()
    assert not np.isnan(petal_ratios).any()
    assert not np.isinf(sepal_ratios).any()
    assert not np.isinf(petal_ratios).any()


# ----------------------
# Feature Engineering Integration Tests
# ----------------------


@pytest.mark.unit
def test__engineer_features__heuristic_model_type(sample_iris_data: np.ndarray, feature_names: List[str]) -> None:
    X_enhanced, names_enhanced = engineer_features(sample_iris_data, feature_names, ModelType.HEURISTIC)

    # Should return original data unchanged
    np.testing.assert_array_equal(X_enhanced, sample_iris_data)
    assert names_enhanced == feature_names
    assert X_enhanced.shape == (3, 4)


@pytest.mark.unit
def test__engineer_features__decision_tree_model_type(sample_iris_data: np.ndarray, feature_names: List[str]) -> None:
    X_enhanced, names_enhanced = engineer_features(sample_iris_data, feature_names, ModelType.DECISION_TREE)

    # Should add 1 feature (petal_area)
    assert X_enhanced.shape == (3, 5)
    assert len(names_enhanced) == 5
    assert names_enhanced == feature_names + ["petal_area"]

    # Verify original features are unchanged
    np.testing.assert_array_equal(X_enhanced[:, :4], sample_iris_data)

    # Verify petal_area calculation
    expected_petal_area = sample_iris_data[:, 2] * sample_iris_data[:, 3]
    np.testing.assert_array_almost_equal(X_enhanced[:, 4], expected_petal_area, decimal=6)


@pytest.mark.unit
def test__engineer_features__random_forest_model_type(sample_iris_data: np.ndarray, feature_names: List[str]) -> None:
    X_enhanced, names_enhanced = engineer_features(sample_iris_data, feature_names, ModelType.RANDOM_FOREST)

    # Should add 10 engineered features (4 original + 10 engineered = 14 total)
    assert X_enhanced.shape == (3, 14)
    assert len(names_enhanced) == 14

    expected_engineered_names = [
        "petal_area",
        "sepal_area",
        "petal_aspect_ratio",
        "sepal_aspect_ratio",
        "total_area",
        "area_ratio",
        "is_likely_setosa",
        "petal_to_sepal_length_ratio",
        "petal_to_sepal_width_ratio",
        "size_index",
    ]
    assert names_enhanced == feature_names + expected_engineered_names

    # Verify original features are unchanged
    np.testing.assert_array_equal(X_enhanced[:, :4], sample_iris_data)

    # Verify a few key engineered features
    petal_area = sample_iris_data[:, 2] * sample_iris_data[:, 3]
    np.testing.assert_array_almost_equal(X_enhanced[:, 4], petal_area, decimal=6)

    # Verify setosa flag (column 10: is_likely_setosa)
    expected_setosa = (sample_iris_data[:, 2] < 2.0).astype(float)
    np.testing.assert_array_equal(X_enhanced[:, 10], expected_setosa)


@pytest.mark.unit
def test__engineer_features__xgboost_model_type(sample_iris_data: np.ndarray, feature_names: List[str]) -> None:
    X_enhanced, names_enhanced = engineer_features(sample_iris_data, feature_names, ModelType.XGBOOST)

    # Should add 5 targeted engineered features (4 original + 5 engineered = 9 total)
    assert X_enhanced.shape == (3, 9)
    assert len(names_enhanced) == 9

    expected_engineered_names = [
        "petal_area",
        "area_ratio",
        "is_likely_setosa",
        "versicolor_virginica_interaction",
        "petal_to_sepal_width_ratio",
    ]
    assert names_enhanced == feature_names + expected_engineered_names

    # Verify original features are unchanged
    np.testing.assert_array_equal(X_enhanced[:, :4], sample_iris_data)

    # Verify petal_area feature
    expected_petal_area = sample_iris_data[:, 2] * sample_iris_data[:, 3]
    np.testing.assert_array_almost_equal(X_enhanced[:, 4], expected_petal_area, decimal=6)


@pytest.mark.unit
def test__engineer_features__unknown_model_type_raises_error(
    sample_iris_data: np.ndarray, feature_names: List[str]
) -> None:
    # Create a mock model type that doesn't exist
    with pytest.raises(ValueError, match="Unknown model type"):
        # We can't create a new ModelType enum value, so we'll monkey-patch temporarily
        class FakeModelType:
            value = "unknown_model"

        engineer_features(sample_iris_data, feature_names, FakeModelType())  # type: ignore


@pytest.mark.unit
def test__engineer_features__preserves_data_types_and_shapes() -> None:
    # Create test data with specific dtype
    test_data = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
    feature_names = get_feature_names()

    X_enhanced, names_enhanced = engineer_features(test_data, feature_names, ModelType.RANDOM_FOREST)

    # Check data types are preserved/promoted appropriately
    assert X_enhanced.dtype in [np.float32, np.float64]  # May be promoted to float64
    assert isinstance(names_enhanced, list)
    assert all(isinstance(name, str) for name in names_enhanced)


@pytest.mark.unit
def test__engineer_features__consistent_feature_order() -> None:
    test_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.0, 3.0, 4.0, 1.3]])
    feature_names = get_feature_names()

    # Run multiple times to ensure consistency
    results = []
    for _ in range(3):
        X_enhanced, names_enhanced = engineer_features(test_data, feature_names, ModelType.RANDOM_FOREST)
        results.append((X_enhanced.copy(), names_enhanced.copy()))

    # All results should be identical
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0][0], results[i][0])
        assert results[0][1] == results[i][1]


# ----------------------
# Integration and Utility Tests
# ----------------------


@pytest.mark.unit
def test__get_feature_names__returns_correct_names() -> None:
    names = get_feature_names()

    expected_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    assert names == expected_names
    assert len(names) == 4
    assert all(isinstance(name, str) for name in names)


@pytest.mark.unit
def test__feature_engineering__maintains_sample_count() -> None:
    # Test with different sample sizes
    for n_samples in [1, 10, 100]:
        test_data = np.random.rand(n_samples, 4) * 10  # Random data scaled appropriately
        feature_names = get_feature_names()

        for model_type in ModelType:
            X_enhanced, names_enhanced = engineer_features(test_data, feature_names, model_type)
            assert X_enhanced.shape[0] == n_samples, f"Sample count changed for {model_type.value}"


@pytest.mark.functional
def test__feature_engineering__real_iris_dataset_integration() -> None:
    from sklearn.datasets import load_iris

    # Load real iris dataset
    iris = load_iris()
    X = iris.data  # Shape: (150, 4)
    feature_names = get_feature_names()

    # Test each model type with real data
    for model_type in ModelType:
        X_enhanced, names_enhanced = engineer_features(X, feature_names, model_type)

        # Verify basic properties
        assert X_enhanced.shape[0] == 150  # Same number of samples
        assert len(names_enhanced) >= 4  # At least original features
        assert not np.isnan(X_enhanced).any(), f"NaN values found in {model_type.value} features"
        assert not np.isinf(X_enhanced).any(), f"Infinite values found in {model_type.value} features"

        # Verify original features are preserved
        np.testing.assert_array_equal(X_enhanced[:, :4], X)

        # Verify feature counts for each model type
        if model_type == ModelType.HEURISTIC:
            assert X_enhanced.shape[1] == 4
        elif model_type == ModelType.DECISION_TREE:
            assert X_enhanced.shape[1] == 5
        elif model_type == ModelType.RANDOM_FOREST:
            assert X_enhanced.shape[1] == 14
        elif model_type == ModelType.XGBOOST:
            assert X_enhanced.shape[1] == 9


@pytest.mark.unit
def test__feature_engineering__mathematical_properties() -> None:
    # Create test data with known properties
    test_data = np.array(
        [
            [4.0, 2.0, 1.0, 0.5],  # setosa-like (petal_length < 2.0)
            [6.0, 3.0, 4.0, 1.5],  # versicolor-like
            [7.0, 3.5, 6.0, 2.0],  # virginica-like
        ]
    )

    feature_names = get_feature_names()
    X_enhanced, _ = engineer_features(test_data, feature_names, ModelType.RANDOM_FOREST)

    # Test mathematical properties
    petal_areas = X_enhanced[:, 4]  # petal_area column
    sepal_areas = X_enhanced[:, 5]  # sepal_area column
    total_areas = X_enhanced[:, 8]  # total_area column

    # Total area should equal sum of petal and sepal areas
    np.testing.assert_array_almost_equal(total_areas, petal_areas + sepal_areas, decimal=6)

    # Areas should be positive
    assert (petal_areas >= 0).all()
    assert (sepal_areas >= 0).all()
    assert (total_areas >= 0).all()

    # Setosa flag should be 1 for first sample (petal_length < 2.0)
    setosa_flags = X_enhanced[:, 10]  # is_likely_setosa column
    assert setosa_flags[0] == 1.0
    assert setosa_flags[1] == 0.0
    assert setosa_flags[2] == 0.0
