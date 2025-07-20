from typing import Any, AsyncGenerator, Dict

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient, Response

from ai_flora_mind.server.schemas import IrisPredictionResponse

# ----------------------
# Fixture Setup
# ----------------------


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    from ai_flora_mind.server.main import get_app

    transport = ASGITransport(app=get_app())
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ----------------------
# Core API Tests
# ----------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test__predict_endpoint__returns_correct_format(async_client: AsyncClient) -> None:
    """Test that prediction endpoint returns the correct response format."""
    payload: Dict[str, Any] = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response: Response = await async_client.post("/predict", json=payload)

    assert response.status_code == 200
    response_data: Dict[str, Any] = response.json()

    # Validate response structure
    prediction_response: IrisPredictionResponse = IrisPredictionResponse.model_validate(response_data)
    assert prediction_response.prediction in ["setosa", "versicolor", "virginica"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload, expected_status",
    [
        # Missing required fields
        ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}, 422),
        ({"sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, 422),
        ({}, 422),
        # Invalid data types
        ({"sepal_length": "invalid", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, 422),
        ({"sepal_length": 5.1, "sepal_width": True, "petal_length": 1.4, "petal_width": 0.2}, 422),
        # Negative values
        ({"sepal_length": -1.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, 422),
    ],
)
@pytest.mark.asyncio
async def test__predict_endpoint__rejects_invalid_input(
    async_client: AsyncClient, payload: Dict[str, Any], expected_status: int
) -> None:
    """Test that prediction endpoint only accepts supported input values."""
    response: Response = await async_client.post("/predict", json=payload)
    assert response.status_code == expected_status


@pytest.mark.unit
@pytest.mark.parametrize(
    "measurements, expected_prediction",
    [
        # Setosa samples - Rule 1: petal_length < 2.0 (perfect separation)
        ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "setosa"),
        ({"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2}, "setosa"),
        ({"sepal_length": 5.4, "sepal_width": 3.9, "petal_length": 1.7, "petal_width": 0.4}, "setosa"),
        ({"sepal_length": 4.6, "sepal_width": 3.1, "petal_length": 1.5, "petal_width": 0.2}, "setosa"),
        # Versicolor samples - Rule 2: petal_length >= 2.0 and petal_width < 1.7
        ({"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}, "versicolor"),
        ({"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5}, "versicolor"),
        ({"sepal_length": 5.7, "sepal_width": 2.8, "petal_length": 4.1, "petal_width": 1.3}, "versicolor"),
        ({"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}, "versicolor"),
        # Virginica samples - Rule 3: petal_length >= 2.0 and petal_width >= 1.7
        ({"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}, "virginica"),
        ({"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9}, "virginica"),
        ({"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1}, "virginica"),
        ({"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2}, "virginica"),
        # Boundary cases
        (
            {"sepal_length": 5.0, "sepal_width": 3.0, "petal_length": 1.9, "petal_width": 0.5},
            "setosa",
        ),  # Just below setosa threshold
        (
            {"sepal_length": 6.0, "sepal_width": 3.0, "petal_length": 4.0, "petal_width": 1.6},
            "versicolor",
        ),  # Just below versicolor width threshold
        (
            {"sepal_length": 6.0, "sepal_width": 3.0, "petal_length": 4.0, "petal_width": 1.7},
            "virginica",
        ),  # At versicolor width threshold
    ],
)
@pytest.mark.asyncio
async def test__predict_endpoint__realistic_predictions(
    async_client: AsyncClient, measurements: Dict[str, Any], expected_prediction: str
) -> None:
    """Test that prediction endpoint returns correct species for realistic iris measurements."""
    response: Response = await async_client.post("/predict", json=measurements)

    assert response.status_code == 200
    response_data: Dict[str, Any] = response.json()

    # Validate response structure
    prediction_response: IrisPredictionResponse = IrisPredictionResponse.model_validate(response_data)

    # Check correct prediction
    assert prediction_response.prediction == expected_prediction, (
        f"Expected {expected_prediction} for petal_length={measurements['petal_length']}, "
        f"petal_width={measurements['petal_width']}, but got {prediction_response.prediction}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test__health_endpoint(async_client: AsyncClient) -> None:
    """Test health endpoint returns expected format."""
    response: Response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
