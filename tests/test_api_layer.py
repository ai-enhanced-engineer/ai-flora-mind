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
@pytest.mark.asyncio
async def test__health_endpoint(async_client: AsyncClient) -> None:
    """Test health endpoint returns expected format."""
    response: Response = await async_client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}