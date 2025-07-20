"""HTTP API entry point for the AI Flora Mind iris prediction service."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..configs import IrisMeasurements, ServiceConfig
from ..factory import get_predictor
from ..logging import configure_structlog, get_logger
from .schemas import IrisPredictionRequest, IrisPredictionResponse

configure_structlog()
logger = get_logger(__name__)


class FloraAPI:
    def __init__(self) -> None:
        logger.info("Initializing Flora API service")

        # Load configuration and initialize the predictor
        config = ServiceConfig()
        self.predictor = get_predictor(config)
        logger.info(
            "Predictor initialized from configuration",
            model_type=config.model_type.value,
            model_path=config.get_model_path() or "N/A",
        )

        self.app: FastAPI = FastAPI(
            title="AI Flora Mind",
            description="Iris flower species prediction API",
            version="0.1.0",
            lifespan=self.lifespan,
        )
        self.register_routes()

    @asynccontextmanager
    async def lifespan(self, _app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Application is starting up")
        yield
        logger.info("Application is shutting down")

    def register_routes(self) -> None:
        self.app.add_api_route(
            "/predict",
            self.predict_endpoint,
            methods=["POST"],
            response_model=IrisPredictionResponse,
            summary="Predict iris species",
            description="Predict the iris species based on flower measurements",
        )
        self.app.add_api_route(
            "/health",
            self.health_check_endpoint,
            methods=["GET"],
            summary="Health check",
            description="Check if the service is healthy",
        )

    async def predict_endpoint(self, request: IrisPredictionRequest) -> IrisPredictionResponse:
        logger.info(
            "Prediction request received",
            sepal_length=request.sepal_length,
            sepal_width=request.sepal_width,
            petal_length=request.petal_length,
            petal_width=request.petal_width,
        )

        # Convert request to IrisMeasurements
        measurements = IrisMeasurements(
            sepal_length=request.sepal_length,
            sepal_width=request.sepal_width,
            petal_length=request.petal_length,
            petal_width=request.petal_width,
        )

        # Use the predictor with new interface
        prediction = self.predictor.predict(measurements)

        logger.info("Prediction completed", prediction=prediction)

        return IrisPredictionResponse(prediction=prediction)

    async def health_check_endpoint(self) -> JSONResponse:
        logger.debug("Health check requested")
        return JSONResponse(content={"status": "healthy"})


def get_app() -> FastAPI:
    flora_api = FloraAPI()
    return flora_api.app
