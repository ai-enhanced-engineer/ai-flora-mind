"""HTTP API entry point for the AI Flora Mind iris prediction service."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..logging import configure_structlog, get_logger
from .schemas import IrisPredictionRequest, IrisPredictionResponse

configure_structlog()
logger = get_logger(__name__)


class FloraAPI:
    """Encapsulates the FastAPI server for iris prediction."""

    def __init__(self) -> None:
        logger.info("Initializing Flora API service")
        
        self.app: FastAPI = FastAPI(
            title="AI Flora Mind",
            description="Iris flower species prediction API",
            version="0.1.0",
            lifespan=self.lifespan
        )
        self.register_routes()

    @asynccontextmanager
    async def lifespan(self, _app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Application is starting up")
        yield
        logger.info("Application is shutting down")

    def register_routes(self) -> None:
        """Define the HTTP routes exposed by the service."""
        self.app.add_api_route(
            "/predict",
            self.predict_endpoint,
            methods=["POST"],
            response_model=IrisPredictionResponse,
            summary="Predict iris species",
            description="Predict the iris species based on flower measurements"
        )
        self.app.add_api_route(
            "/health", 
            self.health_check_endpoint, 
            methods=["GET"],
            summary="Health check",
            description="Check if the service is healthy"
        )

    async def predict_endpoint(self, request: IrisPredictionRequest) -> IrisPredictionResponse:
        """Predict iris species based on flower measurements."""
        logger.info(
            "Prediction request received",
            sepal_length=request.sepal_length,
            sepal_width=request.sepal_width, 
            petal_length=request.petal_length,
            petal_width=request.petal_width
        )
        
        # TODO: Replace with actual model prediction
        # For now, return hardcoded response as requested
        prediction = "setosa"
        
        logger.info("Prediction completed", prediction=prediction)
        
        return IrisPredictionResponse(prediction=prediction)

    async def health_check_endpoint(self) -> JSONResponse:
        """Health check endpoint."""
        logger.debug("Health check requested")
        return JSONResponse(content={"status": "healthy"})


def get_app() -> FastAPI:
    """Return a fully initialized FastAPI application."""
    flora_api = FloraAPI()
    return flora_api.app