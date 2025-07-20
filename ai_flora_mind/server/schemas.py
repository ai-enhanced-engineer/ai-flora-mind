from typing import Any

from pydantic import BaseModel, Field, field_validator


class IrisPredictionRequest(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm", ge=0.0, json_schema_extra={"example": 5.1})
    sepal_width: float = Field(description="Sepal width in cm", ge=0.0, json_schema_extra={"example": 3.5})
    petal_length: float = Field(description="Petal length in cm", ge=0.0, json_schema_extra={"example": 1.4})
    petal_width: float = Field(description="Petal width in cm", ge=0.0, json_schema_extra={"example": 0.2})

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width", mode="before")
    @classmethod
    def validate_numeric_fields(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("Boolean values are not allowed")
        if isinstance(v, str):
            raise ValueError("String values are not allowed")
        return float(v)


class IrisPredictionResponse(BaseModel):
    prediction: str = Field(
        description="Predicted iris species",
        pattern="^(setosa|versicolor|virginica)$",
        json_schema_extra={"example": "setosa"},
    )
