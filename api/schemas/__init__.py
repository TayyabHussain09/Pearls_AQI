"""API schemas module for AQI Prediction System."""

from api.schemas.data_models import (
    LocationBase,
    WeatherData,
    AQIData,
    CombinedData,
    HistoricalDataRequest,
    HistoricalDataResponse,
    ForecastRequest,
    ForecastResponse,
    ModelMetrics,
    HealthCheck
)

__all__ = [
    "LocationBase",
    "WeatherData",
    "AQIData",
    "CombinedData",
    "HistoricalDataRequest",
    "HistoricalDataResponse",
    "ForecastRequest",
    "ForecastResponse",
    "ModelMetrics",
    "HealthCheck"
]
