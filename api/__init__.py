"""API module for AQI Prediction System."""

from api.orchestrator import DataOrchestrator, get_orchestrator
from api.fetchers import (
    BaseFetcher,
    APIError,
    RateLimitError,
    OpenMeteoFetcher,
    AQICNFetcher,
    OpenWeatherMapFetcher,
    WeatherAPIFetcher
)
from api.schemas import (
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
    # Orchestrator
    "DataOrchestrator",
    "get_orchestrator",
    # Fetchers
    "BaseFetcher",
    "APIError",
    "RateLimitError",
    "OpenMeteoFetcher",
    "AQICNFetcher",
    "OpenWeatherMapFetcher",
    "WeatherAPIFetcher",
    # Schemas
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
