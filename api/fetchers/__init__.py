"""API fetchers module for AQI Prediction System."""

from api.fetchers.base import BaseFetcher, APIError, RateLimitError
from api.fetchers.providers import (
    OpenMeteoFetcher,
    OpenWeatherMapFetcher,
    WeatherAPIFetcher,
    AQICNFetcher
)

__all__ = [
    "BaseFetcher",
    "APIError",
    "RateLimitError",
    "OpenMeteoFetcher",
    "OpenWeatherMapFetcher",
    "WeatherAPIFetcher",
    "AQICNFetcher"
]
