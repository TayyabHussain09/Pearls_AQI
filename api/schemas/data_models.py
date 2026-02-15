"""
Data models for AQI Prediction System.
Defines Pydantic schemas for API responses and internal data structures.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LocationBase(BaseModel):
    """Base location information."""
    name: str
    latitude: float
    longitude: float
    country: str = "Pakistan"


class WeatherData(BaseModel):
    """Weather data from Open-Meteo API."""
    datetime: datetime
    temperature_2m: Optional[float] = None
    relative_humidity_2m: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    wind_direction_10m: Optional[float] = None
    pressure_msl: Optional[float] = None
    cloud_cover: Optional[float] = None
    visibility: Optional[float] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AQIData(BaseModel):
    """Air Quality Index data from AQICN API."""
    datetime: datetime
    aqi: int
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    o3: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    main_pollutant: Optional[str] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CombinedData(BaseModel):
    """Combined weather and AQI data."""
    location: LocationBase
    weather: WeatherData
    aqi: AQIData
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HistoricalDataRequest(BaseModel):
    """Request model for historical data fetching."""
    start_date: datetime
    end_date: datetime
    include_weather: bool = True
    include_aqi: bool = True
    timezone: str = "Asia/Karachi"


class HistoricalDataResponse(BaseModel):
    """Response model for historical data."""
    location: LocationBase
    records: List[Dict[str, Any]]
    total_records: int
    date_range: Dict[str, datetime]
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class ForecastRequest(BaseModel):
    """Request model for AQI forecasting."""
    hours_ahead: int = Field(default=24, ge=1, le=168)
    include_confidence: bool = True


class ForecastResponse(BaseModel):
    """Response model for AQI forecast."""
    location: LocationBase
    forecasts: List[Dict[str, Any]]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: Optional[str] = None


class ModelMetrics(BaseModel):
    """Model evaluation metrics."""
    model_name: str
    rmse: float
    mae: float
    r2_score: float
    mape: float
    training_time_seconds: float
    feature_count: int
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
