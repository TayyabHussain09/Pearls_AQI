"""
Configuration settings for AQI Prediction System.
Loads environment variables and provides typed configuration.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # ===================================================================
    # Karachi Location Configuration
    # ===================================================================
    KARACHI_LAT: float = 24.8607
    KARACHI_LON: float = 67.0011
    KARACHI_NAME: str = "Karachi"
    
    # ===================================================================
    # API Configuration - Open-Meteo (Free, no key required)
    # ===================================================================
    OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    
    # ===================================================================
    # API Configuration - OpenWeatherMap
    # ===================================================================
    OPENWEATHERMAP_URL: str = "https://api.openweathermap.org/data/3.0"
    OPENWEATHERMAP_API_KEY: Optional[str] = os.getenv("OPENWEATHERMAP_API_KEY")
    
    # ===================================================================
    # API Configuration - WeatherAPI
    # ===================================================================
    WEATHERAPI_URL: str = "http://api.weatherapi.com/v1"
    WEATHERAPI_KEY: Optional[str] = os.getenv("WEATHERAPI_KEY")
    
    # ===================================================================
    # API Configuration - AQICN
    # ===================================================================
    AQICN_API_URL: str = "https://api.waqi.info/feed"
    AQICN_API_TOKEN: Optional[str] = os.getenv("AQICN_API_TOKEN")
    
    # ===================================================================
    # Hopsworks Configuration
    # ===================================================================
    HOPSWORKS_API_KEY: Optional[str] = os.getenv("HOPSWORKS_API_KEY")
    HOPSWORKS_PROJECT_NAME: str = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")
    HOPSWORKS_HOST: str = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
    
    # ===================================================================
    # Feature Store Configuration
    # ===================================================================
    FEATURE_GROUP_NAME: str = "karachi_aqi_features"
    FEATURE_GROUP_VERSION: int = 1
    MODEL_REGISTRY_NAME: str = "aqi_prediction_models"
    
    # ===================================================================
    # Model Training Configuration
    # ===================================================================
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.1
    RANDOM_STATE: int = 42
    
    TARGET_COLUMN: str = "aqi"
    TIME_COLUMN: str = "datetime"
    
    # ===================================================================
    # Forecast Configuration
    # ===================================================================
    FORECAST_HOURS: int = 72  # 3-day forecast
    ROLLING_WINDOW_24H: int = 24
    ROLLING_WINDOW_7D: int = 168
    
    # ===================================================================
    # Dashboard Configuration
    # ===================================================================
    DASHBOARD_TITLE: str = "Karachi AQI Prediction System"
    DASHBOARD_ICON: str = "🌬️"
    
    # ===================================================================
    # Logging Configuration
    # ===================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ===================================================================
    # Data Paths
    # ===================================================================
    DATA_DIR: Path = Path(__file__).parent.parent / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = Path(__file__).parent.parent / "models"
    MODEL_DIR: Path = MODELS_DIR / "karachi"
    DOCS_DIR: Path = Path(__file__).parent.parent / "docs"
    
    @classmethod
    def get_api_keys(cls) -> dict:
        """Get all API keys as a dictionary."""
        return {
            "hopsworks": cls.HOPSWORKS_API_KEY,
            "aqicn": cls.AQICN_API_TOKEN,
            "openweathermap": cls.OPENWEATHERMAP_API_KEY,
            "weatherapi": cls.WEATHERAPI_KEY,
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required settings are present."""
        required = [
            ("HOPSWORKS_API_KEY", cls.HOPSWORKS_API_KEY, True),
        ]
        
        for name, value, required_flag in required:
            if required_flag and value is None:
                raise ValueError(f"Missing required environment variable: {name}")
        return True


# Global settings instance
settings = Settings()
