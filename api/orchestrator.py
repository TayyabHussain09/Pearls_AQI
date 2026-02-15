"""
API Orchestrator for AQI Prediction System.
Coordinates data fetching from multiple sources.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import numpy as np

from config.settings import settings
from api.fetchers.providers import (
    OpenMeteoFetcher,
    AQICNFetcher,
    OpenWeatherMapFetcher,
    WeatherAPIFetcher
)
from api.schemas.data_models import HistoricalDataResponse

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """Orchestrates data collection from multiple sources."""
    
    def __init__(self):
        self.open_meteo = OpenMeteoFetcher()
        self.aqi_fetcher = AQICNFetcher()
        self.owm_fetcher = OpenWeatherMapFetcher()
        self.weather_api = WeatherAPIFetcher()
    
    def fetch_current_combined(self) -> Dict[str, Any]:
        """Fetch current data from all sources."""
        try:
            weather = self.open_meteo.fetch_current()
            aqi = self.aqi_fetcher.fetch_current()
            
            try:
                owm_pollution = self.owm_fetcher.fetch_current()
            except Exception as e:
                logger.warning(f"OpenWeatherMap pollution failed: {e}")
                owm_pollution = {}
            
            try:
                weather_details = self.weather_api.fetch_current()
            except Exception as e:
                logger.warning(f"WeatherAPI failed: {e}")
                weather_details = {}
            
            return {
                "location": {
                    "name": settings.KARACHI_NAME,
                    "latitude": settings.KARACHI_LAT,
                    "longitude": settings.KARACHI_LON,
                    "country": "Pakistan"
                },
                "weather": {**weather, **weather_details},
                "aqi": {**aqi, **owm_pollution},
                "fetched_at": datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Combined fetch failed: {e}")
            raise
    
    def backfill_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Backfill historical data using Open-Meteo."""
        logger.info(f"Backfilling from {start_date.date()} to {end_date.date()}")
        
        chunk_size = timedelta(days=30)
        current_start = start_date
        all_records = []
        
        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            
            # Fetch weather data
            weather_records = self.open_meteo.fetch_historical(current_start, current_end)
            
            # Get current AQI for real-time data
            try:
                real_aqi = self.aqi_fetcher.fetch_current()
            except Exception as e:
                logger.warning(f"AQICN fetch failed: {e}")
                real_aqi = {"aqi": 120, "pm25": 45, "pm10": 70, "no2": 35, "o3": 50}
            
            # Generate simulated AQI based on weather conditions for each record
            for record in weather_records:
                simulated_aqi = self._simulate_aqi(record, real_aqi)
                # Flatten weather into record
                record["aqi"] = simulated_aqi
                record["aqi_value"] = simulated_aqi["aqi"]
            
            all_records.extend(weather_records)
            
            logger.info(f"Fetched {len(weather_records)} records ({current_start.date()} to {current_end.date()})")
            current_start = current_end
        
        logger.info(f"Total: {len(all_records)} historical records")
        return all_records
    
    def _simulate_aqi(self, record: Dict, real_aqi: Dict) -> Dict:
        """Simulate AQI values based on weather conditions."""
        weather = record
        
        base_aqi = real_aqi.get("aqi", 120)
        
        temp = weather.get("temperature_2m", 25)
        humidity = weather.get("relative_humidity_2m", 50)
        wind_speed = weather.get("wind_speed_10m", 10)
        pressure = weather.get("pressure_msl", 1013)
        clouds = weather.get("cloud_cover", 0)
        
        temp_effect = (temp - 25) * 0.5
        humidity_effect = (humidity - 50) * 0.3
        wind_effect = -wind_speed * 0.5
        pressure_effect = (pressure - 1013) * 0.1
        cloud_effect = -clouds * 0.1
        
        ts = int(weather.get("datetime", datetime.utcnow()).timestamp())
        np.random.seed(ts % (2**31))
        random_variation = np.random.normal(0, 5)
        
        simulated = base_aqi + temp_effect + humidity_effect + wind_effect + pressure_effect + cloud_effect + random_variation
        simulated = max(30, min(500, simulated))
        
        pm25 = real_aqi.get("pm25", 45)
        pm10 = real_aqi.get("pm10", 70)
        no2 = real_aqi.get("no2", 35)
        o3 = real_aqi.get("o3", 50)
        
        if base_aqi > 0:
            ratio = simulated / base_aqi
            pm25 = pm25 * ratio
            pm10 = pm10 * ratio
            no2 = no2 * ratio
            o3 = o3 * ratio
        
        return {
            "aqi": round(simulated, 1),
            "pm25": round(pm25, 1),
            "pm10": round(pm10, 1),
            "no2": round(no2, 1),
            "o3": round(o3, 1),
            "so2": round(no2 * 0.3, 1),
            "co": round(2.5 * (simulated / 100), 1),
            "main_pollutant": "pm25" if simulated > 100 else "o3",
            "source": "simulated"
        }


def get_orchestrator() -> DataOrchestrator:
    """Get a DataOrchestrator instance."""
    return DataOrchestrator()
