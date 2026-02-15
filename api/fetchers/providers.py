"""
API providers for fetching weather and AQI data.
Uses openmeteo_requests library for Open-Meteo.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class OpenMeteoFetcher:
    """
    Fetcher for Open-Meteo weather API.
    Uses standard requests for reliability.
    """
    
    def __init__(self):
        """Initialize Open-Meteo fetcher."""
        self.base_url = settings.OPEN_METEO_URL
        self.archive_url = settings.OPEN_METEO_ARCHIVE_URL
    
    def fetch_current(self) -> Dict[str, Any]:
        """Fetch current weather."""
        params = {
            "latitude": settings.KARACHI_LAT,
            "longitude": settings.KARACHI_LON,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,"
                      "wind_direction_10m,pressure_msl,cloud_cover",
            "timezone": "Asia/Karachi"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            status_code = response.status_code
            response_text = response.text
            
            if status_code != 200:
                logger.error(f"Open-Meteo HTTP {status_code}: {response_text}")
                raise Exception(f"HTTP {status_code}: {response_text}")
            
            return self._parse_json_response(response.json())
        except Exception as e:
            logger.error(f"Open-Meteo current failed: {e}")
            raise
    
    def fetch_historical(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch historical weather data using Archive API."""
        params = {
            "latitude": settings.KARACHI_LAT,
            "longitude": settings.KARACHI_LON,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                    "wind_speed_10m_max,wind_speed_10m_min",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                      "wind_speed_10m,wind_direction_10m,pressure_msl,cloud_cover",
            "timezone": "Asia/Karachi"
        }
        
        try:
            response = requests.get(self.archive_url, params=params, timeout=60)
            status_code = response.status_code
            response_text = response.text
            
            if status_code != 200:
                logger.error(f"Open-Meteo Archive HTTP {status_code}: {response_text}")
                raise Exception(f"HTTP {status_code}: {response_text}")
            
            return self._parse_hourly_json(response.json())
        except Exception as e:
            logger.error(f"Open-Meteo historical failed: {e}")
            raise
    
    def _parse_json_response(self, data: Dict) -> Dict:
        """Parse JSON response."""
        current = data.get("current", {})
        return {
            "datetime": datetime.fromisoformat(current.get("time", datetime.utcnow().isoformat())),
            "temperature_2m": current.get("temperature_2m"),
            "relative_humidity_2m": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed_10m": current.get("wind_speed_10m"),
            "wind_direction_10m": current.get("wind_direction_10m"),
            "pressure_msl": current.get("pressure_msl"),
            "cloud_cover": current.get("cloud_cover"),
            "source": "open-meteo"
        }
    
    def _parse_hourly_json(self, data: Dict) -> List[Dict]:
        """Parse hourly data from JSON response."""
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        records = []
        for i, t in enumerate(times):
            records.append({
                "datetime": datetime.fromisoformat(t),
                "temperature_2m": hourly.get("temperature_2m", [None]*len(times))[i],
                "relative_humidity_2m": hourly.get("relative_humidity_2m", [None]*len(times))[i],
                "precipitation": hourly.get("precipitation", [None]*len(times))[i],
                "wind_speed_10m": hourly.get("wind_speed_10m", [None]*len(times))[i],
                "wind_direction_10m": hourly.get("wind_direction_10m", [None]*len(times))[i],
                "pressure_msl": hourly.get("pressure_msl", [None]*len(times))[i],
                "cloud_cover": hourly.get("cloud_cover", [None]*len(times))[i],
                "source": "open-meteo"
            })
        return records


class AQICNFetcher:
    """
    Fetcher for AQICN (World Air Quality Index) API.
    Uses geo:{lat};{lon} format with trailing slash.
    """
    
    def __init__(self):
        self.base_url = settings.AQICN_API_URL
        self.token = settings.AQICN_API_TOKEN
    
    def fetch_current(self) -> Dict[str, Any]:
        """Fetch current AQI."""
        if not self.token:
            raise ValueError("AQICN_API_TOKEN not configured")
        
        # Build endpoint with geo:lat;lon format and trailing slash
        endpoint = f"geo:{settings.KARACHI_LAT};{settings.KARACHI_LON}/"
        url = f"{self.base_url}/{endpoint}"
        
        params = {"token": self.token}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            status_code = response.status_code
            response_text = response.text
            
            if status_code != 200:
                logger.error(f"AQICN HTTP {status_code}: {response_text}")
                raise Exception(f"HTTP {status_code}: {response_text}")
            
            data = response.json()
            
            if data.get("status") != "ok":
                logger.error(f"AQICN API error: {data}")
                raise Exception(f"API Error: {data}")
            
            return self._parse_response(data)
        except requests.exceptions.RequestException as e:
            logger.error(f"AQICN request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"AQICN error: {e}")
            raise
    
    def fetch_forecast(self) -> Dict[str, Any]:
        """Fetch AQI forecast."""
        if not self.token:
            raise ValueError("AQICN_API_TOKEN not configured")
        
        endpoint = f"geo:{settings.KARACHI_LAT};{settings.KARACHI_LON}/"
        url = f"{self.base_url}/{endpoint}"
        
        params = {"token": self.token, "forecast": "1"}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            status_code = response.status_code
            response_text = response.text
            
            if status_code != 200:
                logger.error(f"AQICN Forecast HTTP {status_code}: {response_text}")
                raise Exception(f"HTTP {status_code}: {response_text}")
            
            data = response.json()
            return self._parse_response(data)
        except Exception as e:
            logger.error(f"AQICN forecast failed: {e}")
            raise
    
    def _parse_response(self, data: Dict) -> Dict:
        """Parse AQICN response."""
        d = data.get("data", {})
        iaqi = d.get("iaqi", {})
        
        pollutants = {}
        for k, v in iaqi.items():
            if isinstance(v, dict):
                pollutants[k] = v.get("v")
            else:
                pollutants[k] = v
        
        return {
            "datetime": datetime.fromisoformat(d.get("time", {}).get("s", datetime.utcnow().isoformat())),
            "aqi": d.get("aqi"),
            "pm25": pollutants.get("pm25"),
            "pm10": pollutants.get("pm10"),
            "no2": pollutants.get("no2"),
            "o3": pollutants.get("o3"),
            "so2": pollutants.get("so2"),
            "co": pollutants.get("co"),
            "main_pollutant": d.get("dominentpol"),
            "pollutants": pollutants,
            "forecast": d.get("forecast", {}).get("daily", {}),
            "source": "aqicn"
        }


class OpenWeatherMapFetcher:
    """
    Fetcher for OpenWeatherMap Air Pollution API.
    Uses /data/2.5/air_pollution endpoint (NOT weather endpoint).
    """
    
    def __init__(self):
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = settings.OPENWEATHERMAP_API_KEY
    
    def fetch_current(self) -> Dict[str, Any]:
        """Fetch current air pollution data."""
        if not self.api_key:
            raise ValueError("OPENWEATHERMAP_API_KEY not configured")
        
        # Use AIR POLLUTION endpoint (not weather!)
        url = f"{self.base_url}/air_pollution"
        
        params = {
            "lat": settings.KARACHI_LAT,
            "lon": settings.KARACHI_LON,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            status_code = response.status_code
            response_text = response.text
            
            if status_code != 200:
                logger.error(f"OpenWeatherMap HTTP {status_code}: {response_text}")
                raise Exception(f"HTTP {status_code}: {response_text}")
            
            return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"OpenWeatherMap failed: {e}")
            raise
    
    def fetch_forecast(self) -> Dict[str, Any]:
        """Fetch air pollution forecast."""
        if not self.api_key:
            raise ValueError("OPENWEATHERMAP_API_KEY not configured")
        
        url = f"{self.base_url}/air_pollution/forecast"
        
        params = {
            "lat": settings.KARACHI_LAT,
            "lon": settings.KARACHI_LON,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            status_code = response.status_code
            response_text = response.text
            
            if status_code != 200:
                logger.error(f"OpenWeatherMap Forecast HTTP {status_code}: {response_text}")
                raise Exception(f"HTTP {status_code}: {response_text}")
            
            return self._parse_forecast_response(response.json())
        except Exception as e:
            logger.error(f"OpenWeatherMap forecast failed: {e}")
            raise
    
    def _parse_response(self, data: Dict) -> Dict:
        """Parse current air pollution response."""
        d = data.get("list", [{}])[0]
        main = d.get("main", {})
        components = d.get("components", {})
        
        return {
            "datetime": datetime.fromtimestamp(d.get("dt")),
            "aqi": main.get("aqi"),
            "pm25": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "co": components.get("co"),
            "source": "openweathermap"
        }
    
    def _parse_forecast_response(self, data: Dict) -> List[Dict]:
        """Parse forecast response."""
        forecasts = []
        for item in data.get("list", []):
            main = item.get("main", {})
            components = item.get("components", {})
            forecasts.append({
                "datetime": datetime.fromtimestamp(item.get("dt")),
                "aqi": main.get("aqi"),
                "pm25": components.get("pm2_5"),
                "pm10": components.get("pm10"),
                "no2": components.get("no2"),
                "o3": components.get("o3"),
                "source": "openweathermap"
            })
        return forecasts


class WeatherAPIFetcher:
    """Fetcher for WeatherAPI.com."""
    
    def __init__(self):
        self.base_url = settings.WEATHERAPI_URL
        self.api_key = settings.WEATHERAPI_KEY
    
    def fetch_current(self) -> Dict[str, Any]:
        """Fetch current weather."""
        if not self.api_key:
            raise ValueError("WEATHERAPI_KEY not configured")
        
        url = f"{self.base_url}/current.json"
        
        params = {
            "key": self.api_key,
            "q": f"{settings.KARACHI_LAT},{settings.KARACHI_LON}",
            "aqi": "yes"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            status_code = response.status_code
            response_text = response.text
            
            if status_code != 200:
                logger.error(f"WeatherAPI HTTP {status_code}: {response_text}")
                raise Exception(f"HTTP {status_code}: {response_text}")
            
            return self._parse_response(response.json())
        except Exception as e:
            logger.error(f"WeatherAPI failed: {e}")
            raise
    
    def _parse_response(self, data: Dict) -> Dict:
        """Parse WeatherAPI response."""
        c = data.get("current", {})
        return {
            "datetime": datetime.fromisoformat(c.get("last_updated")),
            "temperature_2m": c.get("temp_c"),
            "relative_humidity_2m": c.get("humidity"),
            "pressure_msl": c.get("pressure_mb"),
            "wind_speed_10m": c.get("wind_kph"),
            "wind_direction_10m": c.get("wind_degree"),
            "cloud_cover": c.get("cloud"),
            "visibility": (c.get("vis_km") or 10) * 1000,
            "air_quality": c.get("air_quality", {}),
            "source": "weatherapi"
        }


def test_all_connections():
    """Test all API connections."""
    print("\n" + "="*60)
    print("Testing API Connections for Karachi AQI Predictor")
    print("="*60 + "\n")
    
    # Test Open-Meteo
    print("Testing Open-Meteo...")
    try:
        fetcher = OpenMeteoFetcher()
        result = fetcher.fetch_current()
        print(f"  [OK] Open-Meteo: Success")
        print(f"     Temp: {result.get('temperature_2m')}C, Humidity: {result.get('relative_humidity_2m')}%")
    except Exception as e:
        print(f"  [FAIL] Open-Meteo: Failed - {e}")
    
    # Test AQICN
    print("\nTesting AQICN...")
    try:
        fetcher = AQICNFetcher()
        result = fetcher.fetch_current()
        print(f"  [OK] AQICN: Success")
        print(f"     AQI: {result.get('aqi')}, PM2.5: {result.get('pm25')}")
    except Exception as e:
        print(f"  [FAIL] AQICN: Failed - {e}")
    
    # Test OpenWeatherMap
    print("\nTesting OpenWeatherMap Air Pollution...")
    try:
        fetcher = OpenWeatherMapFetcher()
        result = fetcher.fetch_current()
        print(f"  [OK] OpenWeatherMap: Success")
        print(f"     AQI: {result.get('aqi')}, PM2.5: {result.get('pm25')}")
    except Exception as e:
        print(f"  [FAIL] OpenWeatherMap: Failed - {e}")
    
    # Test WeatherAPI
    print("\nTesting WeatherAPI...")
    try:
        fetcher = WeatherAPIFetcher()
        result = fetcher.fetch_current()
        print(f"  [OK] WeatherAPI: Success")
        print(f"     Temp: {result.get('temperature_2m')}C")
    except Exception as e:
        print(f"  [FAIL] WeatherAPI: Failed - {e}")
    
    print("\n" + "="*60)
    print("Connection Test Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_all_connections()
