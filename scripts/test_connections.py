#!/usr/bin/env python
"""
Verification script to test all API connections.
Run this script to verify your API keys and endpoints are working correctly.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from api.fetchers.providers import (
    OpenMeteoFetcher,
    AQICNFetcher,
    OpenWeatherMapFetcher,
    WeatherAPIFetcher
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def print_success(message: str):
    """Print success message."""
    print(f"  [OK] {message}")


def print_error(message: str):
    """Print error message."""
    print(f"  [FAIL] {message}")


def print_info(message: str):
    """Print info message."""
    print(f"  [INFO] {message}")


def test_open_meteo():
    """Test Open-Meteo API connection."""
    print_header("Testing Open-Meteo API")
    
    try:
        fetcher = OpenMeteoFetcher()
        result = fetcher.fetch_current()
        
        print_success(f"Connection successful!")
        print(f"  Temperature: {result.get('temperature_2m')}C")
        print(f"  Humidity: {result.get('relative_humidity_2m')}%")
        print(f"  Wind Speed: {result.get('wind_speed_10m')} km/h")
        print(f"  Pressure: {result.get('pressure_msl')} hPa")
        print(f"  Cloud Cover: {result.get('cloud_cover')}%")
        print(f"  Source: {result.get('source')}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def test_aqicn():
    """Test AQICN API connection."""
    print_header("Testing AQICN API")
    
    try:
        fetcher = AQICNFetcher()
        result = fetcher.fetch_current()
        
        print_success(f"Connection successful!")
        print(f"  AQI: {result.get('aqi')}")
        print(f"  PM2.5: {result.get('pm25')} ug/m3")
        print(f"  PM10: {result.get('pm10')} ug/m3")
        print(f"  NO2: {result.get('no2')} ug/m3")
        print(f"  O3: {result.get('o3')} ug/m3")
        print(f"  CO: {result.get('co')} mg/m3")
        print(f"  Main Pollutant: {result.get('main_pollutant')}")
        print(f"  Source: {result.get('source')}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def test_openweathermap():
    """Test OpenWeatherMap Air Pollution API connection."""
    print_header("Testing OpenWeatherMap Air Pollution API")
    
    try:
        fetcher = OpenWeatherMapFetcher()
        result = fetcher.fetch_current()
        
        print_success(f"Connection successful!")
        print(f"  AQI: {result.get('aqi')}")
        print(f"  PM2.5: {result.get('pm25')} ug/m3")
        print(f"  PM10: {result.get('pm10')} ug/m3")
        print(f"  NO2: {result.get('no2')} ug/m3")
        print(f"  O3: {result.get('o3')} ug/m3")
        print(f"  SO2: {result.get('so2')} ug/m3")
        print(f"  CO: {result.get('co')} ug/m3")
        print(f"  Source: {result.get('source')}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def test_weatherapi():
    """Test WeatherAPI connection."""
    print_header("Testing WeatherAPI")
    
    try:
        fetcher = WeatherAPIFetcher()
        result = fetcher.fetch_current()
        
        print_success(f"Connection successful!")
        print(f"  Temperature: {result.get('temperature_2m')}C")
        print(f"  Humidity: {result.get('relative_humidity_2m')}%")
        print(f"  Pressure: {result.get('pressure_msl')} hPa")
        print(f"  Wind Speed: {result.get('wind_speed_10m')} km/h")
        print(f"  Wind Direction: {result.get('wind_direction_10m')} deg")
        print(f"  Cloud Cover: {result.get('cloud_cover')}%")
        print(f"  Visibility: {result.get('visibility')} m")
        print(f"  Source: {result.get('source')}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def print_configuration():
    """Print current configuration."""
    print_header("Current Configuration")
    
    print(f"  Location: {settings.KARACHI_NAME}")
    print(f"  Latitude: {settings.KARACHI_LAT}")
    print(f"  Longitude: {settings.KARACHI_LON}")
    print()
    
    print("  API Keys:")
    print(f"    Open-Meteo: [OK] Configured (free, no key)")
    print(f"    AQICN: {'[OK] Configured' if settings.AQICN_API_TOKEN else '[FAIL] Not configured'}")
    print(f"    OpenWeatherMap: {'[OK] Configured' if settings.OPENWEATHERMAP_API_KEY else '[FAIL] Not configured'}")
    print(f"    WeatherAPI: {'[OK] Configured' if settings.WEATHERAPI_KEY else '[FAIL] Not configured'}")
    print(f"    Hopsworks: {'[OK] Configured' if settings.HOPSWORKS_API_KEY else '[FAIL] Not configured'}")


def main():
    """Main test function."""
    print("\n" + "="*58)
    print("  API Connection Tester")
    print("  Karachi AQI Prediction System")
    print("="*58)
    
    print(f"\n  Test Time: {datetime.utcnow().isoformat()} UTC")
    print(f"  Location: Karachi ({settings.KARACHI_LAT}, {settings.KARACHI_LON})")
    
    print_configuration()
    
    results = {}
    
    results["Open-Meteo"] = test_open_meteo()
    results["AQICN"] = test_aqicn()
    results["OpenWeatherMap"] = test_openweathermap()
    results["WeatherAPI"] = test_weatherapi()
    
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {name}: {status}")
    
    print()
    print(f"  Results: {passed}/{total} APIs working")
    
    if passed == total:
        print_success("All APIs are configured correctly!")
        return 0
    else:
        print_error("Some APIs failed. Check your configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
