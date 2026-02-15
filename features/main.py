"""
Feature Engineering Pipeline Main Module.
Coordinates data fetching and feature generation.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import argparse

import pandas as pd

from config.settings import settings
from api.orchestrator import DataOrchestrator
from features.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Main pipeline for feature engineering."""
    
    def __init__(self):
        self.orchestrator = DataOrchestrator()
        self.engineer = FeatureEngineer()
        self.output_dir = Path(settings.DATA_DIR) / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Run pipeline on historical data."""
        logger.info(f"Fetching historical data: {start_date.date()} to {end_date.date()}")
        
        records = self.orchestrator.backfill_historical_data(start_date, end_date)
        
        if not records:
            logger.warning("No records fetched!")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(records)} records...")
        
        df = self._records_to_dataframe(records)
        
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"Sample data:\n{df.head(2)}")
        
        features_df = self.engineer.fit_transform(df)
        
        logger.info(f"Generated {len(features_df.columns)} features")
        
        output_file = output_file or f"karachi_features_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        output_path = self.output_dir / output_file
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
        
        return features_df
    
    def run_backfill(
        self,
        years: int = 2,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Backfill historical data."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=years*365)
        return self.run_historical(start_date, end_date, output_file)
    
    def run_incremental(
        self,
        hours: int = 24,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Run incremental update."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        return self.run_historical(start_date, end_date, output_file)
    
    def _records_to_dataframe(self, records: list) -> pd.DataFrame:
        """Convert records to DataFrame."""
        if not records:
            return pd.DataFrame()
        
        data = []
        for r in records:
            dt = r.get("datetime")
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt)
            
            # Flatten weather and aqi from record
            weather = r if "temperature_2m" in r else r.get("weather", {})
            aqi = r.get("aqi", {}) if isinstance(r.get("aqi", {}), dict) else r
            
            row = {
                "datetime": dt,
                "temperature_2m": weather.get("temperature_2m"),
                "relative_humidity_2m": weather.get("relative_humidity_2m"),
                "precipitation": weather.get("precipitation"),
                "wind_speed_10m": weather.get("wind_speed_10m"),
                "wind_direction_10m": weather.get("wind_direction_10m"),
                "pressure_msl": weather.get("pressure_msl"),
                "cloud_cover": weather.get("cloud_cover"),
                "aqi": aqi.get("aqi", aqi.get("aqi_value")),
                "pm25": aqi.get("pm25"),
                "pm10": aqi.get("pm10"),
                "no2": aqi.get("no2"),
                "o3": aqi.get("o3"),
                "source": weather.get("source", "unknown")
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--mode", choices=["backfill", "historical", "incremental"], default="backfill")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=2, help="Years of backfill")
    parser.add_argument("--hours", type=int, default=24, help="Hours for incremental")
    parser.add_argument("--output", help="Output filename")
    
    args = parser.parse_args()
    
    pipeline = FeaturePipeline()
    
    if args.mode == "backfill":
        df = pipeline.run_backfill(years=args.years, output_file=args.output)
    elif args.mode == "historical":
        start = datetime.fromisoformat(args.start) if args.start else datetime.utcnow() - timedelta(days=30)
        end = datetime.fromisoformat(args.end) if args.end else datetime.utcnow()
        df = pipeline.run_historical(start, end, output_file=args.output)
    else:
        df = pipeline.run_incremental(hours=args.hours, output_file=args.output)
    
    logger.info(f"Pipeline complete: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
