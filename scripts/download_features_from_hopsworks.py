#!/usr/bin/env python
"""Script to download features from Hopsworks Feature Store."""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import hopsworks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_features():
    """Download features from Hopsworks Feature Store."""
    
    # Connect to Hopsworks
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT", "tayyabhu"),
        api_key=os.getenv("HOPSWORKS_API_KEY")
    )
    
    # Get feature store
    fs = project.get_feature_store()
    
    # Get the feature group
    try:
        feature_group = fs.get_feature_group(
            name="karachi_aqi_features",
            version=1
        )
        
        # Read latest features (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        logger.info(f"Reading features from {start_date.date()} to {end_date.date()}")
        
        # Read features from feature group
        df = feature_group.read()
        
        # Filter to last 30 days
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'] >= start_date]
        
        # Save to local file
        output_path = "data/processed/karachi_features_latest.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Downloaded {len(df)} rows to {output_path}")
        
    except Exception as e:
        logger.error(f"Error downloading features: {e}")
        raise


if __name__ == "__main__":
    download_features()
