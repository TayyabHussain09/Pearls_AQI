"""
Script to upload feature data to Hopsworks Feature Store.
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry.hopsworks_pipeline import HopsworksFeatureStore
from config.settings import settings

def main():
    """Upload feature data to Hopsworks."""
    
    # Load the feature data
    data_file = Path("data/processed/karachi_features_20240216_20260215.csv")
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Run feature pipeline first: python features/main.py --mode backfill")
        return
    
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    fs = HopsworksFeatureStore()
    
    if not fs.connect():
        print("Failed to connect to Hopsworks!")
        return
    
    print("Connected to Hopsworks Feature Store")
    
    # Create feature group
    print(f"\nCreating feature group: {settings.FEATURE_GROUP_NAME}...")
    
    try:
        fg = fs.create_feature_group(
            df=df,
            name=settings.FEATURE_GROUP_NAME,
            version=1,
            description="Karachi AQI features with weather data and engineered features"
        )
        print(f"Successfully uploaded {len(df)} records to Hopsworks!")
        print(f"Feature group: {settings.FEATURE_GROUP_NAME} v1")
    except Exception as e:
        print(f"Error creating feature group: {e}")
    
    # Disconnect
    fs.disconnect()
    print("\nDone!")

if __name__ == "__main__":
    main()
