"""
Script to upload feature data to Hopsworks Feature Store.
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config first to avoid circular imports
from config.settings import settings

# Now import hopsworks - this will work because config is already loaded
import hopsworks


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
    project = hopsworks.login(
        project=settings.HOPSWORKS_PROJECT_NAME,
        api_key_value=settings.HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    
    print("Connected to Hopsworks Feature Store")
    
    # Create feature group
    print(f"\nCreating feature group: {settings.FEATURE_GROUP_NAME}...")
    
    try:
        # Try to get existing feature group first
        try:
            fg = fs.get_feature_group(name=settings.FEATURE_GROUP_NAME, version=1)
            print(f"Feature group already exists. Inserting data...")
            fg.insert(df, overwrite=False)
            print(f"Successfully inserted {len(df)} records to Hopsworks!")
        except:
            # Create new feature group
            fg = fs.create_feature_group(
                name=settings.FEATURE_GROUP_NAME,
                version=1,
                description="Karachi AQI features with weather data and engineered features"
            )
            fg.insert(df)
            print(f"Successfully created and uploaded {len(df)} records to Hopsworks!")
        
        print(f"Feature group: {settings.FEATURE_GROUP_NAME} v1")
    except Exception as e:
        print(f"Error with feature group: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
