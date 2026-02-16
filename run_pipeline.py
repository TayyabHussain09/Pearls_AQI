#!/usr/bin/env python
"""
Main entry point for the AQI Prediction Pipeline.
Run this script to fetch data, engineer features, train models, and store to Hopsworks.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

from config.settings import settings
from features.main import FeaturePipeline
from models.trainer import ModelFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_feature_pipeline(mode: str, years: int = 2, hours: int = 24):
    """Run the feature engineering pipeline."""
    pipeline = FeaturePipeline()
    
    if mode == "backfill":
        logger.info(f"Starting backfill for {years} years...")
        df = pipeline.run_backfill(years=years)
    else:
        logger.info(f"Starting incremental update for {hours} hours...")
        df = pipeline.run_incremental(hours=hours)
    
    logger.info(f"Feature pipeline complete: {len(df)} rows")
    return df


def run_training_pipeline():
    """Run the model training pipeline."""
    logger.info("Starting model training pipeline...")
    
    factory = ModelFactory()
    
    # Find latest data file
    data_dir = Path(settings.DATA_DIR) / "processed"
    data_files = sorted(data_dir.glob("karachi_features*.csv"))
    
    if not data_files:
        logger.error("No feature data found. Run feature pipeline first!")
        return None
    
    latest_file = data_files[-1]
    logger.info(f"Loading data from {latest_file}")
    
    import pandas as pd
    df = pd.read_csv(latest_file)
    
    models, results = factory.train(df)
    
    logger.info("Training complete!")
    return models, results


def store_to_hopsworks_features(df):
    """Store features to Hopsworks Feature Store."""
    try:
        from models.registry.hopsworks_pipeline import HopsworksFeatureStore
        
        fs = HopsworksFeatureStore()
        if fs.connect():
            # Prepare dataframe for storage
            df_store = df.copy()
            if 'datetime' in df_store.columns:
                df_store['datetime'] = pd.to_datetime(df_store['datetime'])
            
            # Store to feature store
            fs.create_feature_group(
                df_store,
                version=2,
                description="Karachi AQI prediction features - weather and pollutant data"
            )
            fs.disconnect()
            logger.info("Successfully stored features to Hopsworks")
            return True
    except Exception as e:
        logger.error(f"Failed to store features to Hopsworks: {e}")
    return False


def store_to_hopsworks_model(model_path: Path, metrics: dict):
    """Store model to Hopsworks Model Registry."""
    try:
        from models.registry.hopsworks_pipeline import HopsworksModelRegistry
        
        mr = HopsworksModelRegistry()
        if mr.connect():
            mr.register_model(
                model_path,
                model_name="karachi_aqi_predictor",
                metrics=metrics,
                description="Karachi AQI prediction model trained on weather and pollutant features"
            )
            mr.disconnect()
            logger.info("Successfully stored model to Hopsworks")
            return True
    except Exception as e:
        logger.error(f"Failed to store model to Hopsworks: {e}")
    return False


def run_full_pipeline(years: int = 2, store_hopsworks: bool = False):
    """Run the complete pipeline."""
    logger.info("=" * 50)
    logger.info("Starting Full AQI Prediction Pipeline")
    logger.info("=" * 50)
    
    # Step 1: Feature Engineering
    logger.info("\n[Step 1/3] Feature Engineering...")
    df = run_feature_pipeline(mode="backfill", years=years)
    
    if df.empty:
        logger.error("Feature engineering failed!")
        return
    
    # Step 2: Store to Hopsworks (optional)
    if store_hopsworks:
        logger.info("\n[Step 2/3] Storing features to Hopsworks...")
        store_to_hopsworks_features(df)
    
    # Step 3: Model Training
    logger.info("\n[Step 3/3] Model Training...")
    result = run_training_pipeline()
    
    if result:
        models, results = result
        
        # Store model to Hopsworks
        if store_hopsworks:
            logger.info("\nStoring model to Hopsworks...")
            model_path = Path("models/karachi")
            best_model = results.get('best_model', 'XGBoost')
            best_rmse = results.get('best_rmse', 0)
            
            metrics = {
                'rmse': best_rmse,
                'mae': results.get('all_results', {}).get(best_model, {}).get('mae_mean', 0),
                'r2': results.get('all_results', {}).get(best_model, {}).get('r2_mean', 0)
            }
            
            store_to_hopsworks_model(model_path, metrics)
    
    logger.info("\n" + "=" * 50)
    logger.info("Pipeline Complete!")
    logger.info("=" * 50)
    
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AQI Prediction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "features", "train"],
        default="full",
        help="Pipeline mode: full (features + train), features only, or train only"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["backfill", "incremental"],
        default="backfill",
        help="Feature pipeline mode"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Years of historical data for backfill"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours for incremental update"
    )
    parser.add_argument(
        "--hopsworks",
        action="store_true",
        help="Store features and models to Hopsworks"
    )
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_full_pipeline(years=args.years, store_hopsworks=args.hopsworks)
    elif args.mode == "features":
        run_feature_pipeline(args.feature_mode, args.years, args.hours)
    else:
        # For train mode, optionally store to hopsworks
        result = run_training_pipeline()
        if args.hopsworks and result:
            models, results = result
            model_path = Path("models/karachi")
            best_model = results.get('best_model', 'Lasso')
            best_rmse = results.get('best_rmse', 0)
            metrics = {
                'rmse': best_rmse,
                'mae': results.get('all_results', {}).get(best_model, {}).get('mae_mean', 0),
                'r2': results.get('all_results', {}).get(best_model, {}).get('r2_mean', 0)
            }
            store_to_hopsworks_model(model_path, metrics)


if __name__ == "__main__":
    main()
