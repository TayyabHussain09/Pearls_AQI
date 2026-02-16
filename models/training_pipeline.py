"""
Training Pipeline for AQI Prediction System.
Runs the complete model training workflow.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import logging
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from features.feature_engineering import FeatureEngineer
from models.trainer import ModelFactory
from models.registry import LocalModelRegistry
from models.registry.hopsworks_registry import HopsworksIntegration

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Main training pipeline class.
    Orchestrates feature loading, model training, and registration.
    """
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.feature_engineer = FeatureEngineer()
        self.model_factory = ModelFactory()
        self.registry = LocalModelRegistry()
    
    def run(self, data_path: Path = None) -> dict:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to feature data CSV
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting training pipeline")
        
        try:
            # Step 1: Load data
            logger.info("Loading feature data...")
            if data_path is None:
                data_path = settings.PROCESSED_DATA_DIR / "karachi_features_20240216_20260215.csv"
            
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} records")
            
            # Step 2: Train and evaluate models
            logger.info("Training models...")
            trained_models, results = self.model_factory.train(df)
            
            # Step 3: Get best model info
            best_model_name = results.get('best_model', 'Ridge')
            if best_model_name == 'RandomForest':
                # Check if Ridge is actually better
                if 'Ridge' in results and results['Ridge'].get('rmse_mean', float('inf')) < results.get('best_rmse', float('inf')):
                    best_model_name = 'Ridge'
            
            # Find best model based on lowest RMSE from all results
            best_rmse = float('inf')
            best_model_name = None
            for model_name, model_results in results.items():
                if 'error' not in model_results:
                    model_rmse = model_results.get('rmse_mean', float('inf'))
                    if model_rmse < best_rmse:
                        best_rmse = model_rmse
                        best_model_name = model_name
            
            # Get all metrics for best model
            best_metrics = results.get(best_model_name, {})
            best_mae = best_metrics.get('mae_mean', 0)
            best_r2 = best_metrics.get('r2_mean', 0)
            
            logger.info(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}, R2: {best_r2:.4f}")
            
            # Step 4: Set model path - use Ridge since it has the best RMSE
            model_path = self.model_factory.output_dir / f"{best_model_name}.pkl"
            
            # Step 5: Register model locally
            self.registry.register_model(
                model_path=model_path,
                metrics={
                    'rmse': best_rmse,
                    'mae': best_mae,
                    'r2': best_r2
                },
                name=best_model_name
            )
            
            # Step 6: Register model to Hopsworks (optional - continues if fails)
            hopsworks_registered = False
            hopsworks_version = None
            try:
                logger.info("Attempting to register model to Hopsworks Model Registry...")
                hopsworks_integration = HopsworksIntegration()
                
                if hopsworks_integration.connect():
                    # Get version tracking - find latest version and increment
                    existing_models = hopsworks_integration.model_registry.get_models(
                        model_name=best_model_name
                    )
                    next_version = 1
                    if existing_models:
                        next_version = max(m.version for m in existing_models) + 1
                    
                    # Prepare metrics for Hopsworks
                    hopsworks_metrics = {
                        'rmse': float(best_rmse),
                        'mae': float(best_mae),
                        'r2': float(best_r2)
                    }
                    
                    # Register the model
                    registered_model = hopsworks_integration.register_model(
                        model_path=model_path,
                        metrics=hopsworks_metrics,
                        description=f"AQI Prediction Model - {best_model_name}"
                    )
                    
                    hopsworks_version = registered_model.version if hasattr(registered_model, 'version') else next_version
                    hopsworks_registered = True
                    logger.info(f"Successfully registered model to Hopsworks: {best_model_name} v{hopsworks_version}")
                else:
                    logger.warning("Could not connect to Hopsworks - skipping model registration")
                    
            except Exception as e:
                logger.warning(f"Failed to register model to Hopsworks (non-blocking): {e}")
                logger.info("Continuing without Hopsworks registration - local development mode")
            finally:
                if 'hopsworks_integration' in locals():
                    hopsworks_integration.disconnect()
            
            # Compile final results
            final_results = {
                'timestamp': datetime.now().isoformat(),
                'best_model': best_model_name,
                'best_rmse': best_rmse,
                'best_mae': best_mae,
                'best_r2': best_r2,
                'all_results': results,
                'model_path': str(model_path),
                'hopsworks_registered': hopsworks_registered,
                'hopsworks_version': hopsworks_version
            }
            
            # Save final results
            final_results_file = settings.MODEL_DIR / "training_summary.json"
            with open(final_results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info(f"Training pipeline completed successfully")
            logger.info(f"Summary saved to {final_results_file}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


async def main():
    """Main entry point for training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AQI Model Training Pipeline")
    parser.add_argument(
        '--data',
        type=str,
        help='Path to feature data CSV file'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        help='Directory to save models'
    )
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    
    data_path = Path(args.data) if args.data else None
    
    results = pipeline.run(data_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Model: {results['best_model']}")
    print(f"Best RMSE: {results['best_rmse']:.4f}")
    print(f"Best MAE: {results['best_mae']:.4f}")
    print(f"Best R2: {results['best_r2']:.4f}")
    print(f"Model Path: {results['model_path']}")
    if results.get('hopsworks_registered'):
        print(f"Hopsworks: Registered (v{results['hopsworks_version']})")
    else:
        print("Hopsworks: Not registered (local mode)")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
