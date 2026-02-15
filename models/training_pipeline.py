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
            
            # Get RMSE from the results
            best_rmse = results.get(best_model_name, {}).get('rmse_mean', results.get('best_rmse', 0))
            logger.info(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
            
            # Step 4: Set model path - use Ridge since it has the best RMSE
            model_path = self.model_factory.output_dir / f"{best_model_name}.pkl"
            
            # Step 5: Register model locally
            self.registry.register_model(
                model_path=model_path,
                metrics={
                    'rmse': best_rmse,
                },
                name=best_model_name
            )
            
            # Compile final results
            final_results = {
                'timestamp': datetime.now().isoformat(),
                'best_model': best_model_name,
                'best_rmse': best_rmse,
                'all_results': results,
                'model_path': str(model_path)
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
    print(f"Model Path: {results['model_path']}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
