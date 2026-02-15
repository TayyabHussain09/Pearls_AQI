"""
Script to train and register models in Hopsworks Model Registry.
"""

import pandas as pd
from pathlib import Path
import sys
import json
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training_pipeline import TrainingPipeline
from models.registry.hopsworks_pipeline import HopsworksModelRegistry
from config.settings import settings

def main():
    """Train models and register in Hopsworks."""
    
    # First, run training
    print("="*60)
    print("STEP 1: Training Models")
    print("="*60)
    
    pipeline = TrainingPipeline()
    
    # Use existing processed data
    data_path = Path("data/processed/karachi_features_20240216_20260215.csv")
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return
    
    print(f"Training with data from {data_path}...")
    results = pipeline.run(data_path)
    
    print(f"\nBest Model: {results['best_model']}")
    print(f"Best RMSE: {results['best_rmse']:.4f}")
    
    # Now register in Hopsworks
    print("\n" + "="*60)
    print("STEP 2: Registering Model in Hopsworks")
    print("="*60)
    
    # Connect to Hopsworks
    mr = HopsworksModelRegistry()
    
    if not mr.connect():
        print("Failed to connect to Hopsworks!")
        return
    
    print("Connected to Hopsworks Model Registry")
    
    # Get model path and metrics
    model_name = results['best_model']
    best_rmse = results['best_rmse']
    
    # Create a model directory for Hopsworks
    model_dir = Path(f"models/karachi/{model_name}_hopsworks")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model file
    model_file = Path(f"models/karachi/{model_name}.pkl")
    if model_file.exists():
        shutil.copy(model_file, model_dir / f"{model_name}.pkl")
    
    # Copy scaler
    scaler_file = Path("models/karachi/scaler.pkl")
    if scaler_file.exists():
        shutil.copy(scaler_file, model_dir / "scaler.pkl")
    
    # Copy col_medians
    col_medians_file = Path("models/karachi/col_medians.pkl")
    if col_medians_file.exists():
        shutil.copy(col_medians_file, model_dir / "col_medians.pkl")
    
    # Get metrics from results - handle different formats
    all_results = results.get('all_results', {})
    model_result = all_results.get(model_name, {})
    
    metrics = {
        'rmse': best_rmse,
    }
    
    # Add other metrics if available
    if 'rmse_mean' in model_result:
        metrics['rmse_mean'] = model_result['rmse_mean']
    if 'r2_mean' in model_result:
        metrics['r2_mean'] = model_result['r2_mean']
    
    print(f"\nRegistering model: {model_name}")
    print(f"Metrics: {metrics}")
    
    try:
        model_version = mr.register_model(
            model_path=model_dir,
            model_name=model_name,
            metrics=metrics,
            description=f"AQI Prediction Model - {model_name}"
        )
        print(f"Successfully registered model: {model_name}")
        print(f"Model path: {model_dir}")
    except Exception as e:
        print(f"Error registering model: {e}")
    
    # Disconnect
    mr.disconnect()
    print("\nDone!")

if __name__ == "__main__":
    main()
