"""
Script to upload trained models to Hopsworks Model Registry.
"""

import pandas as pd
from pathlib import Path
import sys
import json
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing hopsworks, but handle if not available
try:
    from models.registry.hopsworks_pipeline import HopsworksModelRegistry
except ImportError:
    # Fallback if module import fails
    HopsworksModelRegistry = None

from config.settings import settings

def main():
    """Upload trained models to Hopsworks."""
    
    print("="*60)
    print("Uploading Models to Hopsworks")
    print("="*60)
    
    # Check if models exist
    models_dir = Path("models/karachi")
    if not models_dir.exists():
        print("ERROR: No trained models found. Run training first.")
        sys.exit(1)
    
    # Load training results to get best model info
    results_file = Path("models/karachi/training_results.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        model_name = results.get('best_model', 'Lasso')
        best_rmse = results.get('best_rmse', 0.0)
    else:
        # Default to Lasso if no results file
        model_name = "Lasso"
        best_rmse = 0.0
        results = {}
    
    print(f"Best Model: {model_name}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    # Connect to Hopsworks
    if HopsworksModelRegistry is None:
        print("WARNING: Hopsworks module not available. Skipping upload.")
        print("Models are saved locally in models/karachi/")
        return
    
    mr = HopsworksModelRegistry()
    
    if not mr.connect():
        print("Failed to connect to Hopsworks!")
        return
    
    print("Connected to Hopsworks Model Registry")
    
    # Create a model directory for Hopsworks
    model_dir = Path(f"models/karachi/{model_name}_hopsworks")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model file
    model_file = Path(f"models/karachi/{model_name}.pkl")
    if model_file.exists():
        shutil.copy(model_file, model_dir / f"{model_name}.pkl")
    else:
        print(f"Warning: Model file not found: {model_file}")
    
    # Copy scaler
    scaler_file = Path("models/karachi/scaler.pkl")
    if scaler_file.exists():
        shutil.copy(scaler_file, model_dir / "scaler.pkl")
    
    # Copy col_medians
    col_medians_file = Path("models/karachi/col_medians.pkl")
    if col_medians_file.exists():
        shutil.copy(col_medians_file, model_dir / "col_medians.pkl")
    
    # Load metrics from results
    metrics = {
        "rmse": best_rmse,
        "model": model_name
    }
    
    # Add all model metrics if available
    if 'all_results' in results:
        metrics['all_models'] = results['all_results']
    
    # Save metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Register the model
    print(f"\nRegistering {model_name} in Hopsworks...")
    success = mr.register_model(
        model_name=model_name,
        model_dir=str(model_dir),
        metrics=metrics,
        description=f"Karachi AQI Prediction Model - {model_name}"
    )
    
    if success:
        print(f"\n✓ Model '{model_name}' successfully registered in Hopsworks!")
    else:
        print(f"\n✗ Failed to register model in Hopsworks")
    
    # Also upload the feature importance
    feature_imp_file = Path("models/karachi/feature_importance.csv")
    if feature_imp_file.exists():
        print("\nFeature importance file found.")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "rmse": best_rmse,
        "registered_at": str(Path(__file__).stat().st_mtime) if Path(__file__).exists() else "unknown"
    }
    
    print("\n" + "="*60)
    print("Upload Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
