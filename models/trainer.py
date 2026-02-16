"""
Model Training Pipeline for AQI Prediction.
Implements ModelFactory with 6+ candidate models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Try to import optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorFlowNN:
    """TensorFlow Neural Network wrapper."""
    
    def __init__(self, hidden_units: list = [64, 32], dropout: float = 0.3):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.model = None
        self.scaler = None
        self._built = False
    
    def _build_model(self, input_dim: int):
        """Build neural network model."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        for units in self.hidden_units:
            self.model.add(tf.keras.layers.Dense(units, activation="relu"))
            self.model.add(tf.keras.layers.Dropout(self.dropout))
        
        self.model.add(tf.keras.layers.Dense(1))
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> "TensorFlowNN":
        """Train the model."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        
        self.scaler = scaler
        
        X_scaled = scaler.fit_transform(X)
        
        if not self._built:
            self._build_model(X.shape[1])
            self._built = True
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        
        self.model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.scaler is None:
            raise RuntimeError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def save(self, path: str):
        """Save model and scaler."""
        if self.model:
            self.model.save(f"{path}_model.keras")
        if self.scaler:
            joblib.dump(self.scaler, f"{path}_scaler.pkl")
    
    def load(self, path: str):
        """Load model and scaler."""
        if TF_AVAILABLE and Path(f"{path}_model.keras").exists():
            self.model = tf.keras.models.load_model(f"{path}_model.keras")
            self._built = True
        if Path(f"{path}_scaler.pkl").exists():
            self.scaler = joblib.load(f"{path}_scaler.pkl")


class ModelFactory:
    """Factory for training and evaluating AQI prediction models."""
    
    def __init__(self, output_dir: str = "models/karachi"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "training_results.json"
        self.col_medians = None  # Store column medians for inference
    
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "aqi",
        n_splits: int = 5
    ) -> Tuple[Dict, Dict]:
        """Train all models and select the best one."""
        # Prepare features - exclude datetime, target, source, and lag features to avoid leakage
        exclude_cols = ["datetime", "aqi", "source", "main_pollutant"]
        lag_cols = [c for c in df.columns if "_lag_" in c]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and c not in lag_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Drop rows with missing targets
        valid_mask = ~y.isna()
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        
        # Handle missing/infinite values - robust preprocessing
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Calculate column medians from training data
        col_medians = X.median()
        
        # Fill NaN with column medians
        X = X.fillna(col_medians)
        
        # Double check for any remaining NaN (if entire column is NaN, use 0)
        X = X.fillna(0)
        
        # Convert to float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Excluded lag features to avoid data leakage")
        logger.info(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}")
        
        # Store col_medians for later use
        self.col_medians = col_medians
        
        # Define models - 8 different models for comprehensive comparison
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
            "AdaBoost": AdaBoostRegressor(n_estimators=50, learning_rate=0.1),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbosity=0
            ) if XGB_AVAILABLE else None,
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbose=-1
            ) if LGB_AVAILABLE else None,
            "NeuralNetwork": TensorFlowNN(hidden_units=[64, 32])
        }
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            if model is None:
                logger.info(f"Skipping {name} (not available)")
                continue
            
            logger.info(f"Training {name}...")
            
            try:
                model, result = self._train_model(name, model, X, y, tscv)
                results[name] = result
                trained_models[name] = model
                
                logger.info(f"  {name}: RMSE={result['rmse_mean']:.4f} (+/- {result['rmse_std']:.4f}), R2={result['r2_mean']:.4f}")
            
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                results[name] = {"rmse_mean": float("inf"), "rmse_std": 0, "error": str(e)}
        
        # Find best model
        best_name = min(results, key=lambda k: results[k].get("rmse_mean", float("inf")))
        best_result = results[best_name]
        
        logger.info(f"\nBest model: {best_name} (RMSE: {best_result['rmse_mean']:.4f})")
        
        # Save all models
        scaler = StandardScaler()
        scaler.fit(X)
        
        for name, model in trained_models.items():
            if hasattr(model, "save"):
                model.save(str(self.output_dir / name))
            else:
                joblib.dump(model, str(self.output_dir / f"{name}.pkl"))
        
        joblib.dump(scaler, str(self.output_dir / "scaler.pkl"))
        joblib.dump(self.col_medians, str(self.output_dir / "col_medians.pkl"))
        
        # Save results
        training_results = {
            "best_model": best_name,
            "best_rmse": best_result["rmse_mean"],
            "training_date": datetime.utcnow().isoformat(),
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "all_results": results
        }
        
        with open(self.results_file, "w") as f:
            json.dump(training_results, f, indent=2)
        
        # Compute SHAP analysis (optional - won't fail if SHAP unavailable)
        try:
            shap_results = self.compute_shap_analysis(trained_models, X, feature_cols)
            if shap_results:
                training_results["shap_analysis"] = shap_results
                # Update results file with SHAP info
                with open(self.results_file, "w") as f:
                    json.dump(training_results, f, indent=2)
        except Exception as e:
            logger.warning(f"SHAP analysis failed (non-blocking): {e}")
        
        return trained_models, results
    
    def _train_model(self, name: str, model, X: pd.DataFrame, y: pd.Series, tscv):
        """Train a single model with cross-validation."""
        scaler = StandardScaler()
        
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit scaler on train
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Fit model
            if name == "NeuralNetwork":
                model.fit(X_train_scaled, y_train.values, scaler)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
        
        # Retrain on full data
        scaler_final = StandardScaler()
        scaler_final.fit(X)
        X_scaled = scaler_final.transform(X)
        
        if name == "NeuralNetwork":
            model.fit(X_scaled, y.values, scaler_final)
        else:
            model.fit(X_scaled, y)
        
        result = {
            "rmse_mean": float(np.mean(rmse_scores)),
            "rmse_std": float(np.std(rmse_scores)),
            "mae_mean": float(np.mean(mae_scores)),
            "r2_mean": float(np.mean(r2_scores)),
            "n_folds": len(rmse_scores)
        }
        
        return model, result
    
    def compute_shap_analysis(
        self,
        trained_models: Dict[str, Any],
        X: pd.DataFrame,
        feature_cols: list
    ) -> Dict[str, Any]:
        """
        Compute SHAP values and generate feature importance explanations.
        
        Args:
            trained_models: Dictionary of trained model objects
            X: Feature DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Dictionary containing SHAP values and feature importance for each model
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - skipping SHAP analysis")
            return {}
        
        shap_results = {}
        shap_dir = self.output_dir / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample data for SHAP analysis (use subset for efficiency)
        sample_size = min(100, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        X_sample = X_sample.astype(np.float32)
        
        # Tree-based models that work with TreeExplainer
        tree_models = [
            "RandomForest",
            "XGBoost",
            "LightGBM",
            "GradientBoosting",
            "ExtraTrees"
        ]
        
        for name, model in trained_models.items():
            try:
                logger.info(f"Computing SHAP analysis for {name}...")
                
                model_result = {"model_type": "unknown", "shap_values": None, "feature_importance": {}}
                
                # Get the underlying sklearn-compatible model
                if name == "NeuralNetwork":
                    # For Neural Network, use KernelExplainer
                    model_result["model_type"] = "neural_network"
                    
                    # Get scaled data for prediction
                    scaler = joblib.load(self.output_dir / "scaler.pkl")
                    X_scaled = scaler.transform(X_sample)
                    
                    # Use a sample of the data for KernelExplainer
                    X_background = X_scaled[:min(50, len(X_scaled))]
                    
                    # Create prediction function that works with the scaler
                    def predict_fn(x):
                        return model.model.predict(x, verbose=0).flatten()
                    
                    # Use KernelExplainer for neural network
                    explainer = shap.KernelExplainer(predict_fn, X_background)
                    shap_values = explainer.shap_values(X_scaled, nsamples=100)
                    
                    model_result["shap_values"] = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
                    
                    # Calculate feature importance from SHAP values
                    feature_importance = np.abs(shap_values).mean(axis=0)
                    model_result["feature_importance"] = dict(zip(feature_cols, feature_importance.tolist()))
                    
                elif name in tree_models:
                    # For tree-based models, use TreeExplainer
                    model_result["model_type"] = "tree_based"
                    
                    # Get the underlying model
                    if name == "XGBoost":
                        underlying_model = model
                    elif name == "LightGBM":
                        underlying_model = model
                    else:
                        underlying_model = model
                    
                    explainer = shap.TreeExplainer(underlying_model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    model_result["shap_values"] = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
                    
                    # Calculate feature importance from SHAP values
                    feature_importance = np.abs(shap_values).mean(axis=0)
                    model_result["feature_importance"] = dict(zip(feature_cols, feature_importance.tolist()))
                    
                    # Generate and save SHAP summary plot
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
                        plt.tight_layout()
                        plt.savefig(shap_dir / f"{name}_shap_summary.png", dpi=150, bbox_inches='tight')
                        plt.close()
                        logger.info(f"  Saved SHAP summary plot for {name}")
                    except Exception as e:
                        logger.warning(f"  Could not save SHAP plot for {name}: {e}")
                else:
                    # Skip non-tree models that don't have good SHAP support
                    logger.info(f"  Skipping SHAP for {name} (not tree-based or neural network)")
                    continue
                
                shap_results[name] = model_result
                logger.info(f"  SHAP analysis completed for {name}")
                
            except Exception as e:
                logger.warning(f"  SHAP analysis failed for {name}: {e}")
                shap_results[name] = {"model_type": "unknown", "error": str(e)}
        
        # Generate combined feature importance CSV
        self._save_feature_importance_csv(shap_results, feature_cols)
        
        # Save SHAP results to JSON
        shap_file = self.output_dir / "shap_results.json"
        with open(shap_file, 'w') as f:
            json.dump(shap_results, f, indent=2)
        logger.info(f"SHAP results saved to {shap_file}")
        
        return shap_results
    
    def _save_feature_importance_csv(
        self,
        shap_results: Dict[str, Any],
        feature_cols: list
    ):
        """Save feature importance values to CSV file."""
        try:
            importance_data = []
            
            for model_name, result in shap_results.items():
                if "feature_importance" in result:
                    row = {"model": model_name}
                    row.update(result["feature_importance"])
                    importance_data.append(row)
            
            if importance_data:
                df_importance = pd.DataFrame(importance_data)
                csv_path = self.output_dir / "feature_importance.csv"
                df_importance.to_csv(csv_path, index=False)
                logger.info(f"Feature importance saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Could not save feature importance CSV: {e}")
    
    def get_best_model(self):
        """Load the best performing model."""
        if not self.results_file.exists():
            raise FileNotFoundError("Training results not found")
        
        with open(self.results_file) as f:
            results = json.load(f)
        
        best_name = results["best_model"]
        model_path = self.output_dir / best_name
        
        if best_name == "NeuralNetwork":
            model = TensorFlowNN()
            model.load(str(model_path))
            scaler = model.scaler
        else:
            model = joblib.load(f"{model_path}.pkl")
            scaler = joblib.load(self.output_dir / "scaler.pkl")
        
        # Load column medians for inference
        col_medians = joblib.load(self.output_dir / "col_medians.pkl")
        
        return model, scaler, results, col_medians
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        model, scaler, results, col_medians = self.get_best_model()
        
        # Apply same preprocessing as training
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(col_medians).fillna(0)
        
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)
