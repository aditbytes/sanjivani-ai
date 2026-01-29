"""
Sanjivani AI - XGBoost Forecasting Model

Spatial feature-based resource prediction using XGBoost.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Feature names for XGBoost model
SPATIAL_FEATURES = [
    "elevation_m", "slope_deg", "distance_to_river_km", "drainage_density",
    "soil_type", "land_use", "population_density", "household_count",
    "vulnerable_population", "historical_flood_freq", "avg_flood_depth_m",
    "road_density", "hospital_count", "school_count", "shelter_capacity",
    "current_water_level_m", "rainfall_24h_mm", "rainfall_forecast_mm",
    "upstream_discharge", "groundwater_level",
]

RESOURCE_OUTPUTS = ["food_packets", "medical_kits", "rescue_boats", "shelters"]


class XGBoostForecaster:
    """XGBoost-based resource forecaster using spatial features."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(settings.forecasting_xgboost_path)
        self._models = {}  # One model per resource type
        logger.info("XGBoostForecaster initialized")
    
    def _get_model(self, resource_type: str):
        """Get or load model for resource type."""
        if resource_type not in self._models:
            try:
                import xgboost as xgb
                path = Path(self.model_path).parent / f"xgboost_{resource_type}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        self._models[resource_type] = pickle.load(f)
                else:
                    # Create default model
                    self._models[resource_type] = xgb.XGBRegressor(
                        n_estimators=200, max_depth=6, learning_rate=0.1
                    )
                    logger.warning(f"No trained model for {resource_type}")
            except ImportError:
                logger.error("xgboost not installed")
                raise
        return self._models[resource_type]
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict resource needs.
        
        Args:
            features: Array of spatial features (20 features)
            
        Returns:
            Dict mapping resource types to predicted quantities
        """
        predictions = {}
        for resource in RESOURCE_OUTPUTS:
            try:
                model = self._get_model(resource)
                if hasattr(model, "predict"):
                    pred = model.predict(features.reshape(1, -1))[0]
                    predictions[resource] = max(0, int(pred))
                else:
                    predictions[resource] = 0
            except Exception as e:
                logger.error(f"Prediction error for {resource}: {e}")
                predictions[resource] = 0
        
        return predictions
    
    def train(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> None:
        """Train models on historical data."""
        import xgboost as xgb
        
        for resource in RESOURCE_OUTPUTS:
            if resource in y:
                model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)
                model.fit(X, y[resource])
                self._models[resource] = model
                logger.info(f"Trained XGBoost for {resource}")
    
    def save(self, output_dir: Path) -> None:
        """Save all trained models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for resource, model in self._models.items():
            with open(output_dir / f"xgboost_{resource}.pkl", "wb") as f:
                pickle.dump(model, f)
        logger.info(f"Models saved to {output_dir}")
