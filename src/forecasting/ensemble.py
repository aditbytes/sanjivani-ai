"""
Sanjivani AI - Ensemble Forecaster

Weighted ensemble combining XGBoost and LSTM predictions.
"""

from typing import Dict, Optional

import numpy as np

from src.forecasting.xgboost_model import XGBoostForecaster, RESOURCE_OUTPUTS
from src.forecasting.lstm_model import LSTMForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResourceForecaster:
    """Ensemble forecaster combining XGBoost (spatial) and LSTM (temporal)."""
    
    def __init__(self, xgboost_weight: float = 0.6, lstm_weight: float = 0.4):
        self.xgboost_weight = xgboost_weight
        self.lstm_weight = lstm_weight
        self._xgboost = None
        self._lstm = None
        logger.info(f"ResourceForecaster initialized (XGB: {xgboost_weight}, LSTM: {lstm_weight})")
    
    @property
    def xgboost(self):
        if self._xgboost is None:
            self._xgboost = XGBoostForecaster()
        return self._xgboost
    
    @property
    def lstm(self):
        if self._lstm is None:
            self._lstm = LSTMForecaster()
        return self._lstm
    
    def predict(
        self,
        spatial_features: Optional[np.ndarray] = None,
        temporal_sequence: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        """
        Predict resource needs using ensemble.
        
        Args:
            spatial_features: Shape (20,) - spatial features for XGBoost
            temporal_sequence: Shape (24, 10) - temporal sequence for LSTM
            
        Returns:
            Dict with predicted quantities for each resource type
        """
        predictions = {r: 0 for r in RESOURCE_OUTPUTS}
        total_weight = 0
        
        if spatial_features is not None:
            xgb_preds = self.xgboost.predict(spatial_features)
            for r in RESOURCE_OUTPUTS:
                predictions[r] += xgb_preds[r] * self.xgboost_weight
            total_weight += self.xgboost_weight
        
        if temporal_sequence is not None:
            lstm_preds = self.lstm.predict(temporal_sequence)
            for r in RESOURCE_OUTPUTS:
                predictions[r] += lstm_preds[r] * self.lstm_weight
            total_weight += self.lstm_weight
        
        # Normalize if only one model used
        if total_weight > 0 and total_weight != 1.0:
            for r in RESOURCE_OUTPUTS:
                predictions[r] = int(predictions[r] / total_weight)
        else:
            predictions = {r: int(predictions[r]) for r in RESOURCE_OUTPUTS}
        
        return predictions
    
    def predict_for_district(self, district: str, horizon_hours: int = 24) -> Dict:
        """
        Predict resource needs for a district.
        
        This is a placeholder that would use actual district data.
        """
        logger.info(f"Forecasting for {district}, horizon: {horizon_hours}h")
        
        # Generate placeholder predictions based on district characteristics
        np.random.seed(hash(district) % 2**32)
        
        base_preds = {
            "food_packets": np.random.randint(500, 5000),
            "medical_kits": np.random.randint(50, 500),
            "rescue_boats": np.random.randint(5, 50),
            "shelters": np.random.randint(2, 20),
        }
        
        # Scale by horizon
        scale = horizon_hours / 24
        return {
            "district": district,
            "horizon_hours": horizon_hours,
            "predictions": {k: int(v * scale) for k, v in base_preds.items()},
            "confidence": 0.75,
        }


_forecaster: Optional[ResourceForecaster] = None


def get_forecaster() -> ResourceForecaster:
    """Get global forecaster instance."""
    global _forecaster
    if _forecaster is None:
        _forecaster = ResourceForecaster()
    return _forecaster
