"""
Sanjivani AI - Forecasting Inference

Production inference for resource forecasting.
"""

from typing import Any, Dict, Optional

from src.forecasting.ensemble import ResourceForecaster, get_forecaster
from src.forecasting.feature_engineering import extract_spatial_features, extract_temporal_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ForecastingInferenceEngine:
    """Production inference engine for resource forecasting."""
    
    def __init__(self):
        self.forecaster = get_forecaster()
        logger.info("ForecastingInferenceEngine initialized")
    
    def forecast(self, district: str, horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Generate resource forecast for a district.
        
        Args:
            district: Bihar district name
            horizon_hours: Forecast horizon in hours (24, 48, or 72)
            
        Returns:
            Dict with predictions and confidence
        """
        return self.forecaster.predict_for_district(district, horizon_hours)
    
    def forecast_with_features(
        self,
        district: str,
        additional_data: Dict = None,
    ) -> Dict[str, Any]:
        """Forecast with additional feature data."""
        spatial = extract_spatial_features(district, additional_data)
        temporal = extract_temporal_features(district)
        
        predictions = self.forecaster.predict(
            spatial_features=spatial,
            temporal_sequence=temporal,
        )
        
        return {
            "district": district,
            "predictions": predictions,
            "confidence": 0.75,
        }
    
    def batch_forecast(self, districts: list, horizon_hours: int = 24) -> Dict[str, Dict]:
        """Forecast for multiple districts."""
        return {d: self.forecast(d, horizon_hours) for d in districts}


_engine: Optional[ForecastingInferenceEngine] = None


def get_forecasting_engine() -> ForecastingInferenceEngine:
    """Get global forecasting engine."""
    global _engine
    if _engine is None:
        _engine = ForecastingInferenceEngine()
    return _engine
