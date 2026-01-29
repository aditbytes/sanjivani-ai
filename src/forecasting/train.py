"""
Sanjivani AI - Forecasting Training

Training pipeline for forecasting models.
"""

from pathlib import Path
from typing import List, Dict

from src.config import get_settings
from src.data.loaders import load_json
from src.forecasting.xgboost_model import XGBoostForecaster
from src.forecasting.lstm_model import LSTMForecaster
from src.forecasting.feature_engineering import prepare_training_data
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def train_forecasting_models(
    historical_data: List[Dict] = None,
    output_dir: Path = None,
) -> Dict:
    """Train XGBoost and LSTM forecasting models."""
    output_dir = output_dir or settings.models_dir / "forecasting"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data if not provided
    if historical_data is None:
        historical_data = load_json(settings.data_dir / "processed" / "historical_floods.json") or []
    
    if len(historical_data) < 10:
        logger.warning("Insufficient training data")
        return {"status": "insufficient_data"}
    
    # Prepare features
    X_spatial, X_temporal, y = prepare_training_data(historical_data)
    
    # Train XGBoost
    logger.info("Training XGBoost models...")
    xgb = XGBoostForecaster()
    xgb.train(X_spatial, y)
    xgb.save(output_dir)
    
    # Train LSTM
    logger.info("Training LSTM model...")
    lstm = LSTMForecaster()
    y_combined = list(zip(y["food_packets"], y["medical_kits"], y["rescue_boats"], y["shelters"]))
    lstm.train(X_temporal, y_combined, epochs=50)
    lstm.save(str(output_dir / "lstm_model.h5"))
    
    return {"status": "success", "samples": len(historical_data)}


if __name__ == "__main__":
    train_forecasting_models()
