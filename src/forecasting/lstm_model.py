"""
Sanjivani AI - LSTM Forecasting Model

Temporal pattern-based resource prediction using LSTM.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

TEMPORAL_FEATURES = [
    "rainfall_mm", "river_level_m", "temperature_c", "humidity_pct",
    "wind_speed_kmh", "alerts_count", "evacuees_count", "resources_deployed",
    "upstream_rainfall", "downstream_level",
]

RESOURCE_OUTPUTS = ["food_packets", "medical_kits", "rescue_boats", "shelters"]


class LSTMForecaster:
    """LSTM-based resource forecaster using temporal patterns."""
    
    def __init__(self, model_path: Optional[str] = None, sequence_length: int = 24):
        self.model_path = model_path or str(settings.forecasting_lstm_path)
        self.sequence_length = sequence_length
        self._model = None
        logger.info("LSTMForecaster initialized")
    
    @property
    def model(self):
        """Lazy load TensorFlow model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load or create LSTM model."""
        try:
            from tensorflow.keras.models import load_model, Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            if Path(self.model_path).exists():
                self._model = load_model(self.model_path)
            else:
                # Create default architecture
                self._model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(TEMPORAL_FEATURES))),
                    Dropout(0.2),
                    LSTM(32),
                    Dropout(0.2),
                    Dense(len(RESOURCE_OUTPUTS)),
                ])
                self._model.compile(optimizer="adam", loss="mse")
                logger.warning("Using untrained LSTM model")
        except ImportError:
            logger.error("tensorflow not installed")
            raise
    
    def predict(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        Predict resource needs from time series.
        
        Args:
            sequence: Shape (24, 10) - 24 hours of 10 temporal features
            
        Returns:
            Dict mapping resource types to predicted quantities
        """
        if sequence.shape != (self.sequence_length, len(TEMPORAL_FEATURES)):
            logger.error(f"Invalid sequence shape: {sequence.shape}")
            return {r: 0 for r in RESOURCE_OUTPUTS}
        
        try:
            preds = self.model.predict(sequence.reshape(1, self.sequence_length, -1), verbose=0)[0]
            return {resource: max(0, int(pred)) for resource, pred in zip(RESOURCE_OUTPUTS, preds)}
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {r: 0 for r in RESOURCE_OUTPUTS}
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM on historical sequences."""
        from tensorflow.keras.callbacks import EarlyStopping
        
        early_stop = EarlyStopping(patience=5, restore_best_weights=True)
        history = self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2,
            callbacks=[early_stop], verbose=1
        )
        return history.history
    
    def save(self, path: str = None) -> None:
        """Save trained model."""
        path = path or self.model_path
        self.model.save(path)
        logger.info(f"LSTM model saved to {path}")
