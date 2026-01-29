"""
Sanjivani AI - Feature Engineering

Feature extraction for forecasting models.
"""

from typing import Dict, List

import numpy as np

from src.utils.helpers import BIHAR_DISTRICTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_spatial_features(district: str, additional_data: Dict = None) -> np.ndarray:
    """
    Extract 20 spatial features for a district.
    
    Uses district characteristics and any additional data provided.
    """
    additional_data = additional_data or {}
    
    # Get base district info
    coords = BIHAR_DISTRICTS.get(district, {"lat": 25.5, "lon": 85.1})
    
    # Generate features (placeholder - would use actual GIS data)
    np.random.seed(hash(district) % 2**32)
    
    features = np.array([
        additional_data.get("elevation_m", 50 + np.random.rand() * 100),
        additional_data.get("slope_deg", np.random.rand() * 10),
        additional_data.get("distance_to_river_km", np.random.rand() * 50),
        additional_data.get("drainage_density", np.random.rand()),
        additional_data.get("soil_type", np.random.randint(1, 5)),
        additional_data.get("land_use", np.random.randint(1, 7)),
        additional_data.get("population_density", 500 + np.random.rand() * 2000),
        additional_data.get("household_count", 10000 + np.random.randint(0, 50000)),
        additional_data.get("vulnerable_population", 1000 + np.random.randint(0, 10000)),
        additional_data.get("historical_flood_freq", np.random.rand() * 5),
        additional_data.get("avg_flood_depth_m", np.random.rand() * 3),
        additional_data.get("road_density", np.random.rand()),
        additional_data.get("hospital_count", np.random.randint(1, 10)),
        additional_data.get("school_count", np.random.randint(10, 100)),
        additional_data.get("shelter_capacity", np.random.randint(100, 5000)),
        additional_data.get("current_water_level_m", np.random.rand() * 5),
        additional_data.get("rainfall_24h_mm", np.random.rand() * 200),
        additional_data.get("rainfall_forecast_mm", np.random.rand() * 150),
        additional_data.get("upstream_discharge", np.random.rand() * 1000),
        additional_data.get("groundwater_level", np.random.rand() * 10),
    ], dtype=np.float32)
    
    return features


def extract_temporal_features(district: str, hours: int = 24) -> np.ndarray:
    """
    Extract temporal sequence for a district.
    
    Returns shape (hours, 10) with 10 temporal features per hour.
    """
    np.random.seed(hash(district) % 2**32)
    
    # Generate time series (placeholder)
    sequence = np.zeros((hours, 10), dtype=np.float32)
    
    for h in range(hours):
        sequence[h] = [
            np.random.rand() * 20,       # rainfall_mm
            5 + np.random.rand() * 5,    # river_level_m
            25 + np.random.rand() * 10,  # temperature_c
            60 + np.random.rand() * 30,  # humidity_pct
            np.random.rand() * 30,       # wind_speed_kmh
            np.random.randint(0, 50),    # alerts_count
            np.random.randint(0, 1000),  # evacuees_count
            np.random.randint(0, 100),   # resources_deployed
            np.random.rand() * 30,       # upstream_rainfall
            np.random.rand() * 3,        # downstream_level
        ]
    
    return sequence


def prepare_training_data(historical_events: List[Dict]) -> tuple:
    """Prepare training data from historical flood events."""
    X_spatial = []
    X_temporal = []
    y = {"food_packets": [], "medical_kits": [], "rescue_boats": [], "shelters": []}
    
    for event in historical_events:
        district = event.get("primary_district", "Patna")
        
        X_spatial.append(extract_spatial_features(district))
        X_temporal.append(extract_temporal_features(district))
        
        y["food_packets"].append(event.get("food_packets", 0))
        y["medical_kits"].append(event.get("medical_kits", 0))
        y["rescue_boats"].append(event.get("rescue_boats", 0))
        y["shelters"].append(event.get("shelters_established", 0))
    
    return (
        np.array(X_spatial),
        np.array(X_temporal),
        {k: np.array(v) for k, v in y.items()},
    )
