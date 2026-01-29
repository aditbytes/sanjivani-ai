"""
Sanjivani AI - Forecasting API Endpoints
"""

from typing import Dict, List

from fastapi import APIRouter

from src.api.schemas.prediction import ForecastRequest, ForecastResponse
from src.forecasting.inference import get_forecasting_engine
from src.utils.helpers import get_all_district_names
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/forecast", response_model=ForecastResponse)
async def forecast_resources(request: ForecastRequest) -> ForecastResponse:
    """Forecast resource needs for a district."""
    engine = get_forecasting_engine()
    result = engine.forecast(request.district, request.horizon_hours)
    
    return ForecastResponse(
        district=result["district"],
        horizon_hours=result["horizon_hours"],
        food_packets=result["predictions"]["food_packets"],
        medical_kits=result["predictions"]["medical_kits"],
        rescue_boats=result["predictions"]["rescue_boats"],
        shelters=result["predictions"]["shelters"],
        confidence=result["confidence"],
    )


@router.get("/forecast/{district}")
async def forecast_district(district: str, horizon: int = 24) -> Dict:
    """Get forecast for a specific district."""
    engine = get_forecasting_engine()
    return engine.forecast(district, horizon)


@router.get("/districts")
async def list_districts() -> List[str]:
    """List all Bihar districts."""
    return get_all_district_names()
