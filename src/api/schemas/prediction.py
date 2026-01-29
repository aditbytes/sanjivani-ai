"""
Sanjivani AI - Prediction Schemas
"""

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request schema for resource forecast."""
    district: str = Field(..., description="Bihar district name")
    horizon_hours: int = Field(default=24, ge=1, le=168, description="Forecast horizon in hours")


class ForecastResponse(BaseModel):
    """Response schema for resource forecast."""
    district: str
    horizon_hours: int
    food_packets: int = Field(..., ge=0)
    medical_kits: int = Field(..., ge=0)
    rescue_boats: int = Field(..., ge=0)
    shelters: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
