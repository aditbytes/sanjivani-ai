"""
Sanjivani AI - Image Schemas
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ImageAnalysisResponse(BaseModel):
    """Response schema for image analysis."""
    flood_percentage: float = Field(..., ge=0, le=100)
    num_people_detected: int = Field(default=0, ge=0)
    num_vehicles_detected: int = Field(default=0, ge=0)
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    inference_time_ms: float = 0.0
