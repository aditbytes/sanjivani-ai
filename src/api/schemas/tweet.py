"""
Sanjivani AI - Tweet Schemas
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TweetInput(BaseModel):
    """Input schema for tweet analysis."""
    text: str = Field(..., min_length=1, max_length=1000, description="Tweet text")
    source_id: Optional[str] = Field(None, description="Original tweet ID")
    
    model_config = {"json_schema_extra": {"example": {"text": "URGENT! Flooding in Patna. Need rescue immediately!"}}}


class BatchTweetInput(BaseModel):
    """Input schema for batch tweet analysis."""
    texts: List[str] = Field(..., min_length=1, max_length=100)


class TweetAnalysisResponse(BaseModel):
    """Response schema for tweet analysis."""
    text: str
    processed_text: str
    urgency: str = Field(..., description="Urgency level: Critical, High, Medium, Low, Non-Urgent")
    urgency_confidence: float = Field(..., ge=0, le=1)
    resource_needed: Optional[str] = None
    vulnerability: Optional[str] = None
    district: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    inference_time_ms: float = 0.0
