"""
Sanjivani AI - Vision API Endpoints
"""

import base64
from typing import Dict

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File

from src.api.schemas.image import ImageAnalysisResponse
from src.vision.inference import get_vision_engine
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)) -> ImageAnalysisResponse:
    """Analyze satellite image for flood damage."""
    try:
        import cv2
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        engine = get_vision_engine()
        result = engine.analyze_image(image)
        
        return ImageAnalysisResponse(
            flood_percentage=result["flood_percentage"],
            num_people_detected=result["num_people"],
            num_vehicles_detected=result["num_vehicles"],
            detections=result["detections"],
            inference_time_ms=result["inference_time_ms"],
        )
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-flood")
async def segment_flood(file: UploadFile = File(...)) -> Dict:
    """Get flood segmentation mask only."""
    try:
        import cv2
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        engine = get_vision_engine()
        result = engine.segment_only(image)
        
        return result
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
