"""
Sanjivani AI - Vision Inference Engine

Production inference for satellite imagery analysis.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VisionInferenceEngine:
    """Production inference engine for vision models."""
    
    def __init__(self):
        self._segmentation_model = None
        self._detection_model = None
        self._change_model = None
        logger.info("VisionInferenceEngine initialized")
    
    @property
    def segmentation_model(self):
        if self._segmentation_model is None:
            from src.vision.segmentation import FloodSegmentationModel
            path = settings.vision_segmentation_model_path
            if Path(path).exists():
                self._segmentation_model = FloodSegmentationModel.load(str(path))
            else:
                self._segmentation_model = FloodSegmentationModel()
                logger.warning("Using untrained segmentation model")
        return self._segmentation_model
    
    @property
    def detector(self):
        if self._detection_model is None:
            from src.vision.detection import FloodDetector
            self._detection_model = FloodDetector()
        return self._detection_model
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Full analysis of satellite image.
        
        Returns:
            Dict with segmentation mask, detections, and metrics
        """
        start = time.time()
        
        # Segmentation
        mask, flood_pct = self.segmentation_model.predict(image)
        
        # Detection
        detections = self.detector.detect(image)
        counts = self.detector.count_objects(image)
        
        inference_time = (time.time() - start) * 1000
        
        return {
            "flood_mask": mask.tolist(),
            "flood_percentage": round(flood_pct * 100, 2),
            "detections": detections,
            "object_counts": counts,
            "num_people": counts.get("person", 0),
            "num_vehicles": counts.get("vehicle", 0) + counts.get("car", 0),
            "inference_time_ms": round(inference_time, 2),
        }
    
    def segment_only(self, image: np.ndarray) -> Dict[str, Any]:
        """Run segmentation only."""
        mask, flood_pct = self.segmentation_model.predict(image)
        return {"mask": mask.tolist(), "flood_percentage": round(flood_pct * 100, 2)}
    
    def detect_only(self, image: np.ndarray) -> List[Dict]:
        """Run detection only."""
        return self.detector.detect(image)


_engine: Optional[VisionInferenceEngine] = None


def get_vision_engine() -> VisionInferenceEngine:
    """Get global vision inference engine."""
    global _engine
    if _engine is None:
        _engine = VisionInferenceEngine()
    return _engine
