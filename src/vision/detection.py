"""
Sanjivani AI - Flood Object Detection

YOLOv8 wrapper for detecting flood-related objects.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Detection classes
DETECTION_CLASSES = ["person", "vehicle", "boat", "building_damaged", "animal"]


class FloodDetector:
    """YOLOv8-based flood object detector."""
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5):
        self.model_path = model_path or str(settings.vision_detection_model_path)
        self.conf_threshold = conf_threshold
        self._model = None
        logger.info(f"FloodDetector initialized (conf: {conf_threshold})")
    
    @property
    def model(self):
        """Lazy load YOLO model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            if Path(self.model_path).exists():
                self._model = YOLO(self.model_path)
                logger.info(f"Loaded custom model from {self.model_path}")
            else:
                self._model = YOLO("yolov8n.pt")  # Use nano model as fallback
                logger.warning("Using pretrained YOLOv8n (custom model not found)")
        except ImportError:
            logger.error("ultralytics not installed")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image.
        
        Returns:
            List of detections with bbox, class, confidence
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                })
        
        logger.debug(f"Detected {len(detections)} objects")
        return detections
    
    def count_objects(self, image: np.ndarray) -> Dict[str, int]:
        """Count objects by class in image."""
        detections = self.detect(image)
        counts = {}
        for det in detections:
            class_name = det["class_name"]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Detect objects in multiple images."""
        return [self.detect(img) for img in images]


def get_detector(model_path: Optional[str] = None) -> FloodDetector:
    """Get detector instance."""
    return FloodDetector(model_path)
