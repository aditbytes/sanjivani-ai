"""
Sanjivani AI - NLP Pipeline

Complete pipeline combining preprocessing, classification, and location extraction.
"""

from typing import Any, Dict, List, Optional

from src.nlp.inference import NLPInferenceEngine, get_nlp_engine
from src.nlp.location_extractor import LocationExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NLPPipeline:
    """Complete NLP pipeline for crisis tweet analysis."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.classifier = NLPInferenceEngine(model_path)
        self.location_extractor = LocationExtractor()
        logger.info("NLPPipeline initialized")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Full analysis of a crisis tweet.
        
        Returns:
            Complete analysis including classification and location extraction
        """
        # Get classification
        result = self.classifier.predict(text)
        
        # Extract locations
        locations = self.location_extractor.extract(text)
        primary_location = self.location_extractor.extract_primary(text)
        
        result["locations"] = locations
        result["primary_location"] = primary_location
        
        if primary_location:
            result["district"] = primary_location["name"]
            result["latitude"] = primary_location["lat"]
            result["longitude"] = primary_location["lon"]
        else:
            result["district"] = None
            result["latitude"] = None
            result["longitude"] = None
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple tweets."""
        return [self.analyze(text) for text in texts]


# Global pipeline singleton
_pipeline: Optional[NLPPipeline] = None


def get_nlp_pipeline() -> NLPPipeline:
    """Get or create the global NLP pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = NLPPipeline()
    return _pipeline


def analyze_tweet(text: str) -> Dict[str, Any]:
    """Convenience function for single tweet analysis."""
    return get_nlp_pipeline().analyze(text)
