"""
Sanjivani AI - NLP Inference Engine

Production inference for crisis tweet classification.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.config import get_settings
from src.nlp.model import DistilBERTClassifier, URGENCY_CLASSES, RESOURCE_CLASSES, VULNERABILITY_CLASSES
from src.nlp.preprocessing import TextPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class NLPInferenceEngine:
    """Production inference engine for NLP classification."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        self.device = device or settings.inference_device
        self.model_path = model_path or str(settings.nlp_model_path)
        
        self.preprocessor = TextPreprocessor()
        self._model = None
        self._tokenizer = None
        
        logger.info(f"NLPInferenceEngine initialized (device: {self.device})")
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import DistilBertTokenizer
            self._tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        return self._tokenizer
    
    def _load_model(self):
        """Load model weights."""
        try:
            if Path(self.model_path).exists():
                self._model = DistilBERTClassifier.load(self.model_path)
            else:
                logger.warning(f"Model not found at {self.model_path}, using fresh model")
                self._model = DistilBERTClassifier()
            
            self._model.to(self.device)
            self._model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict crisis classification for a single tweet.
        
        Returns:
            Dict with urgency, resource, vulnerability predictions and confidence
        """
        start_time = time.time()
        
        # Preprocess
        processed_text = self.preprocessor(text)
        
        # Tokenize
        encoding = self.tokenizer(
            processed_text,
            max_length=settings.max_tweet_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Predict
        predictions = self.model.predict(input_ids, attention_mask)
        
        inference_time = (time.time() - start_time) * 1000
        
        urgency_idx, urgency_conf = predictions["urgency"]
        resource_idx, resource_conf = predictions["resource"]
        vuln_idx, vuln_conf = predictions["vulnerability"]
        
        return {
            "urgency": URGENCY_CLASSES[urgency_idx],
            "urgency_confidence": round(urgency_conf, 4),
            "resource_needed": RESOURCE_CLASSES[resource_idx],
            "resource_confidence": round(resource_conf, 4),
            "vulnerability": VULNERABILITY_CLASSES[vuln_idx],
            "vulnerability_confidence": round(vuln_conf, 4),
            "processed_text": processed_text,
            "inference_time_ms": round(inference_time, 2),
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict for multiple texts."""
        return [self.predict(text) for text in texts]


# Global inference engine singleton
_engine: Optional[NLPInferenceEngine] = None


def get_nlp_engine() -> NLPInferenceEngine:
    """Get or create the global NLP inference engine."""
    global _engine
    if _engine is None:
        _engine = NLPInferenceEngine()
    return _engine
