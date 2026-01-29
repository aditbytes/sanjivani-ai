"""
Sanjivani AI - DistilBERT Crisis Classifier

Multi-label crisis tweet classifier using fine-tuned DistilBERT.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Class labels
URGENCY_CLASSES = ["Critical", "High", "Medium", "Low", "Non-Urgent"]
RESOURCE_CLASSES = ["Rescue", "Medical", "Food", "Shelter", "Information", "Water"]
VULNERABILITY_CLASSES = ["Elderly", "Children", "Disabled", "Pregnant", "None"]


class DistilBERTClassifier(nn.Module):
    """Multi-label DistilBERT classifier for crisis tweets."""
    
    def __init__(
        self,
        num_urgency_classes: int = 5,
        num_resource_classes: int = 6,
        num_vulnerability_classes: int = 5,
        dropout: float = 0.3,
        pretrained_model: str = "distilbert-base-uncased",
    ):
        super().__init__()
        
        self.num_urgency_classes = num_urgency_classes
        self.num_resource_classes = num_resource_classes
        self.num_vulnerability_classes = num_vulnerability_classes
        
        try:
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained(pretrained_model)
            self.hidden_size = self.bert.config.hidden_size
        except ImportError:
            logger.error("transformers not installed")
            raise
        
        self.dropout = nn.Dropout(dropout)
        
        # Multi-head classification
        self.urgency_classifier = nn.Linear(self.hidden_size, num_urgency_classes)
        self.resource_classifier = nn.Linear(self.hidden_size, num_resource_classes)
        self.vulnerability_classifier = nn.Linear(self.hidden_size, num_vulnerability_classes)
        
        logger.info(f"DistilBERTClassifier initialized with {pretrained_model}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits for all classification heads."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        
        return {
            "urgency_logits": self.urgency_classifier(pooled),
            "resource_logits": self.resource_classifier(pooled),
            "vulnerability_logits": self.vulnerability_classifier(pooled),
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Tuple[int, float]]:
        """Predict classes with confidence scores."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            
            urgency_probs = torch.softmax(logits["urgency_logits"], dim=-1)
            resource_probs = torch.softmax(logits["resource_logits"], dim=-1)
            vuln_probs = torch.softmax(logits["vulnerability_logits"], dim=-1)
            
            return {
                "urgency": (urgency_probs.argmax(dim=-1).item(), urgency_probs.max().item()),
                "resource": (resource_probs.argmax(dim=-1).item(), resource_probs.max().item()),
                "vulnerability": (vuln_probs.argmax(dim=-1).item(), vuln_probs.max().item()),
            }
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "DistilBERTClassifier":
        """Load model from weights file."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Model loaded from {path}")
        return model


def get_class_label(class_type: str, idx: int) -> str:
    """Get human-readable label for class index."""
    labels = {
        "urgency": URGENCY_CLASSES,
        "resource": RESOURCE_CLASSES,
        "vulnerability": VULNERABILITY_CLASSES,
    }
    return labels.get(class_type, [])[idx] if idx < len(labels.get(class_type, [])) else "Unknown"
