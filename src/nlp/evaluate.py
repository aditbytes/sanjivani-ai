"""
Sanjivani AI - NLP Evaluation

Evaluation metrics for crisis classifier.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)

from src.nlp.model import URGENCY_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: List[int], y_pred: List[int], labels: List[str] = None) -> Dict:
    """
    Compute classification metrics.
    
    Returns:
        Dict with accuracy, precision, recall, f1, and per-class metrics
    """
    labels = labels or URGENCY_CLASSES
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0),
    }


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def print_evaluation_report(y_true: List[int], y_pred: List[int], labels: List[str] = None):
    """Print formatted evaluation report."""
    labels = labels or URGENCY_CLASSES
    
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    
    metrics = compute_metrics(y_true, y_pred, labels)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    print("=" * 60)
