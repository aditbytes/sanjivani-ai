"""
Sanjivani AI - NLP Dataset

PyTorch Dataset for crisis tweet classification.
"""

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.nlp.model import URGENCY_CLASSES, RESOURCE_CLASSES, VULNERABILITY_CLASSES
from src.nlp.preprocessing import TextPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrisisTweetDataset(Dataset):
    """PyTorch Dataset for crisis tweet classification."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 128,
        preprocess: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = TextPreprocessor() if preprocess else None
        
        # Label mappings
        self.urgency_to_idx = {label: i for i, label in enumerate(URGENCY_CLASSES)}
        self.resource_to_idx = {label: i for i, label in enumerate(RESOURCE_CLASSES)}
        self.vuln_to_idx = {label: i for i, label in enumerate(VULNERABILITY_CLASSES)}
        
        logger.info(f"Dataset initialized with {len(data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item.get("text", "")
        
        if self.preprocessor:
            text = self.preprocessor(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        # Add labels if available
        if "urgency" in item:
            result["urgency_label"] = torch.tensor(
                self.urgency_to_idx.get(item["urgency"], 2)  # Default to Medium
            )
        if "resource_needed" in item:
            result["resource_label"] = torch.tensor(
                self.resource_to_idx.get(item["resource_needed"], 4)  # Default to Information
            )
        if "vulnerability" in item:
            result["vulnerability_label"] = torch.tensor(
                self.vuln_to_idx.get(item["vulnerability"], 4)  # Default to None
            )
        
        return result


def create_dataloader(
    data: List[Dict],
    tokenizer,
    batch_size: int = 16,
    shuffle: bool = True,
    max_length: int = 128,
):
    """Create DataLoader for training/evaluation."""
    from torch.utils.data import DataLoader
    
    dataset = CrisisTweetDataset(data, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
