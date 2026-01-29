"""
Sanjivani AI - NLP Training

Training script for the DistilBERT crisis classifier.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_settings
from src.nlp.model import DistilBERTClassifier
from src.nlp.dataset import CrisisTweetDataset
from src.data.loaders import load_json
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def train_epoch(model, dataloader, optimizer, device, criterion) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        urgency_labels = batch["urgency_label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs["urgency_logits"], urgency_labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, criterion) -> Dict[str, float]:
    """Evaluate model. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            urgency_labels = batch["urgency_label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs["urgency_logits"], urgency_labels)
            total_loss += loss.item()
            
            preds = outputs["urgency_logits"].argmax(dim=-1)
            correct += (preds == urgency_labels).sum().item()
            total += urgency_labels.size(0)
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total if total > 0 else 0,
    }


def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    output_dir: Path = None,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = None,
) -> Dict:
    """
    Full training pipeline.
    
    Returns:
        Training history dict
    """
    from transformers import DistilBertTokenizer
    
    output_dir = output_dir or settings.models_dir / "nlp"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERTClassifier().to(device)
    
    # Create dataloaders
    train_dataset = CrisisTweetDataset(train_data, tokenizer)
    val_dataset = CrisisTweetDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save(str(output_dir / "best_model.pth"))
            logger.info("Saved best model")
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return history


if __name__ == "__main__":
    train_data = load_json(settings.data_dir / "processed" / "train.json") or []
    val_data = load_json(settings.data_dir / "processed" / "val.json") or []
    
    if train_data and val_data:
        train_model(train_data, val_data)
    else:
        logger.error("No training data found")
