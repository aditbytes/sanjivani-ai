"""
Sanjivani AI - Segmentation Training

Training script for U-Net flood segmentation.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_settings
from src.vision.segmentation import FloodSegmentationModel, CombinedLoss
from src.vision.dataset import SatelliteDataset
from src.vision.preprocessing import get_train_transform, get_val_transform
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate IoU
            preds = outputs.argmax(dim=1)
            intersection = ((preds == 1) & (masks == 1)).sum().item()
            union = ((preds == 1) | (masks == 1)).sum().item()
            iou = intersection / (union + 1e-8)
            total_iou += iou
    
    return {
        "loss": total_loss / len(dataloader),
        "iou": total_iou / len(dataloader),
    }


def train_segmentation(
    train_data: List[Dict],
    val_data: List[Dict],
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path = None,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = None,
) -> Dict:
    """Train segmentation model."""
    output_dir = output_dir or settings.models_dir / "vision"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    train_dataset = SatelliteDataset(train_data, image_dir, mask_dir, get_train_transform())
    val_dataset = SatelliteDataset(val_data, image_dir, mask_dir, get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model and training setup
    model = FloodSegmentationModel().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CombinedLoss()
    
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou = 0.0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            model.save(str(output_dir / "unet_segmentation.pth"))
    
    with open(output_dir / "segmentation_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return history


if __name__ == "__main__":
    logger.info("Run with sample data using scripts/generate_sample_data.py first")
