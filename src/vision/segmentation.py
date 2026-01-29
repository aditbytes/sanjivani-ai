"""
Sanjivani AI - Flood Segmentation Model

U-Net with ResNet50 encoder for flood extent segmentation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Segmentation classes
SEGMENTATION_CLASSES = ["background", "water", "cloud"]
NUM_CLASSES = len(SEGMENTATION_CLASSES)


class FloodSegmentationModel(nn.Module):
    """U-Net model for flood extent segmentation using segmentation-models-pytorch."""
    
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        try:
            import segmentation_models_pytorch as smp
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
            )
            logger.info(f"U-Net initialized with {encoder_name} encoder")
        except ImportError:
            logger.error("segmentation-models-pytorch not installed")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict flood mask for a single image.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Tuple of (mask array, flood percentage)
        """
        self.eval()
        
        # Normalize and convert to tensor
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            output = self.forward(image_tensor)
            mask = output.argmax(dim=1).squeeze().numpy()
        
        # Calculate flood percentage (class 1 = water)
        flood_pct = (mask == 1).sum() / mask.size
        
        return mask, float(flood_pct)
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info(f"Segmentation model saved to {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "FloodSegmentationModel":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Segmentation model loaded from {path}")
        return model


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Focal loss."""
    
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice_loss(pred, target) + self.focal_weight * self.focal_loss(pred, target)
