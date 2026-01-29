"""
Sanjivani AI - Change Detection

Siamese network for pre/post disaster change detection.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SiameseEncoder(nn.Module):
    """CNN encoder for Siamese network."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x


class ChangeDetectionModel(nn.Module):
    """Siamese network for change detection between pre and post disaster images."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = SiameseEncoder(in_channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        logger.info("ChangeDetectionModel initialized")
    
    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Compute change mask between pre and post images."""
        feat_pre = self.encoder(pre)
        feat_post = self.encoder(post)
        diff = torch.cat([feat_pre, feat_post], dim=1)
        return self.decoder(diff)
    
    def predict(self, pre: np.ndarray, post: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Predict change mask.
        
        Returns:
            Tuple of (change mask, change score 0-1)
        """
        self.eval()
        
        pre_t = torch.from_numpy(pre).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        post_t = torch.from_numpy(post).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            change_prob = self.forward(pre_t, post_t)
            change_mask = (change_prob > threshold).squeeze().numpy().astype(np.uint8)
        
        change_score = change_mask.sum() / change_mask.size
        return change_mask, float(change_score)
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str) -> "ChangeDetectionModel":
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
