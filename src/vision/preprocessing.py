"""
Sanjivani AI - Vision Preprocessing

Image augmentation and preprocessing using Albumentations.
"""

from typing import Callable, Dict, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_train_transform(image_size: int = 512) -> Callable:
    """Get training augmentation pipeline."""
    try:
        import albumentations as A
        
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    except ImportError:
        logger.warning("Albumentations not installed, using identity transform")
        return lambda **x: x


def get_val_transform(image_size: int = 512) -> Callable:
    """Get validation/test transform (no augmentation)."""
    try:
        import albumentations as A
        
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    except ImportError:
        return lambda **x: x


def preprocess_image(image: np.ndarray, image_size: int = 512) -> np.ndarray:
    """Preprocess single image for inference."""
    try:
        import cv2
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        return image
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image
