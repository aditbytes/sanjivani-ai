"""
Sanjivani AI - Vision Dataset

PyTorch Dataset for satellite imagery.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SatelliteDataset(Dataset):
    """Dataset for satellite images with segmentation masks."""
    
    def __init__(
        self,
        data: List[Dict],
        image_dir: Path,
        mask_dir: Optional[Path] = None,
        transform: Optional[Callable] = None,
        image_size: int = 512,
    ):
        self.data = data
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size
        
        logger.info(f"SatelliteDataset initialized with {len(data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        try:
            import cv2
            
            # Load image
            image_path = self.image_dir / item["image_file"]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            # Load mask if available
            mask = None
            if self.mask_dir and "mask_file" in item:
                mask_path = self.mask_dir / item["mask_file"]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                # Normalize mask values: convert 0-255 range to class indices 0/1
                # Values > 127 are considered flood (class 1), others background (class 0)
                mask = (mask > 127).astype(np.uint8)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image, mask=mask) if mask is not None else self.transform(image=image)
                image = transformed["image"]
                mask = transformed.get("mask", mask)
            
            # Convert to tensors
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            
            result = {"image": image_tensor, "image_id": item.get("id", str(idx))}
            
            if mask is not None:
                result["mask"] = torch.from_numpy(mask.astype(np.int64)).long()
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return placeholder
            return {
                "image": torch.zeros(3, self.image_size, self.image_size),
                "mask": torch.zeros(self.image_size, self.image_size, dtype=torch.long),
                "image_id": str(idx),
            }


class ChangeDetectionDataset(Dataset):
    """Dataset for change detection with pre/post image pairs."""
    
    def __init__(self, data: List[Dict], image_dir: Path, transform: Optional[Callable] = None):
        self.data = data
        self.image_dir = Path(image_dir)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        try:
            import cv2
            
            pre = cv2.imread(str(self.image_dir / item["pre_image"]))
            post = cv2.imread(str(self.image_dir / item["post_image"]))
            
            pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
            post = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)
            
            pre_tensor = torch.from_numpy(pre).float().permute(2, 0, 1) / 255.0
            post_tensor = torch.from_numpy(post).float().permute(2, 0, 1) / 255.0
            
            return {"pre_image": pre_tensor, "post_image": post_tensor}
            
        except Exception as e:
            logger.error(f"Error loading pair {idx}: {e}")
            return {"pre_image": torch.zeros(3, 512, 512), "post_image": torch.zeros(3, 512, 512)}
