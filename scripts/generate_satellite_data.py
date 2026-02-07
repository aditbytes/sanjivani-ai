"""
Generate synthetic satellite imagery for training vision models.

Creates:
- Flood segmentation masks
- Synthetic satellite images with flood patterns
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import json
import random

# Configuration
IMAGE_SIZE = 256
NUM_TRAIN_IMAGES = 200
NUM_VAL_IMAGES = 50
NUM_TEST_IMAGES = 50

OUTPUT_DIR = project_root / "data" / "satellite"


def create_flood_pattern(size: int) -> np.ndarray:
    """Create a realistic flood pattern mask."""
    mask = np.zeros((size, size), dtype=np.float32)
    
    # Create random flood regions using multiple ellipses
    num_regions = random.randint(2, 6)
    
    for _ in range(num_regions):
        # Random center
        cx = random.randint(size // 4, 3 * size // 4)
        cy = random.randint(size // 4, 3 * size // 4)
        
        # Random radii
        rx = random.randint(size // 8, size // 3)
        ry = random.randint(size // 8, size // 3)
        
        # Create ellipse
        y, x = np.ogrid[:size, :size]
        ellipse = ((x - cx) ** 2 / (rx ** 2 + 1e-6) + (y - cy) ** 2 / (ry ** 2 + 1e-6)) <= 1
        
        mask[ellipse] = 1.0
    
    # Add some noise for realism
    noise = np.random.rand(size, size) * 0.1
    mask = np.clip(mask + noise * mask, 0, 1)
    
    # Smooth edges
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=3)
    mask = (mask > 0.3).astype(np.float32)
    
    return mask


def create_satellite_image(mask: np.ndarray) -> np.ndarray:
    """Create a synthetic satellite image based on flood mask."""
    size = mask.shape[0]
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Land colors (green/brown tones)
    land_r = np.random.randint(80, 120, (size, size)).astype(np.uint8)
    land_g = np.random.randint(100, 150, (size, size)).astype(np.uint8)
    land_b = np.random.randint(60, 100, (size, size)).astype(np.uint8)
    
    # Water colors (blue/cyan tones)
    water_r = np.random.randint(30, 80, (size, size)).astype(np.uint8)
    water_g = np.random.randint(80, 130, (size, size)).astype(np.uint8)
    water_b = np.random.randint(120, 180, (size, size)).astype(np.uint8)
    
    # Combine based on mask
    mask_3d = np.expand_dims(mask, axis=2)
    
    image[:, :, 0] = (land_r * (1 - mask) + water_r * mask).astype(np.uint8)
    image[:, :, 1] = (land_g * (1 - mask) + water_g * mask).astype(np.uint8)
    image[:, :, 2] = (land_b * (1 - mask) + water_b * mask).astype(np.uint8)
    
    # Add some urban structures (gray patches) on land
    num_structures = random.randint(5, 15)
    for _ in range(num_structures):
        x = random.randint(0, size - 20)
        y = random.randint(0, size - 20)
        w = random.randint(5, 20)
        h = random.randint(5, 20)
        
        if mask[y:y+h, x:x+w].mean() < 0.3:  # Only on land
            gray = random.randint(150, 200)
            image[y:y+h, x:x+w, 0] = gray
            image[y:y+h, x:x+w, 1] = gray
            image[y:y+h, x:x+w, 2] = gray
    
    return image


def generate_dataset(split: str, num_images: int, output_dir: Path):
    """Generate a dataset split."""
    images_dir = output_dir / split / "images"
    masks_dir = output_dir / split / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} {split} images...")
    
    for i in range(num_images):
        # Generate mask and image
        mask = create_flood_pattern(IMAGE_SIZE)
        image = create_satellite_image(mask)
        
        # Save image
        img_pil = Image.fromarray(image)
        img_pil.save(images_dir / f"{split}_{i:04d}.png")
        
        # Save mask (as grayscale)
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        mask_pil.save(masks_dir / f"{split}_{i:04d}_mask.png")
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_images}")
    
    print(f"  Saved to {output_dir / split}")


def main():
    """Generate all datasets."""
    print("=" * 60)
    print("Generating Synthetic Satellite Imagery")
    print("=" * 60)
    
    # Try to import scipy, install if not available
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Installing scipy...")
        os.system("pip3 install scipy --quiet")
        from scipy.ndimage import gaussian_filter
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate splits
    generate_dataset("train", NUM_TRAIN_IMAGES, OUTPUT_DIR)
    generate_dataset("val", NUM_VAL_IMAGES, OUTPUT_DIR)
    generate_dataset("test", NUM_TEST_IMAGES, OUTPUT_DIR)
    
    # Create metadata
    metadata = {
        "image_size": IMAGE_SIZE,
        "num_train": NUM_TRAIN_IMAGES,
        "num_val": NUM_VAL_IMAGES,
        "num_test": NUM_TEST_IMAGES,
        "classes": ["background", "flood"],
        "format": "PNG"
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Dataset generated: {NUM_TRAIN_IMAGES + NUM_VAL_IMAGES + NUM_TEST_IMAGES} total images")
    print(f"Location: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
