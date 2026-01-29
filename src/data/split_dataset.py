"""
Sanjivani AI - Dataset Split Utility

Split datasets into train, validation, and test sets with stratification.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import get_settings
from src.data.loaders import load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def stratified_split(
    data: List[Dict],
    label_key: str = "urgency",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data with stratification on label_key.
    
    Args:
        data: List of data dictionaries
        label_key: Key to stratify on
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1.0)
        seed: Random seed
        
    Returns:
        Tuple of (train, val, test) lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    
    random.seed(seed)
    
    # Group by label
    groups: Dict[str, List[Dict]] = {}
    for item in data:
        label = item.get(label_key, "unknown")
        groups.setdefault(label, []).append(item)
    
    train, val, test = [], [], []
    
    for label, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def split_and_save(
    input_path: Path,
    output_dir: Path,
    label_key: str = "urgency",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> bool:
    """Load data, split it, and save to train/val/test files."""
    data = load_json(input_path)
    if not data:
        logger.error(f"Failed to load data from {input_path}")
        return False
    
    train, val, test = stratified_split(data, label_key, train_ratio, val_ratio, test_ratio)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(train, output_dir / "train.json")
    save_json(val, output_dir / "val.json")
    save_json(test, output_dir / "test.json")
    
    return True


if __name__ == "__main__":
    split_and_save(
        settings.data_dir / "raw" / "tweets.json",
        settings.data_dir / "processed",
    )
