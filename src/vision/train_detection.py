"""
Sanjivani AI - Detection Training

Training script for YOLOv8 flood object detection.
"""

from pathlib import Path
from typing import Dict, Optional

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def train_detector(
    data_yaml_path: Path,
    output_dir: Path = None,
    epochs: int = 100,
    batch_size: int = 16,
    model_size: str = "n",  # n, s, m, l, x
    device: str = None,
) -> Dict:
    """
    Fine-tune YOLOv8 for flood object detection.
    
    Args:
        data_yaml_path: Path to YOLO data.yaml file
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        model_size: YOLOv8 model size (n/s/m/l/x)
        device: Training device
    """
    output_dir = output_dir or settings.models_dir / "vision"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from ultralytics import YOLO
        
        # Load pretrained model
        model = YOLO(f"yolov8{model_size}.pt")
        
        # Train
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(output_dir),
            name="yolov8_flood",
            device=device or "cpu",
        )
        
        logger.info(f"Training complete. Model saved to {output_dir}")
        return {"results": str(results)}
        
    except ImportError:
        logger.error("ultralytics not installed")
        return {"error": "ultralytics not installed"}


def create_yolo_dataset_yaml(
    train_dir: Path,
    val_dir: Path,
    output_path: Path,
    classes: list = None,
) -> Path:
    """Create YOLO dataset.yaml file."""
    import yaml
    
    classes = classes or ["person", "vehicle", "boat", "building_damaged", "animal"]
    
    config = {
        "path": str(train_dir.parent),
        "train": str(train_dir.name),
        "val": str(val_dir.name),
        "names": {i: name for i, name in enumerate(classes)},
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    
    return output_path


if __name__ == "__main__":
    logger.info("Prepare YOLO format dataset before training")
