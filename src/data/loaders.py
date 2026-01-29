"""
Sanjivani AI - Data Loaders

This module provides utilities for loading and saving data in various formats.
Supports JSON, CSV, and Parquet files with proper error handling.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# =============================================================================
# JSON Loaders
# =============================================================================

def load_json(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data or None if loading fails
        
    Example:
        >>> data = load_json("data/processed/train.json")
        >>> len(data)
        200
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"JSON file not found: {file_path}")
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def save_json(
    data: Union[Dict, List],
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (dict or list)
        file_path: Path to save the JSON file
        indent: Indentation level for formatting
        ensure_ascii: If False, allow non-ASCII characters
        
    Returns:
        True if save was successful, False otherwise
        
    Example:
        >>> save_json({"key": "value"}, "output.json")
        True
    """
    file_path = Path(file_path)
    
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        logger.info(f"Saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load data from a JSON Lines file (one JSON object per line).
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of parsed JSON objects
        
    Example:
        >>> data = load_jsonl("data/tweets.jsonl")
        >>> len(data)
        1000
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"JSONL file not found: {file_path}")
        return []
    
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
        
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSONL from {file_path}: {e}")
        return []


def save_jsonl(
    data: List[Dict],
    file_path: Union[str, Path],
) -> bool:
    """
    Save data to a JSON Lines file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSONL file
        
    Returns:
        True if save was successful, False otherwise
    """
    file_path = Path(file_path)
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        
        logger.info(f"Saved {len(data)} records to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSONL to {file_path}: {e}")
        return False


# =============================================================================
# CSV Loaders
# =============================================================================

def load_csv(
    file_path: Union[str, Path],
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> List[Dict]:
    """
    Load data from a CSV file as list of dictionaries.
    
    Args:
        file_path: Path to the CSV file
        delimiter: Field delimiter
        encoding: File encoding
        
    Returns:
        List of dictionaries (one per row)
        
    Example:
        >>> data = load_csv("data/alerts.csv")
        >>> data[0]["urgency"]
        'Critical'
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"CSV file not found: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            data = list(reader)
        
        logger.info(f"Loaded {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading CSV from {file_path}: {e}")
        return []


def save_csv(
    data: List[Dict],
    file_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ",",
) -> bool:
    """
    Save list of dictionaries to a CSV file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the CSV file
        fieldnames: Column names (inferred from first row if None)
        delimiter: Field delimiter
        
    Returns:
        True if save was successful, False otherwise
    """
    file_path = Path(file_path)
    
    if not data:
        logger.warning("No data to save to CSV")
        return False
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Saved {len(data)} rows to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving CSV to {file_path}: {e}")
        return False


# =============================================================================
# Pandas DataFrame Loaders
# =============================================================================

def load_dataframe(
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Load data into a pandas DataFrame.
    
    Automatically detects file type from extension if not specified.
    Supports CSV, JSON, Parquet, and Excel files.
    
    Args:
        file_path: Path to the data file
        file_type: File type ('csv', 'json', 'parquet', 'excel'). Auto-detected if None.
        **kwargs: Additional arguments passed to pandas read function
        
    Returns:
        DataFrame or None if loading fails
        
    Example:
        >>> df = load_dataframe("data/alerts.csv")
        >>> df.shape
        (1000, 10)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    # Auto-detect file type
    if file_type is None:
        suffix = file_path.suffix.lower()
        type_map = {
            ".csv": "csv",
            ".json": "json",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".xlsx": "excel",
            ".xls": "excel",
        }
        file_type = type_map.get(suffix, "csv")
    
    try:
        if file_type == "csv":
            df = pd.read_csv(file_path, **kwargs)
        elif file_type == "json":
            df = pd.read_json(file_path, **kwargs)
        elif file_type == "parquet":
            df = pd.read_parquet(file_path, **kwargs)
        elif file_type == "excel":
            df = pd.read_excel(file_path, **kwargs)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return None
        
        logger.info(f"Loaded DataFrame from {file_path}: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {file_path}: {e}")
        return None


def save_dataframe(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    Save a pandas DataFrame to file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        file_type: File type ('csv', 'json', 'parquet'). Auto-detected if None.
        **kwargs: Additional arguments passed to pandas to_* function
        
    Returns:
        True if save was successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Auto-detect file type
    if file_type is None:
        suffix = file_path.suffix.lower()
        type_map = {
            ".csv": "csv",
            ".json": "json",
            ".parquet": "parquet",
            ".pq": "parquet",
        }
        file_type = type_map.get(suffix, "csv")
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_type == "csv":
            df.to_csv(file_path, index=False, **kwargs)
        elif file_type == "json":
            df.to_json(file_path, orient="records", **kwargs)
        elif file_type == "parquet":
            df.to_parquet(file_path, index=False, **kwargs)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return False
        
        logger.info(f"Saved DataFrame to {file_path}: {df.shape}")
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {e}")
        return False


# =============================================================================
# Crisis Data Loaders
# =============================================================================

def load_tweets_dataset(split: str = "train") -> List[Dict]:
    """
    Load the crisis tweets dataset for NLP training.
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        List of tweet dictionaries
        
    Example:
        >>> train_data = load_tweets_dataset("train")
        >>> len(train_data)
        140
    """
    file_path = settings.data_dir / "processed" / f"{split}.json"
    return load_json(file_path) or []


def load_satellite_dataset(split: str = "train") -> List[Dict]:
    """
    Load the satellite imagery dataset metadata.
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        List of image metadata dictionaries
    """
    file_path = settings.data_dir / "processed" / f"satellite_{split}.json"
    return load_json(file_path) or []


def load_historical_floods() -> List[Dict]:
    """
    Load historical flood event data for forecasting.
    
    Returns:
        List of historical flood event records
    """
    file_path = settings.data_dir / "processed" / "historical_floods.json"
    return load_json(file_path) or []


# =============================================================================
# Data Validation
# =============================================================================

def validate_tweet_data(data: List[Dict]) -> List[Dict]:
    """
    Validate and clean tweet data.
    
    Ensures required fields are present and data is properly formatted.
    
    Args:
        data: List of tweet dictionaries
        
    Returns:
        List of valid tweet dictionaries
    """
    required_fields = ["text", "urgency"]
    valid_urgencies = {"Critical", "High", "Medium", "Low", "Non-Urgent"}
    
    valid_data = []
    for i, item in enumerate(data):
        # Check required fields
        if not all(field in item for field in required_fields):
            logger.warning(f"Missing required fields in item {i}")
            continue
        
        # Validate urgency
        if item.get("urgency") not in valid_urgencies:
            logger.warning(f"Invalid urgency '{item.get('urgency')}' in item {i}")
            continue
        
        # Ensure text is not empty
        if not item.get("text", "").strip():
            logger.warning(f"Empty text in item {i}")
            continue
        
        valid_data.append(item)
    
    logger.info(f"Validated {len(valid_data)}/{len(data)} tweets")
    return valid_data
