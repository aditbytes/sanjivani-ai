"""
Sanjivani AI - Common Helper Utilities

This module provides common utility functions used across the application.
All functions include type hints and comprehensive docstrings.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


# =============================================================================
# Text Utilities
# =============================================================================

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding suffix if truncated.
    
    Args:
        text: Input text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if text is truncated
        
    Returns:
        Truncated text with suffix if original exceeded max_length
        
    Example:
        >>> truncate_text("Hello World", max_length=8)
        'Hello...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text by collapsing multiple spaces.
    
    Args:
        text: Input text with potential irregular whitespace
        
    Returns:
        Text with normalized whitespace
        
    Example:
        >>> normalize_whitespace("Hello   World")
        'Hello World'
    """
    return " ".join(text.split())


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    
    Args:
        text: Input text potentially containing URLs
        
    Returns:
        Text with URLs removed
        
    Example:
        >>> remove_urls("Check this https://example.com now")
        'Check this  now'
    """
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def remove_mentions(text: str) -> str:
    """
    Remove @mentions from text (Twitter/social media style).
    
    Args:
        text: Input text potentially containing @mentions
        
    Returns:
        Text with mentions removed
        
    Example:
        >>> remove_mentions("Hello @user please help")
        'Hello  please help'
    """
    return re.sub(r"@\w+", "", text)


# =============================================================================
# JSON Utilities
# =============================================================================

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string, returning default on failure.
    
    Args:
        json_string: JSON string to parse
        default: Value to return if parsing fails
        
    Returns:
        Parsed JSON object or default value
        
    Example:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_loads('invalid', default={})
        {}
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON parsing failed: {e}")
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON string, returning default on failure.
    
    Args:
        obj: Object to serialize
        default: String to return if serialization fails
        
    Returns:
        JSON string or default value
        
    Example:
        >>> safe_json_dumps({"key": "value"})
        '{"key": "value"}'
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed: {e}")
        return default


# =============================================================================
# Date/Time Utilities
# =============================================================================

def format_timestamp(
    dt: Optional[datetime] = None,
    format_str: str = TIMESTAMP_FORMAT
) -> str:
    """
    Format datetime to string, using current time if not provided.
    
    Args:
        dt: Datetime to format. If None, uses current UTC time.
        format_str: strftime format string
        
    Returns:
        Formatted datetime string
        
    Example:
        >>> format_timestamp(datetime(2024, 1, 15, 10, 30, 0))
        '2024-01-15 10:30:00'
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime(format_str)


def parse_timestamp(
    timestamp_str: str,
    format_str: str = TIMESTAMP_FORMAT
) -> Optional[datetime]:
    """
    Parse timestamp string to datetime.
    
    Args:
        timestamp_str: Timestamp string to parse
        format_str: Expected strftime format
        
    Returns:
        Parsed datetime or None if parsing fails
        
    Example:
        >>> parse_timestamp('2024-01-15 10:30:00')
        datetime.datetime(2024, 1, 15, 10, 30)
    """
    try:
        return datetime.strptime(timestamp_str, format_str)
    except ValueError as e:
        logger.warning(f"Timestamp parsing failed for '{timestamp_str}': {e}")
        return None


def get_utc_now() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        Current UTC datetime with timezone info
    """
    return datetime.now(timezone.utc)


# =============================================================================
# File Utilities
# =============================================================================

def get_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> Optional[str]:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        chunk_size: Size of chunks to read at a time
        
    Returns:
        Hex digest of file hash, or None if file doesn't exist
        
    Example:
        >>> get_file_hash("models/nlp/model.pth")
        'a1b2c3d4e5f6...'
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found for hashing: {file_path}")
        return None
    
    try:
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return None


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object of the directory
        
    Example:
        >>> ensure_dir("data/processed")
        PosixPath('data/processed')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data or None if error occurs
        
    Example:
        >>> data = read_json_file("config.json")
    """
    file_path = Path(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return None


def write_json_file(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2
) -> bool:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to write JSON file
        data: Data to serialize
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> write_json_file("output.json", {"key": "value"})
        True
    """
    file_path = Path(file_path)
    try:
        ensure_dir(file_path.parent)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        return False


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunks
        
    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list one level deep.
    
    Args:
        nested_list: List of lists to flatten
        
    Returns:
        Flattened list
        
    Example:
        >>> flatten_list([[1, 2], [3, 4]])
        [1, 2, 3, 4]
    """
    return [item for sublist in nested_list for item in sublist]


# =============================================================================
# Bihar-specific Utilities
# =============================================================================

# Bihar district coordinates (approximate centers)
BIHAR_DISTRICTS: Dict[str, Dict[str, float]] = {
    "Araria": {"lat": 26.1333, "lon": 87.4500},
    "Arwal": {"lat": 25.2500, "lon": 84.6833},
    "Aurangabad": {"lat": 24.7500, "lon": 84.3667},
    "Banka": {"lat": 24.8833, "lon": 86.9167},
    "Begusarai": {"lat": 25.4167, "lon": 86.1333},
    "Bhagalpur": {"lat": 25.2500, "lon": 87.0000},
    "Bhojpur": {"lat": 25.5000, "lon": 84.5000},
    "Buxar": {"lat": 25.5667, "lon": 83.9833},
    "Darbhanga": {"lat": 26.1667, "lon": 85.9000},
    "East Champaran": {"lat": 26.6500, "lon": 84.8500},
    "Gaya": {"lat": 24.7500, "lon": 84.9333},
    "Gopalganj": {"lat": 26.4667, "lon": 84.4333},
    "Jamui": {"lat": 24.9167, "lon": 86.2167},
    "Jehanabad": {"lat": 25.2167, "lon": 84.9833},
    "Kaimur": {"lat": 25.0333, "lon": 83.6000},
    "Katihar": {"lat": 25.5333, "lon": 87.5667},
    "Khagaria": {"lat": 25.5000, "lon": 86.4667},
    "Kishanganj": {"lat": 26.1000, "lon": 87.9500},
    "Lakhisarai": {"lat": 25.1500, "lon": 86.0833},
    "Madhepura": {"lat": 25.9167, "lon": 86.7833},
    "Madhubani": {"lat": 26.3500, "lon": 86.0667},
    "Munger": {"lat": 25.3833, "lon": 86.4833},
    "Muzaffarpur": {"lat": 26.1167, "lon": 85.4000},
    "Nalanda": {"lat": 25.1333, "lon": 85.4500},
    "Nawada": {"lat": 24.8833, "lon": 85.5333},
    "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Purnia": {"lat": 25.7833, "lon": 87.4833},
    "Rohtas": {"lat": 24.9667, "lon": 84.0167},
    "Saharsa": {"lat": 25.8833, "lon": 86.6000},
    "Samastipur": {"lat": 25.8500, "lon": 85.7833},
    "Saran": {"lat": 25.9167, "lon": 84.7500},
    "Sheikhpura": {"lat": 25.1333, "lon": 85.8500},
    "Sheohar": {"lat": 26.5167, "lon": 85.3000},
    "Sitamarhi": {"lat": 26.6000, "lon": 85.4833},
    "Siwan": {"lat": 26.2167, "lon": 84.3500},
    "Supaul": {"lat": 26.1167, "lon": 86.6000},
    "Vaishali": {"lat": 25.6833, "lon": 85.2167},
    "West Champaran": {"lat": 26.7333, "lon": 84.4333},
}


def get_district_coordinates(district_name: str) -> Optional[Dict[str, float]]:
    """
    Get coordinates for a Bihar district.
    
    Args:
        district_name: Name of the district (case-insensitive)
        
    Returns:
        Dictionary with 'lat' and 'lon' keys, or None if not found
        
    Example:
        >>> get_district_coordinates("Patna")
        {'lat': 25.5941, 'lon': 85.1376}
    """
    # Try exact match first
    if district_name in BIHAR_DISTRICTS:
        return BIHAR_DISTRICTS[district_name]
    
    # Try case-insensitive match
    district_lower = district_name.lower()
    for name, coords in BIHAR_DISTRICTS.items():
        if name.lower() == district_lower:
            return coords
    
    logger.warning(f"District not found: {district_name}")
    return None


def get_all_district_names() -> List[str]:
    """
    Get list of all Bihar district names.
    
    Returns:
        List of district names
    """
    return list(BIHAR_DISTRICTS.keys())
