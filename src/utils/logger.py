"""
Sanjivani AI - Logging Configuration

This module provides structured logging setup using loguru.
Supports console output, file logging, and structured JSON logging.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure application-wide logging.
    
    Sets up loguru with console output and optional file logging.
    Should be called once at application startup.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging is used.
        json_format: If True, use JSON format for file logs (useful for log aggregation)
        rotation: Log file rotation policy (e.g., "10 MB", "1 day")
        retention: Log file retention policy (e.g., "1 week", "10 files")
        
    Example:
        >>> setup_logging(log_level="DEBUG", log_file=Path("logs/app.log"))
        >>> logger.info("Application started")
    """
    # Remove default handler
    logger.remove()
    
    # Console format with colors
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if json_format:
            # JSON format for log aggregation systems
            logger.add(
                str(log_file),
                format="{message}",
                level=log_level,
                rotation=rotation,
                retention=retention,
                serialize=True,  # JSON serialization
                compression="gz",
            )
        else:
            # Plain text format for human reading
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
            logger.add(
                str(log_file),
                format=file_format,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="gz",
            )
    
    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> "logger":
    """
    Get a logger instance bound with a specific name/context.
    
    Creates a logger with the given name bound as context,
    useful for tracking which module logged a message.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        A loguru logger instance bound with the name context
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logger.bind(name=name)


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    Provides a self.logger attribute automatically bound with the class name.
    
    Example:
        >>> class MyService(LoggerMixin):
        ...     def process(self):
        ...         self.logger.info("Processing...")
    """
    
    @property
    def logger(self) -> "logger":
        """Get logger bound with class name."""
        return logger.bind(name=self.__class__.__name__)


# Export the base logger for direct use
__all__ = ["logger", "setup_logging", "get_logger", "LoggerMixin"]
