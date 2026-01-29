#!/usr/bin/env python
"""
Sanjivani AI - Database Initialization

Initialize database schema and seed data.
"""

import asyncio
from pathlib import Path

from src.config import get_settings
from src.data.database import get_engine, Base
from src.data.models import Alert, Prediction, Resource, FloodEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def init_database():
    """Initialize database tables."""
    engine = get_engine()
    
    if engine is None:
        logger.error("Database connection not available")
        return
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")


def main():
    """Run database initialization."""
    logger.info("Initializing Sanjivani AI database...")
    init_database()
    logger.info("Database initialization complete")


if __name__ == "__main__":
    main()
