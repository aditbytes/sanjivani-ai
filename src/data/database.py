"""
Sanjivani AI - Database Connection Management

This module provides database connection management using SQLAlchemy.
Supports both synchronous and asynchronous database operations.
"""

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# =============================================================================
# Engine Configuration
# =============================================================================

def create_db_engine(
    database_url: Optional[str] = None,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    """
    Create a SQLAlchemy database engine.
    
    Args:
        database_url: Database connection URL. Defaults to settings value.
        echo: If True, log all SQL statements.
        pool_size: Base pool size for connection pooling.
        max_overflow: Maximum overflow connections beyond pool_size.
        
    Returns:
        SQLAlchemy Engine instance
        
    Example:
        >>> engine = create_db_engine()
        >>> with engine.connect() as conn:
        ...     result = conn.execute(text("SELECT 1"))
    """
    url = database_url or settings.database_url
    
    engine = create_engine(
        url,
        echo=echo or settings.debug,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # Verify connections are alive before using
        pool_recycle=3600,  # Recycle connections after 1 hour
    )
    
    # Add connection event listeners for debugging
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Database connection checked out from pool")
    
    logger.info(f"Database engine created for: {_mask_password(url)}")
    return engine


def _mask_password(url: str) -> str:
    """Mask password in database URL for logging."""
    import re
    return re.sub(r":([^:@]+)@", ":****@", url)


# =============================================================================
# Session Management
# =============================================================================

# Global engine instance (lazy initialization)
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """
    Get or create the global database engine.
    
    Returns:
        SQLAlchemy Engine instance
    """
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    """
    Get or create the session factory.
    
    Returns:
        SQLAlchemy sessionmaker instance
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    
    Use as a FastAPI dependency for request-scoped database sessions.
    The session is automatically closed after the request.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        >>> @app.get("/items")
        ... def get_items(db: Session = Depends(get_db)):
        ...     return db.query(Item).all()
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Use for scripts and background tasks that need database access.
    Automatically handles commit/rollback and session cleanup.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        >>> with get_db_session() as db:
        ...     alert = Alert(raw_text="Help needed!")
        ...     db.add(alert)
        ...     db.commit()
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# =============================================================================
# Async Support
# =============================================================================

try:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False
    AsyncSession = None

_async_engine = None
_AsyncSessionLocal = None


def get_async_engine():
    """
    Get or create the async database engine.
    
    Returns None if async SQLAlchemy is not available.
    
    Returns:
        Async SQLAlchemy Engine instance or None
    """
    global _async_engine
    
    if not HAS_ASYNC:
        logger.warning("Async SQLAlchemy not available")
        return None
    
    if _async_engine is None:
        # Convert sync URL to async URL
        url = settings.database_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        _async_engine = create_async_engine(
            url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
        )
    return _async_engine


async def get_async_db():
    """
    Async dependency that provides a database session.
    
    Use as a FastAPI dependency for async request handlers.
    
    Yields:
        Async SQLAlchemy Session instance
    """
    global _AsyncSessionLocal
    
    if not HAS_ASYNC:
        raise RuntimeError("Async SQLAlchemy not available")
    
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    
    async with _AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """
    High-level database management class.
    
    Provides methods for database initialization, health checks,
    and table management.
    
    Example:
        >>> manager = DatabaseManager()
        >>> manager.init_db()
        >>> assert manager.check_health()
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Optional custom database URL
        """
        self.database_url = database_url or settings.database_url
        self._engine: Optional[Engine] = None
    
    @property
    def engine(self) -> Engine:
        """Get the database engine, creating it if necessary."""
        if self._engine is None:
            self._engine = create_db_engine(self.database_url)
        return self._engine
    
    def init_db(self) -> None:
        """
        Initialize database by creating all tables.
        
        Creates tables defined in models.py if they don't exist.
        """
        from src.data.models import Base
        
        logger.info("Initializing database tables...")
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")
    
    def drop_all(self) -> None:
        """
        Drop all database tables.
        
        WARNING: This will delete all data! Use with caution.
        """
        from src.data.models import Base
        
        logger.warning("Dropping all database tables!")
        Base.metadata.drop_all(self.engine)
        logger.info("All tables dropped")
    
    def check_health(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("Database health check passed")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Remember to close the session when done.
        
        Returns:
            SQLAlchemy Session instance
        """
        SessionLocal = sessionmaker(bind=self.engine)
        return SessionLocal()
    
    def get_table_counts(self) -> dict:
        """
        Get row counts for all tables.
        
        Useful for monitoring and debugging.
        
        Returns:
            Dictionary mapping table names to row counts
        """
        from src.data.models import Base
        from sqlalchemy import text
        
        counts = {}
        with self.engine.connect() as conn:
            for table_name in Base.metadata.tables.keys():
                try:
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    )
                    counts[table_name] = result.scalar()
                except Exception as e:
                    logger.warning(f"Could not count table {table_name}: {e}")
                    counts[table_name] = -1
        
        return counts
    
    def close(self) -> None:
        """Dispose of the database engine and connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connections closed")


# =============================================================================
# Convenience Functions
# =============================================================================

def init_database() -> None:
    """
    Initialize the database with all tables.
    
    Call this at application startup or from init script.
    """
    manager = DatabaseManager()
    manager.init_db()


def check_database_health() -> bool:
    """
    Quick database health check.
    
    Returns:
        True if database is healthy, False otherwise
    """
    manager = DatabaseManager()
    return manager.check_health()
