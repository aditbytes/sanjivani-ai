"""
Sanjivani AI - Configuration Management

This module provides centralized configuration using Pydantic Settings.
All configuration is loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    Use get_settings() to access the cached singleton instance.
    
    Example:
        >>> settings = get_settings()
        >>> print(settings.app_name)
        'sanjivani-ai'
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    app_name: str = Field(default="sanjivani-ai", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # -------------------------------------------------------------------------
    # Database Configuration
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql://sanjivani:password@localhost:5432/sanjivani_db",
        description="PostgreSQL connection URL"
    )
    postgres_user: str = Field(default="sanjivani", description="PostgreSQL username")
    postgres_password: str = Field(default="password", description="PostgreSQL password")
    postgres_db: str = Field(default="sanjivani_db", description="PostgreSQL database name")
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    
    # -------------------------------------------------------------------------
    # Redis Configuration
    # -------------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    
    # -------------------------------------------------------------------------
    # API Configuration
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=4, description="Number of API workers")
    
    # JWT Authentication
    jwt_secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="JWT secret key for token signing"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        description="JWT token expiration time in minutes"
    )
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="API rate limit per minute"
    )
    
    # -------------------------------------------------------------------------
    # Dashboard Configuration
    # -------------------------------------------------------------------------
    streamlit_server_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_server_address: str = Field(
        default="0.0.0.0",
        description="Streamlit server address"
    )
    
    # -------------------------------------------------------------------------
    # Twitter/X API Configuration
    # -------------------------------------------------------------------------
    twitter_bearer_token: Optional[str] = Field(
        default=None,
        description="Twitter API Bearer Token"
    )
    twitter_api_key: Optional[str] = Field(default=None, description="Twitter API Key")
    twitter_api_secret: Optional[str] = Field(default=None, description="Twitter API Secret")
    twitter_access_token: Optional[str] = Field(
        default=None,
        description="Twitter Access Token"
    )
    twitter_access_token_secret: Optional[str] = Field(
        default=None,
        description="Twitter Access Token Secret"
    )
    
    # -------------------------------------------------------------------------
    # Satellite Imagery API
    # -------------------------------------------------------------------------
    sentinel_hub_client_id: Optional[str] = Field(
        default=None,
        description="Sentinel Hub Client ID"
    )
    sentinel_hub_client_secret: Optional[str] = Field(
        default=None,
        description="Sentinel Hub Client Secret"
    )
    copernicus_username: Optional[str] = Field(
        default=None,
        description="Copernicus Data Space username"
    )
    copernicus_password: Optional[str] = Field(
        default=None,
        description="Copernicus Data Space password"
    )
    
    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    nlp_model_path: Path = Field(
        default=Path("models/nlp/best_model.pth"),
        description="Path to NLP model weights"
    )
    vision_segmentation_model_path: Path = Field(
        default=Path("models/vision/unet_segmentation.pth"),
        description="Path to segmentation model weights"
    )
    vision_detection_model_path: Path = Field(
        default=Path("models/vision/yolov8_detection.pt"),
        description="Path to detection model weights"
    )
    forecasting_xgboost_path: Path = Field(
        default=Path("models/forecasting/xgboost_model.pkl"),
        description="Path to XGBoost model"
    )
    forecasting_lstm_path: Path = Field(
        default=Path("models/forecasting/lstm_model.h5"),
        description="Path to LSTM model"
    )
    
    # Model Inference Settings
    max_tweet_length: int = Field(default=128, description="Maximum tweet token length")
    nlp_batch_size: int = Field(default=16, description="NLP inference batch size")
    vision_image_size: int = Field(default=512, description="Vision model input size")
    inference_device: str = Field(
        default="cpu",
        description="Inference device (cpu or cuda)"
    )
    
    # -------------------------------------------------------------------------
    # Bihar Geographic Settings
    # -------------------------------------------------------------------------
    bihar_center_lat: float = Field(
        default=25.5941,
        description="Bihar center latitude"
    )
    bihar_center_lon: float = Field(
        default=85.1376,
        description="Bihar center longitude"
    )
    bihar_default_zoom: int = Field(default=7, description="Default map zoom level")
    
    # -------------------------------------------------------------------------
    # Monitoring and Alerting
    # -------------------------------------------------------------------------
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    slack_webhook_url: Optional[str] = Field(
        default=None,
        description="Slack webhook for alerts"
    )
    alert_email_recipients: Optional[str] = Field(
        default=None,
        description="Comma-separated email recipients for alerts"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper
    
    @field_validator("inference_device")
    @classmethod
    def validate_inference_device(cls, v: str) -> str:
        """Validate inference device is valid."""
        valid_devices = {"cpu", "cuda", "mps"}
        v_lower = v.lower()
        if v_lower not in valid_devices:
            raise ValueError(f"inference_device must be one of {valid_devices}")
        return v_lower
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self.project_root / "data"
    
    @property
    def models_dir(self) -> Path:
        """Get the models directory path."""
        return self.project_root / "models"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings: Application settings instance
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.database_url)
    """
    return Settings()


# Convenience alias
settings = get_settings()
