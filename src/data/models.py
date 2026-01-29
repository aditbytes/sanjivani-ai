"""
Sanjivani AI - SQLAlchemy Database Models

This module defines the database schema for the crisis intelligence system.
Uses SQLAlchemy ORM with PostGIS for spatial data support.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

# Try to import GeoAlchemy2 for spatial support
try:
    from geoalchemy2 import Geometry
    HAS_GEOALCHEMY = True
except ImportError:
    HAS_GEOALCHEMY = False
    Geometry = None


Base = declarative_base()


# =============================================================================
# Enums
# =============================================================================

class UrgencyLevel(str, PyEnum):
    """Urgency classification levels for crisis alerts."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NON_URGENT = "Non-Urgent"


class ResourceType(str, PyEnum):
    """Types of resources needed during crisis."""
    RESCUE = "Rescue"
    MEDICAL = "Medical"
    FOOD = "Food"
    SHELTER = "Shelter"
    INFORMATION = "Information"
    WATER = "Water"
    CLOTHING = "Clothing"


class VulnerabilityType(str, PyEnum):
    """Vulnerability categories for affected population."""
    ELDERLY = "Elderly"
    CHILDREN = "Children"
    DISABLED = "Disabled"
    PREGNANT = "Pregnant"
    NONE = "None"


class AlertStatus(str, PyEnum):
    """Status of crisis alerts."""
    ACTIVE = "Active"
    RESPONDING = "Responding"
    RESOLVED = "Resolved"
    DUPLICATE = "Duplicate"
    FALSE_ALARM = "False Alarm"


# =============================================================================
# Models
# =============================================================================

class Alert(Base):
    """
    Crisis alert from social media or other sources.
    
    Stores processed tweets and other distress signals with their
    classifications, extracted locations, and response status.
    """
    __tablename__ = "alerts"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier for the alert"
    )
    
    # Source information
    source_id = Column(
        String(255),
        index=True,
        doc="Original ID from source platform (e.g., tweet ID)"
    )
    source_type = Column(
        String(50),
        default="twitter",
        doc="Source platform (twitter, whatsapp, call, etc.)"
    )
    source_url = Column(String(512), nullable=True, doc="Link to original content")
    
    # Content
    raw_text = Column(Text, nullable=False, doc="Original text content")
    processed_text = Column(Text, doc="Cleaned/processed text")
    language = Column(String(10), default="en", doc="Detected language code")
    
    # NLP Classification
    urgency = Column(
        Enum(UrgencyLevel),
        default=UrgencyLevel.MEDIUM,
        index=True,
        doc="Urgency classification"
    )
    urgency_confidence = Column(
        Float,
        default=0.0,
        doc="Confidence score for urgency classification"
    )
    
    resource_needed = Column(
        Enum(ResourceType),
        nullable=True,
        doc="Primary resource type needed"
    )
    resources_all = Column(
        JSONB,
        default=list,
        doc="All resource types with probabilities"
    )
    
    vulnerability = Column(
        Enum(VulnerabilityType),
        default=VulnerabilityType.NONE,
        doc="Vulnerability category"
    )
    
    # Location
    district = Column(String(100), index=True, doc="Extracted district name")
    location_raw = Column(String(255), doc="Raw location mention from text")
    latitude = Column(Float, nullable=True, doc="Latitude coordinate")
    longitude = Column(Float, nullable=True, doc="Longitude coordinate")
    
    # Status
    status = Column(
        Enum(AlertStatus),
        default=AlertStatus.ACTIVE,
        index=True,
        doc="Current alert status"
    )
    
    # Timestamps
    source_timestamp = Column(
        DateTime(timezone=True),
        doc="When the original content was posted"
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="When alert was created in system"
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        doc="Last update timestamp"
    )
    resolved_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="When alert was resolved"
    )
    
    # Response tracking
    responder_id = Column(String(255), nullable=True, doc="ID of assigned responder")
    response_notes = Column(Text, nullable=True, doc="Notes from responders")
    
    # Relationships
    predictions = relationship("Prediction", back_populates="alert")
    
    # Indices
    __table_args__ = (
        Index("idx_alerts_urgency_status", "urgency", "status"),
        Index("idx_alerts_district_created", "district", "created_at"),
        Index("idx_alerts_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, urgency={self.urgency}, district={self.district})>"
    
    def to_dict(self) -> dict:
        """Convert alert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "source_id": self.source_id,
            "source_type": self.source_type,
            "raw_text": self.raw_text,
            "processed_text": self.processed_text,
            "urgency": self.urgency.value if self.urgency else None,
            "urgency_confidence": self.urgency_confidence,
            "resource_needed": self.resource_needed.value if self.resource_needed else None,
            "vulnerability": self.vulnerability.value if self.vulnerability else None,
            "district": self.district,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Prediction(Base):
    """
    ML model predictions associated with alerts or forecasts.
    
    Stores detailed prediction results including class probabilities
    and model metadata for analysis and auditing.
    """
    __tablename__ = "predictions"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Link to alert
    alert_id = Column(
        UUID(as_uuid=True),
        ForeignKey("alerts.id"),
        nullable=True,
        index=True
    )
    
    # Model information
    model_name = Column(String(100), nullable=False, doc="Name of the model used")
    model_version = Column(String(50), doc="Version of the model")
    model_type = Column(
        String(50),
        doc="Type: nlp, vision_segmentation, vision_detection, forecast"
    )
    
    # Prediction details
    prediction_class = Column(String(100), doc="Predicted class/label")
    confidence_score = Column(Float, doc="Confidence score 0-1")
    all_probabilities = Column(JSONB, doc="Full probability distribution")
    
    # Performance tracking
    inference_time_ms = Column(Float, doc="Inference time in milliseconds")
    
    # Ground truth for evaluation
    ground_truth = Column(String(100), nullable=True, doc="Actual label if known")
    is_correct = Column(Boolean, nullable=True, doc="Whether prediction was correct")
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    # Relationships
    alert = relationship("Alert", back_populates="predictions")
    
    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, model={self.model_name}, class={self.prediction_class})>"


class Resource(Base):
    """
    Resource allocation and inventory tracking.
    
    Tracks available, allocated, and forecasted resource quantities
    at district level for disaster response planning.
    """
    __tablename__ = "resources"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Location
    district = Column(String(100), nullable=False, index=True)
    
    # Resource type and quantities
    resource_type = Column(
        Enum(ResourceType),
        nullable=False,
        index=True
    )
    quantity_available = Column(Integer, default=0, doc="Currently available units")
    quantity_allocated = Column(Integer, default=0, doc="Units allocated to response")
    quantity_needed = Column(Integer, default=0, doc="Estimated units needed")
    
    # Forecast
    forecast_24h = Column(Integer, default=0, doc="Predicted need in 24 hours")
    forecast_48h = Column(Integer, default=0, doc="Predicted need in 48 hours")
    forecast_72h = Column(Integer, default=0, doc="Predicted need in 72 hours")
    forecast_confidence = Column(Float, doc="Forecast confidence score")
    
    # Metadata
    unit = Column(String(50), default="units", doc="Unit of measurement")
    last_updated = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    __table_args__ = (
        Index("idx_resources_district_type", "district", "resource_type"),
    )
    
    def __repr__(self) -> str:
        return f"<Resource(district={self.district}, type={self.resource_type}, available={self.quantity_available})>"


class SatelliteImage(Base):
    """
    Satellite imagery metadata and analysis results.
    
    Stores references to satellite images and their CV analysis
    including flood extent, damage assessment, and object detections.
    """
    __tablename__ = "satellite_images"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Image identification
    image_path = Column(String(512), nullable=False, doc="Path to image file")
    image_hash = Column(String(64), doc="SHA256 hash for deduplication")
    
    # Source metadata
    source = Column(String(100), default="sentinel-2", doc="Satellite source")
    acquisition_date = Column(DateTime(timezone=True), doc="Image capture date")
    cloud_cover_pct = Column(Float, doc="Cloud cover percentage")
    
    # Geographic bounds
    district = Column(String(100), index=True)
    min_lat = Column(Float)
    max_lat = Column(Float)
    min_lon = Column(Float)
    max_lon = Column(Float)
    
    # Image properties
    width = Column(Integer, doc="Image width in pixels")
    height = Column(Integer, doc="Image height in pixels")
    resolution_m = Column(Float, doc="Ground resolution in meters")
    bands = Column(JSONB, default=["R", "G", "B"], doc="Spectral bands")
    
    # Analysis results
    is_analyzed = Column(Boolean, default=False)
    flood_percentage = Column(Float, doc="Percentage of area flooded")
    damage_level = Column(String(50), doc="Overall damage assessment")
    
    # Detection results
    objects_detected = Column(JSONB, doc="List of detected objects with bboxes")
    num_people_detected = Column(Integer, default=0)
    num_vehicles_detected = Column(Integer, default=0)
    num_buildings_damaged = Column(Integer, default=0)
    
    # Segmentation mask path
    mask_path = Column(String(512), doc="Path to flood segmentation mask")
    
    # Comparison with pre-flood
    pre_flood_image_id = Column(
        UUID(as_uuid=True),
        ForeignKey("satellite_images.id"),
        nullable=True
    )
    change_score = Column(Float, doc="Change detection score")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    analyzed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<SatelliteImage(id={self.id}, district={self.district}, analyzed={self.is_analyzed})>"


class FloodEvent(Base):
    """
    Historical flood event records for forecasting model training.
    
    Aggregates data about past flood events including weather conditions,
    affected areas, and resource requirements.
    """
    __tablename__ = "flood_events"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Event identification
    event_name = Column(String(255), doc="Descriptive event name")
    year = Column(Integer, index=True)
    start_date = Column(DateTime(timezone=True), index=True)
    end_date = Column(DateTime(timezone=True))
    
    # Affected area
    districts_affected = Column(JSONB, doc="List of affected districts")
    primary_district = Column(String(100), index=True)
    
    # Weather conditions
    rainfall_mm = Column(Float, doc="Total rainfall in mm")
    river_level_m = Column(Float, doc="Peak river level in meters")
    river_name = Column(String(100))
    
    # Impact
    population_affected = Column(Integer)
    casualties = Column(Integer, default=0)
    houses_damaged = Column(Integer, default=0)
    area_flooded_sqkm = Column(Float, doc="Flooded area in square km")
    
    # Resources distributed
    food_packets = Column(Integer, default=0)
    medical_kits = Column(Integer, default=0)
    rescue_boats = Column(Integer, default=0)
    shelters_established = Column(Integer, default=0)
    
    # Response metrics
    response_time_hours = Column(Float, doc="Time to first response")
    
    # Source
    data_source = Column(String(255), doc="Source of this record")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        return f"<FloodEvent(name={self.event_name}, year={self.year}, district={self.primary_district})>"


# =============================================================================
# Utility Functions
# =============================================================================

def create_all_tables(engine) -> None:
    """
    Create all database tables.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(engine)


def drop_all_tables(engine) -> None:
    """
    Drop all database tables. Use with caution!
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.drop_all(engine)
