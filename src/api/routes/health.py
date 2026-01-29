"""
Sanjivani AI - Health Check Endpoints
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter

from src.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/ready")
async def readiness_check() -> Dict:
    """Readiness check for Kubernetes."""
    return {"ready": True}


@router.get("/health/live")
async def liveness_check() -> Dict:
    """Liveness check for Kubernetes."""
    return {"alive": True}
