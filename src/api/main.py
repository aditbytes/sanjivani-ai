"""
Sanjivani AI - FastAPI Main Application

Production-ready API for crisis intelligence system with:
- Request ID tracking
- Rate limiting
- Structured error handling
- CORS support
"""

import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.utils.logger import setup_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    setup_logging(log_level=settings.log_level)
    logger.info("Sanjivani AI API starting...")
    logger.info(f"Environment: {'production' if not settings.debug else 'development'}")
    yield
    logger.info("Sanjivani AI API shutting down...")


app = FastAPI(
    title="Sanjivani AI API",
    description="Multimodal Crisis Intelligence System for Flood Disaster Response",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

# ---------- Middleware Stack (order matters: last added = first executed) ----------

# 1. CORS - Allow cross-origin requests
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# 2. Rate Limiting (optional, enable via environment variable)
if os.getenv("ENABLE_RATE_LIMIT", "false").lower() == "true":
    from src.api.middleware.auth import RateLimitMiddleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
        burst_size=int(os.getenv("RATE_LIMIT_BURST", "10"))
    )
    logger.info("Rate limiting enabled")

# 3. Request ID tracking
from src.api.middleware.request_id import RequestIDMiddleware
app.add_middleware(RequestIDMiddleware)

# ---------- Exception Handlers ----------
from src.api.exceptions import register_exception_handlers
register_exception_handlers(app)

# ---------- Routers ----------
from src.api.routes import health, nlp, vision, forecasting
from src.api.routes.metrics import router as metrics_router, MetricsMiddleware

app.include_router(health.router, tags=["Health"])
app.include_router(nlp.router, prefix="/api/v1", tags=["NLP"])
app.include_router(vision.router, prefix="/api/v1", tags=["Vision"])
app.include_router(forecasting.router, prefix="/api/v1", tags=["Forecasting"])
app.include_router(metrics_router, tags=["Monitoring"])

# Add metrics collection middleware
app.add_middleware(MetricsMiddleware)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs" if settings.debug else "Disabled in production",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
