"""
Sanjivani AI - FastAPI Main Application

Production-ready API for crisis intelligence system.
"""

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
    yield
    logger.info("Sanjivani AI API shutting down...")


app = FastAPI(
    title="Sanjivani AI API",
    description="Multimodal Crisis Intelligence System for Flood Disaster Response",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from src.api.routes import health, nlp, vision, forecasting

app.include_router(health.router, tags=["Health"])
app.include_router(nlp.router, prefix="/api/v1", tags=["NLP"])
app.include_router(vision.router, prefix="/api/v1", tags=["Vision"])
app.include_router(forecasting.router, prefix="/api/v1", tags=["Forecasting"])


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
