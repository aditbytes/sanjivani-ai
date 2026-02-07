"""
Sanjivani AI - Custom Exceptions and Error Handlers

Structured error responses for production APIs.
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from src.utils.logger import get_logger
from src.api.middleware.request_id import get_request_id

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    message: str
    detail: Optional[Any] = None
    request_id: Optional[str] = None


class SanjivaniException(Exception):
    """Base exception for Sanjivani AI."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "internal_error",
        status_code: int = 500,
        detail: Optional[Any] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class ModelNotFoundError(SanjivaniException):
    """Raised when a required ML model is not found."""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found. Please ensure the model is trained.",
            error_code="model_not_found",
            status_code=503
        )


class ModelInferenceError(SanjivaniException):
    """Raised when model inference fails."""
    
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code="inference_error",
            status_code=500,
            detail=detail
        )


class ValidationError(SanjivaniException):
    """Raised for input validation failures."""
    
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code="validation_error",
            status_code=422,
            detail=detail
        )


class ResourceNotFoundError(SanjivaniException):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} '{identifier}' not found.",
            error_code="not_found",
            status_code=404
        )


def create_error_response(
    status_code: int,
    error: str,
    message: str,
    detail: Optional[Any] = None
) -> JSONResponse:
    """Create a standardized error response."""
    content = {
        "error": error,
        "message": message,
        "request_id": get_request_id()
    }
    if detail:
        content["detail"] = detail
    
    return JSONResponse(status_code=status_code, content=content)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""
    
    @app.exception_handler(SanjivaniException)
    async def sanjivani_exception_handler(request: Request, exc: SanjivaniException):
        logger.error(
            f"SanjivaniException: {exc.error_code} - {exc.message}",
            extra={"request_id": get_request_id()}
        )
        return create_error_response(
            status_code=exc.status_code,
            error=exc.error_code,
            message=exc.message,
            detail=exc.detail
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # Handle dict detail (from our auth middleware)
        if isinstance(exc.detail, dict):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    **exc.detail,
                    "request_id": get_request_id()
                }
            )
        
        return create_error_response(
            status_code=exc.status_code,
            error="http_error",
            message=str(exc.detail)
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", "Unknown error"),
                "type": error.get("type", "unknown")
            })
        
        request_id = get_request_id()
        logger.warning(f"Validation error: {errors} | request_id={request_id}")
        
        return create_error_response(
            status_code=422,
            error="validation_error",
            message="Request validation failed",
            detail=errors
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception(
            f"Unhandled exception: {exc}",
            extra={"request_id": get_request_id()}
        )
        
        return create_error_response(
            status_code=500,
            error="internal_error",
            message="An unexpected error occurred. Please try again later."
        )
