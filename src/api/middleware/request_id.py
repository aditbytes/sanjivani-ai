"""
Sanjivani AI - Request ID Middleware

Adds unique request IDs for tracing and debugging across distributed systems.
"""

import uuid
from contextvars import ContextVar
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Context variable to store request ID across async boundaries
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request IDs to all requests.
    
    - Generates UUID4 for each request
    - Accepts existing ID from X-Request-ID header
    - Adds ID to response headers
    - Stores ID in context for logging
    """
    
    HEADER_NAME = "X-Request-ID"
    
    async def dispatch(self, request: Request, call_next):
        # Use existing request ID or generate new one
        request_id = request.headers.get(self.HEADER_NAME)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in context for access in handlers and logging
        token = request_id_ctx.set(request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers[self.HEADER_NAME] = request_id
            
            return response
        finally:
            # Reset context
            request_id_ctx.reset(token)
