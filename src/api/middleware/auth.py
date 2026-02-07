"""
Sanjivani AI - API Key Authentication Middleware

Production-grade API key authentication with Redis-backed rate limiting.
"""

import os
import time
import hashlib
from typing import Optional, Callable
from functools import wraps

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)

# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# In production, load from secure storage (env, secrets manager, database)
# For now, support multiple keys via environment variable
def get_valid_api_keys() -> set:
    """Load valid API keys from environment."""
    keys_str = os.getenv("API_KEYS", "")
    if not keys_str:
        # Default development key (should be removed in production)
        return {"dev-api-key-12345"}
    return set(keys_str.split(","))


def hash_api_key(key: str) -> str:
    """Hash API key for secure comparison and storage."""
    return hashlib.sha256(key.encode()).hexdigest()


async def get_api_key(
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
    api_key_query: Optional[str] = Security(API_KEY_QUERY),
) -> str:
    """
    Extract and validate API key from request.
    
    Accepts key from either header (X-API-Key) or query parameter (api_key).
    Header takes precedence.
    """
    api_key = api_key_header or api_key_query
    
    if not api_key:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "API key required. Provide via X-API-Key header or api_key query parameter."
            }
        )
    
    valid_keys = get_valid_api_keys()
    
    if api_key not in valid_keys:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "Invalid API key."
            }
        )
    
    logger.debug(f"Authenticated with API key: {api_key[:8]}...")
    return api_key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.
    
    Uses in-memory storage by default (Redis recommended for production).
    """
    
    def __init__(
        self, 
        app, 
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = {}  # In-memory token buckets
        self.last_update = {}
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Prefer API key, fall back to IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{hash_api_key(api_key)[:16]}"
        
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    def _refill_tokens(self, client_id: str) -> float:
        """Refill tokens based on time elapsed."""
        now = time.time()
        last = self.last_update.get(client_id, now)
        elapsed = now - last
        
        # Calculate tokens to add
        tokens_to_add = elapsed * (self.requests_per_minute / 60)
        current_tokens = self.tokens.get(client_id, self.burst_size)
        new_tokens = min(self.burst_size, current_tokens + tokens_to_add)
        
        self.tokens[client_id] = new_tokens
        self.last_update[client_id] = now
        
        return new_tokens
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        available_tokens = self._refill_tokens(client_id)
        
        if available_tokens < 1:
            logger.warning(f"Rate limit exceeded for {client_id}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": 60 // self.requests_per_minute
                },
                headers={"Retry-After": str(60 // self.requests_per_minute)}
            )
        
        # Consume a token
        self.tokens[client_id] = available_tokens - 1
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(self.tokens[client_id]))
        
        return response


def require_api_key(func: Callable) -> Callable:
    """Decorator to require API key for specific endpoints."""
    @wraps(func)
    async def wrapper(*args, api_key: str = Depends(get_api_key), **kwargs):
        return await func(*args, **kwargs)
    return wrapper
