"""
Training API Gateway - FastAPI application for GPU training tasks
Location: training-api/src/api/gateway.py

This API runs on GPU server and handles:
- YOLO Model Training
- Ray Tune HPO
- Model Export (ONNX, TensorRT)
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import os
import secrets
import time

from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import redis

# JWT imports
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


# Security
API_KEY_HEADER = "X-API-Key"

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable must be set")
JWT_ALGORITHM = "HS256"

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Internal API Key - must be set in environment
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
if not INTERNAL_API_KEY:
    raise ValueError("INTERNAL_API_KEY environment variable must be set")


def get_redis_client():
    """Get Redis client. Returns None if Redis is disabled or unavailable."""
    # Check DISABLE_REDIS env var for environments without Redis (e.g., Google Colab)
    if os.getenv("DISABLE_REDIS", "").lower() in ("1", "true", "yes"):
        return None
    try:
        return redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        return None


def verify_internal_api_key(api_key: str = None):
    """Verify internal API key using constant-time comparison to prevent timing attacks."""
    if api_key is None:
        return False
    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(api_key, INTERNAL_API_KEY)


# Rate limiter dependency for internal API (API-key based)
async def check_rate_limit(
    x_api_key: str = None,
    request: Request = None,
):
    """
    Check rate limit for API key using sliding window algorithm.

    Args:
        x_api_key: API key from header
        request: HTTP request

    Raises:
        HTTPException: If rate limit exceeded
    """
    if not request:
        return

    if not x_api_key:
        return

    redis_client = get_redis_client()
    if not redis_client:
        return

    # Use the API key as the identifier (or a hash of it)
    # For security, we use a truncated hash
    import hashlib
    api_key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
    client_id = f"api_{api_key_hash}"

    # Determine rate limit based on endpoint
    path = request.url.path

    if "/train/" in path:
        rate_limit = 10  # 10 requests per minute for training
    elif "/hpo/" in path:
        rate_limit = 5   # 5 requests per minute for HPO
    elif "/export/" in path:
        rate_limit = 10  # 10 requests per minute for export
    elif "/label/" in path:
        rate_limit = 5   # 5 requests per minute for labeling
    else:
        rate_limit = 60  # 60 requests per minute for other endpoints

    # Sliding window rate limiting with Lua script
    key = f"rate_limit:{client_id}:{path}"

    try:
        # Lua script for sliding window
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        -- Remove old entries outside the window
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

        -- Count current requests
        local count = redis.call('ZCARD', key)

        if count < limit then
            -- Add new request
            redis.call('ZADD', key, now, now .. '-' .. math.random())
            redis.call('EXPIRE', key, window / 1000)
            return 1
        else
            return 0
        end
        """

        current_time_ms = int(time.time() * 1000)
        window_ms = 60 * 1000  # 1 minute window

        result = redis_client.eval(
            lua_script,
            1,
            key,
            rate_limit,
            window_ms,
            current_time_ms,
        )

        if result == 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {rate_limit} requests per minute.",
            )

    except HTTPException:
        raise
    except Exception as e:
        # Log error but don't block request
        print(f"Rate limit check failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    app.state.redis = get_redis_client()
    yield
    # Shutdown
    if app.state.redis:
        app.state.redis.close()


# Create FastAPI app
app = FastAPI(
    title="YOLO Auto-Training Training API",
    description="GPU training API for YOLO model training and export",
    version="7.0.0",
    lifespan=lifespan
)

# CORS settings - use explicit origins for security
# NEVER use allow_origins=["*"] with allow_credentials=True (browsers reject this)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Security Middleware ====================

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB

    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.MAX_BODY_SIZE:
                raise HTTPException(status_code=413, detail="Request body too large")
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# Register security middlewares
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


# ==================== Health ====================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu": "available",
        "cuda_version": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        "timestamp": str(datetime.now())
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "YOLO Auto-Training Training API",
        "version": "7.0.0",
        "docs": "/docs"
    }


# ==================== Import Routes ====================

from .routes import router as internal_router
from .model_routes import model_router

# Include internal routes
app.include_router(internal_router, prefix="/api/v1/internal", tags=["Internal"])
app.include_router(model_router, prefix="/api/v1/models", tags=["Models"])
