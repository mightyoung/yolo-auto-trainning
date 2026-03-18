"""
Business API Gateway - FastAPI application for business logic
Location: business-api/src/api/gateway.py

This API runs on local/terminal and handles:
- Data Discovery (Roboflow, Kaggle, HuggingFace)
- Agent Orchestration (CrewAI)
- Task Scheduling (delegates to Training API)
"""

from contextlib import asynccontextmanager
from typing import Optional, Union
import os
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
import secrets

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import redis

# JWT imports
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


# Security - re-export from auth module
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_TOKEN = HTTPBearer(auto_error=False)

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable must be set")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API Key settings for service-to-service authentication
BUSINESS_API_KEY = os.getenv("BUSINESS_API_KEY", "default-business-api-key")

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Training API settings - must be set in environment
TRAINING_API_URL = os.getenv("TRAINING_API_URL")
if not TRAINING_API_URL:
    raise ValueError("TRAINING_API_URL environment variable must be set")

TRAINING_API_KEY = os.getenv("TRAINING_API_KEY")
if not TRAINING_API_KEY:
    raise ValueError("TRAINING_API_KEY environment variable must be set")


# Redis connection pool (singleton)
_redis_pool: redis.ConnectionPool = None


def get_redis_client():
    """Get Redis client from connection pool."""
    global _redis_pool
    try:
        if _redis_pool is None:
            _redis_pool = redis.ConnectionPool.from_url(
                REDIS_URL,
                decode_responses=True,
                max_connections=20
            )
        return redis.Redis(connection_pool=_redis_pool)
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    app.state.redis = get_redis_client()

    # Import and initialize training client
    from .training_client import TrainingAPIClient
    app.state.training_client = TrainingAPIClient(
        base_url=TRAINING_API_URL,
        api_key=TRAINING_API_KEY
    )

    yield

    # Shutdown
    if app.state.redis:
        app.state.redis.close()
    global _redis_pool
    if _redis_pool:
        _redis_pool.disconnect()
        _redis_pool = None


# Create FastAPI app
app = FastAPI(
    title="YOLO Auto-Training Business API",
    description="Business logic API for data discovery and task scheduling",
    version="7.0.0",
    lifespan=lifespan
)

# CORS middleware
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


# ==================== Authentication ====================
# Authentication is now handled by the auth.py module

from .auth import (
    get_current_user,
    get_optional_user,
    require_role,
    CurrentUser,
    create_access_token,
    verify_token,
    verify_api_key,
)


# ==================== Routes ====================

from .routes import (
    data_router,
    train_router,
    deploy_router,
    callback_router,
    analysis_router
)
from .agent_routes import agent_router

# Register routers
app.include_router(data_router, prefix="/api/v1/data", tags=["Data"])
app.include_router(train_router, prefix="/api/v1/train", tags=["Training"])
app.include_router(deploy_router, prefix="/api/v1/deploy", tags=["Deployment"])
app.include_router(callback_router, prefix="/api/v1/callback", tags=["Callback"])
app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    redis_status = "connected"
    try:
        if request.app.state.redis:
            # Run sync redis.ping() in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, request.app.state.redis.ping)
    except Exception:
        redis_status = "disconnected"

    # Check training API
    training_api_status = "unavailable"
    try:
        if await request.app.state.training_client.health_check():
            training_api_status = "available"
    except Exception:
        pass

    return {
        "status": "healthy",
        "version": "7.0.0",
        "redis": redis_status,
        "training_api": training_api_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "YOLO Auto-Training Business API",
        "version": "7.0.0",
        "docs": "/docs"
    }
