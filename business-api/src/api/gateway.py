"""
Business API Gateway - FastAPI application for business logic
Location: business-api/src/api/gateway.py

This API runs on local/terminal and handles:
- Data Discovery (Roboflow, Kaggle, HuggingFace)
- Agent Orchestration (CrewAI)
- Task Scheduling (delegates to Training API)
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()  # Load business-api/.env

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# JWT imports
try:
    import jwt  # noqa: F401 - used conditionally via JWT_AVAILABLE
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


# ==================== Runtime Settings ====================
# These are read at runtime, not at import time

class RuntimeSettings:
    """Runtime settings that read from environment on each access.

    This avoids the "module-level snapshot" problem where config values
    are frozen at import time instead of being read at request time.
    """

    # JWT settings
    @property
    def JWT_SECRET_KEY(self) -> str:
        return os.getenv("JWT_SECRET_KEY", "")

    @property
    def JWT_ALGORITHM(self) -> str:
        return "HS256"

    @property
    def ACCESS_TOKEN_EXPIRE_MINUTES(self) -> int:
        return 30

    # API Key settings - no default, must be configured
    @property
    def BUSINESS_API_KEY(self) -> str | None:
        return os.getenv("BUSINESS_API_KEY")

    # Redis settings
    @property
    def REDIS_URL(self) -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")

    @property
    def REDIS_PASSWORD(self) -> str | None:
        return os.getenv("REDIS_PASSWORD")

    # Training API settings
    @property
    def TRAINING_API_URL(self) -> str | None:
        return os.getenv("TRAINING_API_URL")

    @property
    def TRAINING_API_KEY(self) -> str | None:
        return os.getenv("TRAINING_API_KEY")

    # CORS
    @property
    def ALLOWED_ORIGINS(self) -> list:
        return os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")


# Global settings instance
settings = RuntimeSettings()


def _validate_startup_config():
    """Validate required configuration when the app actually starts."""
    missing = []
    if not settings.JWT_SECRET_KEY:
        missing.append("JWT_SECRET_KEY")
    if not settings.TRAINING_API_URL:
        missing.append("TRAINING_API_URL")
    if not settings.TRAINING_API_KEY:
        missing.append("TRAINING_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required business API configuration: " + ", ".join(missing)
        )


# Redis connection pool (singleton)
_redis_pool: redis.ConnectionPool = None


def get_redis_client():
    """Get Redis client from connection pool."""
    global _redis_pool
    try:
        if _redis_pool is None:
            _redis_pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
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
    _validate_startup_config()
    app.state.redis = get_redis_client()

    # Import and initialize training client
    from .training_client import TrainingAPIClient
    app.state.training_client = TrainingAPIClient(
        base_url=settings.TRAINING_API_URL,
        api_key=settings.TRAINING_API_KEY
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
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



# ==================== Routes ====================

from .agent_routes import agent_router
from .route_handlers import (
    analysis_router,
    callback_router,
    data_router,
    deploy_router,
    queue_router,
    train_router,
)

# Register routers
app.include_router(data_router, prefix="/api/v1/data", tags=["Data"])
app.include_router(train_router, prefix="/api/v1/train", tags=["Training"])
app.include_router(deploy_router, prefix="/api/v1/deploy", tags=["Deployment"])
app.include_router(callback_router, prefix="/api/v1/callback", tags=["Callback"])
app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(queue_router, prefix="/api/v1/queue", tags=["Queue"])


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
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
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
