"""
Training API Gateway - FastAPI application for GPU training tasks
Location: training-api/src/api/gateway.py

This API runs on GPU server and handles:
- YOLO Model Training
- Ray Tune HPO
- Model Export (ONNX, TensorRT)
"""

from contextlib import asynccontextmanager
from typing import Optional
import os
import secrets

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
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
    """Get Redis client."""
    try:
        return redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        return None


def verify_internal_api_key(api_key: str = None):
    """Verify internal API key."""
    if api_key == INTERNAL_API_KEY:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key"
    )


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
