"""
API Gateway - FastAPI application with security middleware.

Based on FastAPI best practices:
- https://fastapi.tiangolo.com/tutorial/cors/
- https://fastapi.tiangolo.com/tutorial/security/
"""

from contextlib import asynccontextmanager
from typing import List, Optional
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import time

# JWT imports
try:
    import jwt
    from jwt import PyJWT
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# Redis imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_TOKEN = HTTPBearer(auto_error=False)

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# Redis client
def get_redis_client():
    """Get Redis client."""
    if not REDIS_AVAILABLE:
        return None
    return redis.from_url(REDIS_URL, decode_responses=True)


# JWT token management
def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT not installed")

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})

    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT not installed")

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})

    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """Verify JWT token and return payload."""
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT not installed")

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


# API Key management
def generate_api_key() -> str:
    """Generate a new API key."""
    return f"yolo_{secrets.token_urlsafe(32)}"


def store_api_key_in_redis(api_key: str, user_id: str) -> bool:
    """Store API key in Redis with 30-day expiry."""
    redis_client = get_redis_client()
    if not redis_client:
        return False

    try:
        # Store API key -> user_id mapping with 30 day expiry
        redis_client.setex(f"api_key:{api_key}", 30 * 24 * 60 * 60, user_id)
        return True
    except Exception:
        return False


def verify_api_key_in_redis(api_key: str) -> Optional[str]:
    """Verify API key in Redis and return user_id."""
    redis_client = get_redis_client()
    if not redis_client:
        return None

    try:
        user_id = redis_client.get(f"api_key:{api_key}")
        return user_id
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("Starting YOLO Auto Training API...")
    yield
    print("Shutting down YOLO Auto Training API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="YOLO Auto Training API",
        description="AI-driven automated YOLO training and deployment",
        version="6.0.0",
        lifespan=lifespan,
    )

    # CORS Configuration - Production Safe
    # ⚠️ v6: Never use allow_origins=["*"] with credentials
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8080"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    return app


# Authentication dependency
async def get_current_user(
    api_key: Optional[str] = Depends(API_KEY_HEADER),
    token: Optional[HTTPAuthorizationCredentials] = Depends(BEARER_TOKEN),
):
    """
    Verify API key or JWT token.

    Args:
        api_key: API key from header
        token: Bearer token

    Returns:
        User information

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key and not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
        )

    # Verify API key
    if api_key:
        user_id = verify_api_key_in_redis(api_key)
        if user_id:
            return {"user_id": user_id, "api_key": api_key}

    # Verify JWT token
    if token:
        try:
            payload = verify_token(token.credentials)
            user_id = payload.get("sub")
            if user_id:
                return {"user_id": user_id, "token_type": payload.get("type")}

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token verification failed: {str(e)}",
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


# Rate limiter dependency
async def check_rate_limit(
    user: dict = Depends(get_current_user),
    request: Request = None,
):
    """
    Check rate limit for user using sliding window algorithm.

    Args:
        user: Current user
        request: HTTP request

    Raises:
        HTTPException: If rate limit exceeded
    """
    if not request:
        return

    redis_client = get_redis_client()
    if not redis_client:
        return

    user_id = user.get("user_id")
    if not user_id:
        return

    # Determine rate limit based on endpoint
    # Training: 10 req/min, Search: 60 req/min
    path = request.url.path

    if "/train/" in path:
        rate_limit = 10  # 10 requests per minute for training
    elif "/deploy/" in path:
        rate_limit = 10  # 10 requests per minute for deployment
    else:
        rate_limit = 60  # 60 requests per minute for other endpoints

    # Sliding window rate limiting with Lua script
    # Key format: rate_limit:user_id:endpoint
    key = f"rate_limit:{user_id}:{path}"

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


# Create app instance
app = create_app()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "6.0.0"}


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from .metrics import create_metrics_response
    return create_metrics_response()


# Import and include routers
from .routes import router as api_router

app.include_router(api_router, prefix="/api/v1")
