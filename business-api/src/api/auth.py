"""
Authentication Module
Location: business-api/src/api/auth.py

Contains:
- CurrentUser class
- get_current_user dependency
- get_optional_user dependency
- require_role factory
- API key verification
"""

from typing import Optional
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import time

# JWT imports
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_TOKEN = HTTPBearer(auto_error=False)

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API Key settings for service-to-service authentication
BUSINESS_API_KEY = os.getenv("BUSINESS_API_KEY", "default-business-api-key")

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis connection pool (singleton)
_redis_pool = None


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


# Import redis conditionally
# Check DISABLE_REDIS env var for environments without Redis (e.g., Google Colab)
if os.getenv("DISABLE_REDIS", "").lower() in ("1", "true", "yes"):
    REDIS_AVAILABLE = False
else:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False


class CurrentUser:
    """User object returned by authentication."""
    def __init__(self, user_id: str, role: str = "user"):
        self.user_id = user_id
        self.role = role

    def __repr__(self):
        return f"CurrentUser(user_id={self.user_id}, role={self.role})"


def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def verify_api_key(provided_key: str) -> bool:
    """Verify API key using timing-safe comparison to prevent timing attacks."""
    return secrets.compare_digest(provided_key, BUSINESS_API_KEY)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(BEARER_TOKEN),
    api_key: Optional[str] = Depends(API_KEY_HEADER),
) -> CurrentUser:
    """
    Get current user from JWT token or API key.

    Supports two authentication methods:
    1. Bearer token (JWT) - for user authentication
    2. X-API-Key header - for service-to-service authentication

    Returns CurrentUser object with user_id and role.
    """
    # Try API key first (service-to-service)
    if api_key and verify_api_key(api_key):
        return CurrentUser(user_id="service", role="service")

    # Try JWT token
    if credentials:
        if not JWT_AVAILABLE:
            return CurrentUser(user_id="anonymous", role="user")

        token = credentials.credentials
        payload = verify_token(token)
        user_id = payload.get("sub", payload.get("user_id", "anonymous"))
        role = payload.get("role", "user")
        return CurrentUser(user_id=user_id, role=role)

    # No valid credentials provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide either a valid JWT token or API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(BEARER_TOKEN),
    api_key: Optional[str] = Depends(API_KEY_HEADER),
) -> Optional[CurrentUser]:
    """Get current user if authenticated, otherwise return None (optional auth)."""
    try:
        return await get_current_user(credentials, api_key)
    except HTTPException:
        return None


def require_role(required_role: str):
    """Dependency factory for role-based access control."""
    async def role_checker(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required. Your role: '{current_user.role}'"
            )
        return current_user
    return role_checker


# Rate limiter dependency
async def check_rate_limit(
    current_user: CurrentUser = Depends(get_current_user),
    request: Request = None,
):
    """
    Check rate limit for user using sliding window algorithm.

    Args:
        current_user: Current authenticated user
        request: HTTP request

    Raises:
        HTTPException: If rate limit exceeded
    """
    if not request:
        return

    if not REDIS_AVAILABLE:
        return

    redis_client = get_redis_client()
    if not redis_client:
        return

    user_id = current_user.user_id
    if not user_id:
        return

    # Determine rate limit based on endpoint
    path = request.url.path

    if "/train/" in path:
        rate_limit = 10  # 10 requests per minute for training
    elif "/deploy/" in path:
        rate_limit = 10  # 10 requests per minute for deployment
    elif "/agent/" in path:
        rate_limit = 5   # 5 requests per minute for agent endpoints
    else:
        rate_limit = 60  # 60 requests per minute for other endpoints

    # Sliding window rate limiting with Lua script
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
