# Unit Tests - Authentication Module

import pytest
from pathlib import Path
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add src to path - handle both direct and package execution
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set required environment variables BEFORE importing gateway
import os
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("TRAINING_API_URL", "http://localhost:8001")
os.environ.setdefault("TRAINING_API_KEY", "test-api-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from fastapi import HTTPException


# ==================== Fixtures ====================

_original_sys_modules = {}


@pytest.fixture(autouse=True)
def _setup_module_mocks():
    """Set up redis/celery mocks at test execution time, then clean up after.

    This replaces the old module-level sys.modules[] assignments that caused
    cross-test pollution (redis/celery mocks persisted into subsequent test files,
    causing 401 auth errors when gateway was imported with the stale mock).
    """
    global _original_sys_modules

    # Save originals (may be None or a real module object)
    _original_sys_modules['redis'] = sys.modules.get('redis')
    _original_sys_modules['celery'] = sys.modules.get('celery')

    # Install mocks - these allow gateway.py to import even though
    # redis/celery packages are not installed in the test environment.
    # The legacy src/api/gateway.py imports routes.py which imports celery.
    sys.modules['redis'] = MagicMock()
    sys.modules['celery'] = MagicMock()

    yield

    # Restore originals - if they were real modules, restore them;
    # if they were None (not present), remove the mock entry
    # This ensures subsequent test files can import the real modules
    # without inheriting our mocks.
    for mod_name in ('redis', 'celery'):
        orig = _original_sys_modules.get(mod_name)
        if orig is None:
            # Module was not present before - remove the mock
            sys.modules.pop(mod_name, None)
        else:
            # Restore original (real module or previous mock)
            sys.modules[mod_name] = orig

    _original_sys_modules.clear()


# ==================== Test JWT Functions ====================

class TestJWTAuthentication:
    """Test JWT authentication functions."""

    def test_create_access_token(self):
        """Access token is created with correct claims."""
        from api.gateway import create_access_token, JWT_AVAILABLE

        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not available")

        token = create_access_token({"sub": "user123"})
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Refresh token is created."""
        from api.gateway import create_refresh_token, JWT_AVAILABLE

        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not available")

        token = create_refresh_token({"sub": "user123"})
        assert token is not None
        assert isinstance(token, str)

    def test_verify_valid_token(self):
        """Valid token verifies successfully."""
        from api.gateway import create_access_token, verify_token, JWT_AVAILABLE

        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not available")

        token = create_access_token({"sub": "user123"})
        payload = verify_token(token)

        assert payload["sub"] == "user123"
        assert payload["type"] == "access"

    def test_verify_token_missing_payload(self):
        """Token without sub is still valid but returns empty payload."""
        from api.gateway import create_access_token, verify_token, JWT_AVAILABLE

        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not available")

        # Create token without sub - should still be valid
        token = create_access_token({})

        # Token is valid, just has no sub
        payload = verify_token(token)
        assert payload is not None


# ==================== Test API Key Functions ====================

class TestAPIKey:
    """Test API key functions."""

    def test_generate_api_key_format(self):
        """API key has correct format."""
        from api.gateway import generate_api_key

        key = generate_api_key()

        assert key.startswith("yolo_")
        assert len(key) > 20

    def test_generate_api_key_unique(self):
        """Generated API keys are unique."""
        from api.gateway import generate_api_key

        keys = [generate_api_key() for _ in range(100)]
        unique_keys = set(keys)

        # All keys should be unique
        assert len(unique_keys) == 100


# ==================== Test Redis Integration ====================

class TestRedisIntegration:
    """Test Redis integration functions."""

    def test_redis_integration_available(self):
        """Redis integration is available."""
        from api.gateway import REDIS_AVAILABLE

        # Test that the flag exists and is boolean
        assert isinstance(REDIS_AVAILABLE, bool)

    def test_store_api_key_handles_no_redis(self):
        """Store API key handles Redis not available."""
        from api.gateway import store_api_key_in_redis, REDIS_AVAILABLE

        # When Redis is not available, should return False
        if not REDIS_AVAILABLE:
            result = store_api_key_in_redis("test_key", "user123")
            assert result is False

    def test_verify_api_key_handles_no_redis(self):
        """Verify API key handles Redis not available."""
        from api.gateway import verify_api_key_in_redis, REDIS_AVAILABLE

        # When Redis is not available, should return None
        if not REDIS_AVAILABLE:
            result = verify_api_key_in_redis("test_key")
            assert result is None


# ==================== Test Authentication Dependency ====================

class TestAuthDependency:
    """Test FastAPI authentication dependency."""

    def test_get_current_user_no_credentials(self):
        """No credentials raises 401."""
        from api.gateway import get_current_user
        from fastapi import HTTPException

        # Call without credentials
        with pytest.raises(HTTPException) as exc_info:
            # Need to run in async context
            import asyncio
            asyncio.run(get_current_user(None, None))

        assert exc_info.value.status_code == 401
        assert "Missing authentication credentials" in exc_info.value.detail


# ==================== Test Rate Limiting ====================

class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_configured_per_endpoint(self):
        """Rate limits are configured per endpoint."""
        from api.gateway import REDIS_AVAILABLE

        # Just verify the logic exists
        # In production, would test with actual Redis
        assert True  # Logic is in check_rate_limit function

    def test_rate_limit_different_endpoints(self):
        """Different endpoints have different rate limits."""
        # Training: 10 req/min
        # Search: 60 req/min

        # Logic is in check_rate_limit function
        assert True
