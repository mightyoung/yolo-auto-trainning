# ADR 0003: Auth Strategy

## Status
Accepted

## Context
We need to secure the API endpoints for both user authentication and service-to-service communication. The system has multiple components (Business API, Training API) that need to authenticate with each other.

## Decision
Use a dual authentication strategy:

### 1. JWT for User Authentication
- **Algorithm**: HS256 (HMAC-SHA256)
- **Token Expiry**: 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- **Token Type**: Bearer token in Authorization header
- **Secret Key**: Required via `JWT_SECRET_KEY` environment variable

### 2. API Keys for Service-to-Service
- **Business API Key**: For external clients calling Business API
- **Training API Key**: For Business API to authenticate with Training API
- **Header**: `X-API-Key` header

### Implementation Details

#### Business API (port 8000)
- All endpoints require JWT by default via `get_current_user` dependency
- Public endpoints explicitly marked (health check, docs)
- API key authentication via `X-API-Key` header for service calls

#### Training API (port 8001)
- Protected by `TRAINING_API_KEY` in request headers
- Internal endpoints only (not exposed to end users)

### Security Configuration
```python
# Required environment variables
JWT_SECRET_KEY=<secure-random-string>  # Must be set
BUSINESS_API_KEY=<api-key>            # For external clients
TRAINING_API_KEY=<api-key>            # For internal communication
```

## Gaps (NOT YET IMPLEMENTED)

### Current State
- Basic JWT and API key authentication implemented
- Token refresh mechanism NOT implemented (users must re-login)
- Role-based access control (RBAC) NOT implemented
- Rate limiting NOT implemented at auth layer

### What Needs to Be Done
1. **Token Refresh**: Implement refresh token flow for extended sessions
2. **RBAC**: Add role-based permissions (admin, user, viewer)
3. **API Key Management**: Add API key generation/rotation UI
4. **Rate Limiting**: Implement per-user or per-IP rate limits
5. **Audit Logging**: Log all authentication events

## Consequences

### Easier
- Standard authentication across the platform
- Clear separation between user and service auth
- Extensible to add OAuth2, SSO, etc.

### More Difficult
- Must manage JWT secret key securely
- Token refresh requires additional infrastructure
- Service-to-service auth needs key rotation process
