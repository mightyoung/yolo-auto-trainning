"""
Business API Module
Location: business-api/src/api/

Exports:
- audit: Audit logging module
- auth: Authentication utilities
"""

from .audit import audit_logger, AuditLogger, get_audit_logger
from .auth import CurrentUser, get_current_user, get_optional_user, require_role

__all__ = [
    "audit_logger",
    "AuditLogger",
    "get_audit_logger",
    "CurrentUser",
    "get_current_user",
    "get_optional_user",
    "require_role",
]
