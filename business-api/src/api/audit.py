"""
Audit Logging Module
Location: business-api/src/api/audit.py

Contains:
- AuditLogger class for structured audit logging
- Convenience methods for auth, training, and data access events
- FastAPI dependency for injecting audit logger

Usage:
    from .audit import audit_logger, get_audit_logger

    @app.get("/endpoint")
    async def endpoint(
        request: Request,
        audit: AuditLogger = Depends(get_audit_logger)
    ):
        audit.log("action", user_id, "resource", request)
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import Request
import json


class AuditLogger:
    """
    Structured audit logger for tracking user actions and system events.

    Logs are emitted as JSON for easy parsing and analysis by SIEM systems.
    """

    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        # Ensure handler exists to avoid "No handler" warnings
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log(
        self,
        action: str,
        user_id: Optional[str],
        resource: str,
        request: Optional[Request] = None,
        details: Optional[dict] = None,
        status: str = "success"
    ) -> None:
        """
        Log an audit entry with structured data.

        Args:
            action: Type of action (e.g., "auth", "training", "data_access")
            user_id: User identifier (None for anonymous)
            resource: Resource being accessed (e.g., "dataset/123", "training/task_abc")
            request: FastAPI Request object for extracting IP and headers
            details: Additional context-specific data
            status: Operation status ("success", "failure", "error")
        """
        # Extract client IP from request
        client_ip = None
        user_agent = None
        method = None
        path = None

        if request:
            # Get client IP (handles proxies)
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()
            elif request.client:
                client_ip = request.client.host

            user_agent = request.headers.get("User-Agent")
            method = request.method
            path = str(request.url.path)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "ip": client_ip,
            "status": status,
            "details": details or {},
            "request": {
                "method": method,
                "path": path,
                "user_agent": user_agent
            } if method else None
        }

        # Remove None values from entry for cleaner logs
        entry = {k: v for k, v in entry.items() if v is not None}

        self.logger.info(json.dumps(entry))

    # ==================== Convenience Methods ====================

    def log_auth(
        self,
        user_id: str,
        action: str,
        status: str,
        request: Optional[Request] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Log authentication-related events.

        Args:
            user_id: User identifier
            action: Auth action (login, logout, token_refresh, etc.)
            status: success, failure, error
            request: Optional request for IP extraction
            details: Additional auth details
        """
        self.log(
            action="auth",
            user_id=user_id,
            resource=f"auth/{action}",
            request=request,
            details={**(details or {}), "auth_action": action},
            status=status
        )

    def log_training(
        self,
        user_id: str,
        action: str,
        task_id: str,
        request: Optional[Request] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Log training-related events.

        Args:
            user_id: User who initiated the action
            action: Training action (submit, cancel, status_check, etc.)
            task_id: Training task identifier
            request: Optional request for IP extraction
            details: Additional training details (model, epochs, etc.)
        """
        self.log(
            action="training",
            user_id=user_id,
            resource=f"training/{task_id}",
            request=request,
            details={**(details or {}), "training_action": action},
            status="success"
        )

    def log_data_access(
        self,
        user_id: str,
        dataset_id: str,
        action: str,
        request: Optional[Request] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Log dataset/data access events.

        Args:
            user_id: User who accessed the data
            dataset_id: Dataset identifier
            action: Access action (search, download, view, etc.)
            request: Optional request for IP extraction
            details: Additional access details
        """
        self.log(
            action="data_access",
            user_id=user_id,
            resource=f"dataset/{dataset_id}",
            request=request,
            details={**(details or {}), "access_action": action},
            status="success"
        )

    def log_model_operation(
        self,
        user_id: str,
        model_name: str,
        action: str,
        request: Optional[Request] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Log model registry operations.

        Args:
            user_id: User who performed the operation
            model_name: Model name
            action: Operation (create, delete, transition, etc.)
            request: Optional request for IP extraction
            details: Additional operation details
        """
        self.log(
            action="model_operation",
            user_id=user_id,
            resource=f"model/{model_name}",
            request=request,
            details={**(details or {}), "model_action": action},
            status="success"
        )

    def log_api_call(
        self,
        user_id: Optional[str],
        endpoint: str,
        method: str,
        status_code: int,
        request: Optional[Request] = None,
        details: Optional[dict] = None
    ) -> None:
        """
        Log general API calls.

        Args:
            user_id: User identifier (None for anonymous)
            endpoint: API endpoint path
            method: HTTP method
            status_code: Response status code
            request: Optional request for IP extraction
            details: Additional API call details
        """
        self.log(
            action="api_call",
            user_id=user_id,
            resource=endpoint,
            request=request,
            details={
                **(details or {}),
                "http_method": method,
                "status_code": status_code
            },
            status="success" if status_code < 400 else "failure"
        )


# Global audit logger instance
audit_logger = AuditLogger()


async def get_audit_logger() -> AuditLogger:
    """
    FastAPI dependency for injecting the audit logger.

    Usage:
        @app.get("/endpoint")
        async def endpoint(audit: AuditLogger = Depends(get_audit_logger)):
            audit.log("action", user_id, "resource")
    """
    return audit_logger
