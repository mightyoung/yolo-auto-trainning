"""
Structured Logging Configuration for YOLO Auto-Training.

Based on best practices:
- Use JSON format for machine-parseable logs
- Include consistent fields: timestamp, level, service, message
- Include correlation IDs for request tracing
- Use python-json-logger for JSON formatting
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger
import uuid


class StructuredLogFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging.

    Includes:
    - timestamp (ISO 8601 UTC)
    - level
    - service
    - message
    - correlation_id (for request tracing)
    - extra fields
    """

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """Add custom fields to log record."""
        # Timestamp in ISO 8601 format (UTC)
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Log level
        log_record["level"] = record.levelname

        # Service name
        log_record["service"] = "yolo-auto-training"

        # Logger name
        log_record["logger"] = record.name

        # Message
        log_record["message"] = record.getMessage()

        # Correlation ID (if set in thread local)
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            log_record["correlation_id"] = correlation_id

        # Extra fields
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)

        # Exception info
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    service_name: str = "yolo-auto-training",
) -> logging.Logger:
    """
    Setup structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        service_name: Service name for log context

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create JSON formatter
    formatter = StructuredLogFormatter(
        "%(timestamp)s %(level)s %(service)s %(logger)s %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class CorrelationContext:
    """
    Context manager for request correlation IDs.

    Usage:
        with CorrelationContext("request-123"):
            logger.info("Processing request")
    """

    _context: Dict[str, str] = {}

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.previous_id = None

    def __enter__(self) -> str:
        self.previous_id = self._context.get("correlation_id")
        self._context["correlation_id"] = self.correlation_id
        return self.correlation_id

    def __exit__(self, *args):
        if self.previous_id:
            self._context["correlation_id"] = self.previous_id
        else:
            self._context.pop("correlation_id", None)

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID."""
        return cls._context.get("correlation_id")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with structured logging."""
    return logging.getLogger(name)


# Default logger instance
default_logger = setup_logging()


class LogCapture:
    """
    Context manager for capturing logs in tests.

    Usage:
        with LogCapture() as logs:
            logger.info("test message")
        assert any("test message" in log for log in logs)
    """

    def __init__(self, logger_name: str = "yolo-auto-training"):
        self.logger_name = logger_name
        self.logs = []
        self.handler = None

    def __enter__(self):
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.logs.append(record.getMessage())

        logger = logging.getLogger(self.logger_name)
        logger.addHandler(self.handler)
        return self

    def __exit__(self, *args):
        logger = logging.getLogger(self.logger_name)
        logger.removeHandler(self.handler)


# Request logging helper
def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    correlation_id: Optional[str] = None,
) -> None:
    """Log HTTP request with structured data."""
    logger.info(
        f"{method} {path}",
        extra={
            "extra_data": {
                "http": {
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
                "correlation_id": correlation_id,
            }
        },
    )


def log_training_event(
    logger: logging.Logger,
    event: str,
    job_id: str,
    model: str,
    metrics: Optional[Dict[str, float]] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Log training event with structured data."""
    extra = {
        "extra_data": {
            "training": {
                "event": event,
                "job_id": job_id,
                "model": model,
            },
            "correlation_id": correlation_id,
        }
    }

    if metrics:
        extra["extra_data"]["training"]["metrics"] = metrics

    logger.info(f"Training {event}: {job_id}", extra=extra)
