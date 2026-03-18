"""
Prometheus Metrics Module for YOLO Auto-Training API.

Based on best practices:
- https://docs.prometheus.io/docs/instrumenting/exporters/
- https://github.com/prometheus/client_python
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from typing import Optional
import time


# ============================================================
# Request Metrics
# ============================================================

# Request counter - counts total requests
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# Request duration histogram
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Requests in progress
http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "endpoint"]
)


# ============================================================
# Training Metrics
# ============================================================

# Training jobs counter
training_jobs_total = Counter(
    "training_jobs_total",
    "Total training jobs",
    ["status", "model"]  # status: started, completed, failed
)

# Active training jobs
training_jobs_active = Gauge(
    "training_jobs_active",
    "Number of currently running training jobs"
)

# Training duration
training_duration_seconds = Histogram(
    "training_duration_seconds",
    "Training job duration in seconds",
    ["model"],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 43200, 86400)
)

# Training metrics (mAP, etc.)
training_metrics = Gauge(
    "training_metrics",
    "Training result metrics",
    ["metric_name", "model", "job_id"]
)


# ============================================================
# Dataset Metrics
# ============================================================

# Dataset discovery counter
dataset_discoveries_total = Counter(
    "dataset_discoveries_total",
    "Total dataset discovery requests",
    ["source"]  # source: roboflow, kaggle, huggingface
)

# Dataset download counter
dataset_downloads_total = Counter(
    "dataset_downloads_total",
    "Total dataset downloads",
    ["source", "status"]
)


# ============================================================
# Export Metrics
# ============================================================

# Model export counter
model_exports_total = Counter(
    "model_exports_total",
    "Total model export requests",
    ["platform", "status"]  # platform: onnx, tensorrt, etc.
)


# ============================================================
# System Metrics
# ============================================================

# Redis connection status
redis_connected = Gauge(
    "redis_connected",
    "Redis connection status (1=connected, 0=disconnected)"
)

# GPU metrics (if available)
gpu_utilization = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"]
)

gpu_memory_used = Gauge(
    "gpu_memory_used_mb",
    "GPU memory used in MB",
    ["device"]
)


# ============================================================
# Application Info
# ============================================================

app_info = Info("yolo_auto_training", "YOLO Auto-Training application info")


class MetricsMiddleware:
    """Middleware to collect request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope.get("path", "/")

        # Increment in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=path).inc()

        start_time = time.time()

        # Custom send that tracks status
        status_code = 200
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Record duration
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)

            # Record request
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status=status_code
            ).inc()

            # Decrement in-progress
            http_requests_in_progress.labels(method=method, endpoint=path).dec()


def create_metrics_response() -> Response:
    """Create Prometheus metrics response."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


def record_training_start(model: str) -> None:
    """Record training job start."""
    training_jobs_total.labels(status="started", model=model).inc()
    training_jobs_active.inc()


def record_training_complete(model: str, duration: float) -> None:
    """Record training job completion."""
    training_jobs_total.labels(status="completed", model=model).inc()
    training_jobs_active.dec()
    training_duration_seconds.labels(model=model).observe(duration)


def record_training_failed(model: str) -> None:
    """Record training job failure."""
    training_jobs_total.labels(status="failed", model=model).inc()
    training_jobs_active.dec()


def record_training_metric(metric_name: str, value: float, model: str, job_id: str) -> None:
    """Record training metric (mAP, loss, etc.)."""
    training_metrics.labels(
        metric_name=metric_name,
        model=model,
        job_id=job_id
    ).set(value)


def record_dataset_discovery(source: str) -> None:
    """Record dataset discovery request."""
    dataset_discoveries_total.labels(source=source).inc()


def record_dataset_download(source: str, status: str) -> None:
    """Record dataset download."""
    dataset_downloads_total.labels(source=source, status=status).inc()


def record_model_export(platform: str, status: str) -> None:
    """Record model export."""
    model_exports_total.labels(platform=platform, status=status).inc()


def update_redis_status(connected: bool) -> None:
    """Update Redis connection status."""
    redis_connected.set(1 if connected else 0)


def update_gpu_metrics(device: str, utilization: Optional[float] = None, memory_mb: Optional[float] = None) -> None:
    """Update GPU metrics."""
    if utilization is not None:
        gpu_utilization.labels(device=device).set(utilization)
    if memory_mb is not None:
        gpu_memory_used.labels(device=device).set(memory_mb)
