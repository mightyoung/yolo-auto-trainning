"""
Celery Tasks - Async task processing for long-running jobs.

Based on Celery best practices:
- https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html
"""

from pathlib import Path
from celery import Celery
import os

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "yolo_auto_training",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# Celery configuration - best practices
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Requeue on worker shutdown
    worker_prefetch_multiplier=1,

    # Result backend
    result_expires=3600,  # 1 hour
    result_extended=True,

    # Task routing
    task_routes={
        "tasks.train.*": {"queue": "training"},
        "tasks.export.*": {"queue": "export"},
        "tasks.data.*": {"queue": "data"},
    },

    # Retry policy - exponential backoff
    task_annotations={
        "*": {
            "rate_limit": "10/m",
            "time_limit": 43200,  # 12 hours
            "soft_time_limit": 36000,  # 10 hours
        }
    },
)


# Retry configuration - exponential backoff
task_retry_config = {
    "max_retries": 3,
    "interval_start": 0.2,
    "interval_step": 0.5,
    "interval_max": 60,
}


@celery_app.task(bind=True, **task_retry_config)
def training_task(self, config: dict):
    """
    Training task.

    Args:
        config: Training configuration
    """
    try:
        # Import here to avoid circular imports
        from ..training.runner import YOLOTrainer

        trainer = YOLOTrainer(
            model=config.get("model", "yolo11m"),
            output_dir=Path(config.get("output_dir", "./runs")),
        )

        result = trainer.train(
            data_yaml=Path(config["data_yaml"]),
            epochs=config.get("epochs", 100),
        )

        return {
            "status": result.status,
            "model_path": str(result.model_path) if result.model_path else None,
            "metrics": result.metrics,
        }

    except Exception as e:
        self.retry(exc=e)


@celery_app.task(bind=True, **task_retry_config)
def hpo_task(self, config: dict):
    """
    Hyperparameter optimization task.

    Args:
        config: HPO configuration
    """
    try:
        from ..training.runner import YOLOTrainer
        from ..training.config import HPOConfig

        trainer = YOLOTrainer(
            model=config.get("model", "yolo11m"),
            output_dir=Path(config.get("output_dir", "./runs")),
        )

        hpo_config = HPOConfig(
            n_trials=config.get("n_trials", 50),
            epochs_per_trial=config.get("epochs_per_trial", 50),
        )

        result = trainer.tune(
            data_yaml=Path(config["data_yaml"]),
            config=hpo_config,
        )

        return {
            "status": result.status,
            "best_params": result.best_params,
            "metrics": result.metrics,
        }

    except Exception as e:
        self.retry(exc=e)


@celery_app.task(bind=True, **task_retry_config)
def export_task(self, config: dict):
    """
    Model export task.

    Args:
        config: Export configuration
    """
    try:
        from ..deployment.exporter import ModelExporter

        exporter = ModelExporter(
            output_dir=Path(config.get("output_dir", "./runs/export")),
        )

        result = exporter.export(
            model_path=Path(config["model_path"]),
            platform=config.get("platform", "jetson_orin"),
            imgsz=config.get("imgsz", 640),
        )

        return {
            "status": result.status,
            "model_path": str(result.model_path) if result.model_path else None,
            "size_mb": result.size_mb,
            "format": result.format,
        }

    except Exception as e:
        self.retry(exc=e)


@celery_app.task(bind=True, **task_retry_config)
def data_discovery_task(self, config: dict):
    """
    Dataset discovery task.

    Args:
        config: Discovery configuration
    """
    try:
        from ..data.discovery import DatasetDiscovery

        discovery = DatasetDiscovery(
            output_dir=Path(config.get("output_dir", "./data/discovered")),
        )

        results = discovery.search(
            query=config["query"],
            max_results=config.get("max_results", 10),
        )

        return {
            "status": "completed",
            "datasets": [
                {
                    "name": d.name,
                    "source": d.source,
                    "images": d.images,
                    "relevance_score": d.relevance_score,
                }
                for d in results
            ],
        }

    except Exception as e:
        self.retry(exc=e)
