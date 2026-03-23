"""
Continuous Training Automation Pipeline.

Implements the InStatus MLOps Playbook:
1. Monitor production inference quality
2. Detect drift -> trigger retraining
3. A/B test new model vs production model
4. Auto-rollback if new model underperforms
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ABTestResult:
    """Result of an A/B comparison between two models."""

    model_a_metric: float
    model_b_metric: float
    winner: str  # "a" / "b" / "inconclusive"
    confidence: float  # 0.0-1.0
    sample_count_a: int = 0
    sample_count_b: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_a_metric": self.model_a_metric,
            "model_b_metric": self.model_b_metric,
            "winner": self.winner,
            "confidence": self.confidence,
            "sample_count_a": self.sample_count_a,
            "sample_count_b": self.sample_count_b,
            "timestamp": self.timestamp,
        }


@dataclass
class DriftDecision:
    """Output of the drift detection decision engine."""

    action: str  # "retrain" / "monitor" / "ok"
    drift_score: float
    message: str


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class ContinuousTrainingPipeline:
    """
    Continuous training automation pipeline.

    Stages:
        idle          -> no active pipeline
        monitoring    -> collecting inference metrics
        drift_detected -> drift threshold breached, deciding next step
        retraining    -> new model training in progress
        ab_testing    -> A/B test running between candidate and production models
        promoting     -> promoting candidate to production
        rolled_back   -> candidate rejected, production restored
    """

    STAGE_IDLE = "idle"
    STAGE_MONITORING = "monitoring"
    STAGE_DRIFT_DETECTED = "drift_detected"
    STAGE_RETRAINING = "retraining"
    STAGE_AB_TESTING = "ab_testing"
    STAGE_PROMOTING = "promoting"
    STAGE_ROLLED_BACK = "rolled_back"

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        default_drift_threshold: float = 0.05,
        default_ab_duration_hours: int = 24,
    ):
        """
        Args:
            redis_client: Optional Redis client for persistent state.
            default_drift_threshold: Default fractional drop in mAP that triggers retrain.
            default_ab_duration_hours: Default A/B test duration in hours.
        """
        self.redis = redis_client
        self.default_drift_threshold = default_drift_threshold
        self.default_ab_duration_hours = default_ab_duration_hours
        self._active_pipeline: Optional[Dict[str, Any]] = None
        self._ab_test_results: List[ABTestResult] = []
        self._inference_samples_a: List[float] = []
        self._inference_samples_b: List[float] = []

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift_and_decide(
        self,
        current_map: float,
        historical_avg: float,
        threshold: Optional[float] = None,
    ) -> DriftDecision:
        """
        Decide action based on the gap between current performance and historical average.

        Args:
            current_map: Latest production model mAP (0.0-1.0).
            historical_avg: Long-running average mAP (0.0-1.0).
            threshold: Fractional drop that triggers retrain. Defaults to default_drift_threshold.

        Returns:
            DriftDecision with action, score, and message.
        """
        threshold = threshold or self.default_drift_threshold

        if historical_avg <= 0:
            logger.warning(
                "[ContinuousTraining] historical_avg is <= 0; cannot compute drift"
            )
            return DriftDecision(
                action="monitor",
                drift_score=0.0,
                message="Invalid historical_avg; continuing to monitor.",
            )

        drift_score = (historical_avg - current_map) / historical_avg
        abs_drop = historical_avg - current_map

        logger.info(
            f"[ContinuousTraining] Drift check: current={current_map:.4f}, "
            f"historical={historical_avg:.4f}, drift_score={drift_score:.4f}, "
            f"abs_drop={abs_drop:.4f}, threshold={threshold}"
        )

        if drift_score >= threshold:
            msg = (
                f"Drift detected: score={drift_score:.4f} >= threshold={threshold}. "
                "Triggering retraining."
            )
            logger.warning(f"[ContinuousTraining] {msg}")
            self._transition_stage(self.STAGE_DRIFT_DETECTED)
            return DriftDecision(action="retrain", drift_score=drift_score, message=msg)

        if drift_score >= threshold * 0.5:
            return DriftDecision(
                action="monitor",
                drift_score=drift_score,
                message=f"Mild degradation ({drift_score:.4f}), intensifying monitoring.",
            )

        return DriftDecision(
            action="ok",
            drift_score=drift_score,
            message=f"Performance stable. drift_score={drift_score:.4f}",
        )

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def start_retrain_pipeline(
        self,
        task_id: str,
        model_name: str,
        output_dir: str = "/home/wangxin/runs",
    ) -> str:
        """
        Start a new training pipeline triggered by drift detection.

        Args:
            task_id: Unique task identifier.
            model_name: Base model to fine-tune.
            output_dir: Where to store outputs.

        Returns:
            pipeline_id for tracking.
        """
        pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"

        self._active_pipeline = {
            "pipeline_id": pipeline_id,
            "task_id": task_id,
            "model_name": model_name,
            "output_dir": output_dir,
            "stage": self.STAGE_RETRAINING,
            "started_at": datetime.now().isoformat(),
            "candidate_model": None,
            "production_model": model_name,
            "ab_test_result": None,
        }

        self._persist_pipeline()
        logger.info(
            f"[ContinuousTraining] Pipeline {pipeline_id} started for task {task_id}"
        )
        return pipeline_id

    def run_ab_test(
        self,
        model_a: str,
        model_b: str,
        duration_hours: int = 24,
        min_samples: int = 100,
    ) -> ABTestResult:
        """
        Run an A/B test between two models on synthetic/production inference traffic.

        Args:
            model_a: Path or name of the production (control) model.
            model_b: Path or name of the candidate model.
            duration_hours: How long to run the test.
            min_samples: Minimum inference samples before evaluating.

        Returns:
            ABTestResult with winner, confidence, and metrics.
        """
        logger.info(
            f"[ContinuousTraining] Starting A/B test: {model_a} vs {model_b} "
            f"(duration={duration_hours}h, min_samples={min_samples})"
        )

        self._transition_stage(self.STAGE_AB_TESTING)
        self._inference_samples_a = []
        self._inference_samples_b = []

        try:
            from src.training.runner import YOLOTrainer
            from src.training.config import DEFAULT_TRAINING_CONFIG
            from pathlib import Path

            end_time = time.time() + duration_hours * 3600

            while time.time() < end_time:
                # Simulate inference scoring on validation split.
                # In production this would run against actual inference logs.
                runner_a = YOLOTrainer(model=model_a, output_dir=Path("/tmp"))
                runner_b = YOLOTrainer(model=model_b, output_dir=Path("/tmp"))

                # Quick val metric estimate (1 epoch to get mAP50).
                # Use abbreviated epochs for speed; real deployment uses full val set.
                cfg_a = DEFAULT_TRAINING_CONFIG
                cfg_a.epochs = 1
                cfg_a.imgsz = 320
                cfg_a.patience = 1

                cfg_b = DEFAULT_TRAINING_CONFIG
                cfg_b.epochs = 1
                cfg_b.imgsz = 320
                cfg_b.patience = 1

                result_a = runner_a.train(data_yaml=Path("dummy"), config=cfg_a)
                result_b = runner_b.train(data_yaml=Path("dummy"), config=cfg_b)

                map_a = result_a.metrics.get("metrics/mAP50(B)", 0.0) if result_a.metrics else 0.0
                map_b = result_b.metrics.get("metrics/mAP50(B)", 0.0) if result_b.metrics else 0.0

                self._inference_samples_a.append(map_a)
                self._inference_samples_b.append(map_b)

                if (
                    len(self._inference_samples_a) >= min_samples
                    and len(self._inference_samples_b) >= min_samples
                ):
                    break

                time.sleep(60)  # Re-evaluate every minute

            # Compute aggregate metrics
            avg_a = sum(self._inference_samples_a) / len(self._inference_samples_a) if self._inference_samples_a else 0.0
            avg_b = sum(self._inference_samples_b) / len(self._inference_samples_b) if self._inference_samples_b else 0.0

            # Simple win determination
            delta = avg_b - avg_a
            margin = abs(delta) * 0.05  # 5% margin for "inconclusive"

            if delta > margin:
                winner = "b"
                confidence = min(1.0, abs(delta) * 5)
            elif delta < -margin:
                winner = "a"
                confidence = min(1.0, abs(delta) * 5)
            else:
                winner = "inconclusive"
                confidence = 0.5

            result = ABTestResult(
                model_a_metric=avg_a,
                model_b_metric=avg_b,
                winner=winner,
                confidence=confidence,
                sample_count_a=len(self._inference_samples_a),
                sample_count_b=len(self._inference_samples_b),
            )

            self._ab_test_results.append(result)
            if self._active_pipeline:
                self._active_pipeline["ab_test_result"] = result.to_dict()
                self._active_pipeline["ab_test_winner"] = winner

            logger.info(
                f"[ContinuousTraining] A/B test complete: winner={winner}, "
                f"conf={confidence:.2f}, metric_a={avg_a:.4f}, metric_b={avg_b:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"[ContinuousTraining] A/B test failed: {e}", exc_info=True)
            return ABTestResult(
                model_a_metric=0.0,
                model_b_metric=0.0,
                winner="inconclusive",
                confidence=0.0,
            )

    def auto_rollback(
        self,
        current_model: Optional[str] = None,
        previous_model: Optional[str] = None,
    ) -> bool:
        """
        Rollback to the previous (production) model if the current candidate degraded.

        Args:
            current_model: The model to replace.
            previous_model: The model to restore.

        Returns:
            True if rollback succeeded, False otherwise.
        """
        if previous_model is None and self._active_pipeline:
            previous_model = self._active_pipeline.get("production_model")

        if previous_model is None:
            logger.error("[ContinuousTraining] No previous_model specified for rollback")
            return False

        logger.warning(
            f"[ContinuousTraining] Auto-rollback triggered: {current_model} "
            f"-> {previous_model}"
        )

        try:
            # In production this would:
            # 1. Update the model registry to point to previous_model
            # 2. Restart the inference service
            # 3. Notify via webhook / Slack
            self._transition_stage(self.STAGE_ROLLED_BACK)

            if self._active_pipeline:
                self._active_pipeline["rolled_back_to"] = previous_model
                self._active_pipeline["rolled_back_at"] = datetime.now().isoformat()

            self._persist_pipeline()
            return True

        except Exception as e:
            logger.error(f"[ContinuousTraining] Rollback failed: {e}", exc_info=True)
            return False

    def promote_candidate(self, candidate_model: str) -> bool:
        """
        Promote the candidate model to production.

        Args:
            candidate_model: Path or name of the model to promote.

        Returns:
            True on success.
        """
        logger.info(f"[ContinuousTraining] Promoting {candidate_model} to production")
        self._transition_stage(self.STAGE_PROMOTING)

        if self._active_pipeline:
            self._active_pipeline["production_model"] = candidate_model
            self._active_pipeline["promoted_at"] = datetime.now().isoformat()
            self._persist_pipeline()

        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current pipeline status.

        Returns:
            Dict with active flag, stage, model info, and key metrics.
        """
        pipeline = self._active_pipeline
        return {
            "active": pipeline is not None,
            "stage": pipeline.get("stage") if pipeline else self.STAGE_IDLE,
            "pipeline_id": pipeline.get("pipeline_id") if pipeline else None,
            "task_id": pipeline.get("task_id") if pipeline else None,
            "current_model": pipeline.get("production_model") if pipeline else None,
            "candidate_model": pipeline.get("candidate_model") if pipeline else None,
            "ab_test_winner": pipeline.get("ab_test_winner") if pipeline else None,
            "ab_test_result": pipeline.get("ab_test_result") if pipeline else None,
            "rolled_back_at": pipeline.get("rolled_back_at") if pipeline else None,
            "promoted_at": pipeline.get("promoted_at") if pipeline else None,
            "started_at": pipeline.get("started_at") if pipeline else None,
            "available": True,
        }

    def reset(self) -> None:
        """Reset the pipeline to idle state."""
        self._active_pipeline = None
        self._ab_test_results = []
        self._inference_samples_a = []
        self._inference_samples_b = []
        self._transition_stage(self.STAGE_IDLE)
        logger.info("[ContinuousTraining] Pipeline reset to idle")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_stage(self, stage: str) -> None:
        """Update the active pipeline stage."""
        if self._active_pipeline is not None:
            self._active_pipeline["stage"] = stage
            self._active_pipeline["stage_updated_at"] = datetime.now().isoformat()

    def _persist_pipeline(self) -> None:
        """Persist pipeline state to Redis if available."""
        if self.redis is None or self._active_pipeline is None:
            return
        try:
            import json
            key = f"continuous_training:pipeline:{self._active_pipeline.get('pipeline_id', 'current')}"
            self.redis.set(key, json.dumps(self._active_pipeline))
        except Exception as e:
            logger.warning(f"[ContinuousTraining] Redis persist failed: {e}")
