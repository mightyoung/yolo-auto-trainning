"""
Plateau Detection and Breaking Manager.

Moved from training-api/src/api/routes.py and extended to support:
- In-stage restart: Level 1/2 adjustments trigger immediate training restart
  with adjusted parameters, without waiting for the stage to complete.
- LLM diagnosis: Calls DeepSeek API when plateau is confirmed for
  intelligent root-cause analysis and action recommendations.
"""
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# Thread-safe task cache shared across callbacks
_tasks_lock = threading.Lock()
_tasks_cache: Dict[str, Dict[str, Any]] = {}


@dataclass
class PlateauBreakingConfig:
    """Configuration for dynamic plateau detection and breaking during training."""
    enabled: bool = True
    window: int = 10
    min_improvement: float = 0.002
    min_epochs_before_trigger: int = 30

    # Level 1: LR decay
    lr_reduction_factor: float = 0.5
    lr_reduction_max_times: int = 3
    min_lr: float = 1e-6

    # Level 2: Augmentation boost
    augmentation_boost_epochs: int = 15
    boosted_mixup: float = 0.3
    boosted_copy_paste: float = 0.4
    boosted_degrees: float = 15.0
    boosted_translate: float = 0.2
    boosted_scale: float = 0.7

    # Level 3: Data expansion
    auto_expand_data: bool = True
    expansion_target_map: float = 0.90
    max_expansion_rounds: int = 2


@dataclass
class PlateauDecision:
    """Result of plateau analysis — what action to take next."""
    triggered: bool = False
    level: int = 0
    action: str = ""          # "lr_decay" | "augment_boost" | "data_expansion" | "none"
    adjustment: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Optional[Dict[str, Any]] = None  # LLM diagnosis result
    avg_recent_mAP50: float = 0.0
    improvement: float = 0.0


class PlateauManager:
    """Monitors training metrics and triggers plateau-breaking strategies.

    Unlike the original DynamicTrainingManager (which only writes signals to cache),
    this class supports IN-STAGE RESTART: when a plateau is detected, it returns
    a PlateauDecision with the specific parameter adjustments needed, allowing the
    caller to immediately restart training with those parameters.

    Usage:
        pm = PlateauManager(task_id="train_abc123", config=PlateauBreakingConfig())
        # In each epoch:
        decision = pm.on_metric(epoch, total_epochs, metrics)
        if decision.triggered:
            if decision.action == "lr_decay":
                config.lr0 = decision.adjustment["new_lr"]
                # restart training with updated config
            elif decision.action == "augment_boost":
                config.mixup = decision.adjustment["mixup"]
                # ...
    """

    def __init__(
        self,
        task_id: str,
        config: Optional[PlateauBreakingConfig] = None,
        on_decision: Optional[Callable[[PlateauDecision], None]] = None,
        on_llm_diagnosis: Optional[Callable[[Dict], None]] = None,
    ):
        self.task_id = task_id
        self.cfg = config or PlateauBreakingConfig()
        self._map_history: List[Tuple[int, float]] = []
        self._lr_reduction_count = 0
        self._augment_boost_active = False
        self._augment_boost_remaining = 0
        self._expansion_round = 0
        self._signaled_expansion = False
        self._original_augment: Dict[str, float] = {}
        self._triggered_strategies: List[Dict] = []
        self._last_reported_epoch = -1
        self._current_lr: float = 0.01  # Will be overridden by config
        self._current_augment: Dict[str, float] = {}
        self._in_stage_restarts: int = 0
        self._max_in_stage_restarts: int = 3  # Per-stage restart limit
        self._llm_diagnosis: Optional[Dict] = None
        self._on_decision = on_decision
        self._on_llm_diagnosis = on_llm_diagnosis

    def set_current_lr(self, lr: float) -> None:
        self._current_lr = lr

    def set_current_augment(self, augment: Dict[str, float]) -> None:
        self._current_augment = augment.copy()

    def on_metric(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
    ) -> PlateauDecision:
        """Called each epoch with current metrics. Returns PlateauDecision."""
        if not self.cfg.enabled:
            return PlateauDecision()

        if epoch <= self._last_reported_epoch:
            return PlateauDecision()

        self._last_reported_epoch = epoch
        mAP50 = metrics.get("mAP50", 0.0)
        self._map_history.append((epoch, mAP50))

        # Keep history bounded
        if len(self._map_history) > self.cfg.window * 4:
            self._map_history = self._map_history[-self.cfg.window * 3:]

        # Update shared cache for Business API visibility
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["live_metrics"] = metrics
                _tasks_cache[self.task_id]["live_mAP50"] = mAP50
                _tasks_cache[self.task_id]["strategies_triggered"] = self._triggered_strategies
                _tasks_cache[self.task_id]["in_stage_restarts"] = self._in_stage_restarts

        # Don't trigger before minimum epoch threshold
        if epoch < self.cfg.min_epochs_before_trigger:
            return PlateauDecision()

        # Handle augmentation boost countdown
        if self._augment_boost_remaining > 0:
            self._augment_boost_remaining -= 1
            if self._augment_boost_remaining == 0:
                self._end_augment_boost()

        # Check for plateau
        decision = self._check_plateau(epoch, total_epochs)
        if decision.triggered:
            self._apply_decision(decision)

        return decision

    def _check_plateau(self, current_epoch: int, total_epochs: int) -> PlateauDecision:
        """Detect plateau using sliding window comparison."""
        if len(self._map_history) < self.cfg.window:
            return PlateauDecision()

        recent = self._map_history[-self.cfg.window:]
        older = self._map_history[-self.cfg.window * 2:-self.cfg.window]

        if not recent or not older:
            return PlateauDecision()

        avg_recent = sum(m for _, m in recent) / len(recent)
        avg_older = sum(m for _, m in older) / len(older)
        improvement = avg_recent - avg_older

        if improvement >= self.cfg.min_improvement:
            return PlateauDecision()  # Still improving

        # Plateau detected
        decision = PlateauDecision(
            triggered=True,
            improvement=improvement,
            avg_recent_mAP50=avg_recent,
        )

        # Check restart limit
        if self._in_stage_restarts >= self._max_in_stage_restarts:
            logger.warning(
                f"[{self.task_id}][PLATEAU] In-stage restart limit reached "
                f"({self._max_in_stage_restarts}). Signaling data expansion instead."
            )
            decision.level = 3
            decision.action = "data_expansion"
            decision.adjustment = {
                "reason": "restart_limit_reached",
                "avg_mAP50": avg_recent,
            }
            return decision

        # Determine strategy
        if self._lr_reduction_count < self.cfg.lr_reduction_max_times:
            new_lr = max(
                self.cfg.min_lr,
                self._current_lr * self.cfg.lr_reduction_factor,
            )
            decision.level = 1
            decision.action = "lr_decay"
            decision.adjustment = {
                "old_lr": self._current_lr,
                "new_lr": new_lr,
                "decay_count": self._lr_reduction_count + 1,
                "resume_from": None,  # Caller fills this with best.pt path
            }
        elif not self._augment_boost_active:
            decision.level = 2
            decision.action = "augment_boost"
            decision.adjustment = {
                "mixup": self.cfg.boosted_mixup,
                "copy_paste": self.cfg.boosted_copy_paste,
                "degrees": self.cfg.boosted_degrees,
                "translate": self.cfg.boosted_translate,
                "scale": self.cfg.boosted_scale,
                "boost_epochs": self.cfg.augmentation_boost_epochs,
                "resume_from": None,
            }
        elif not self._signaled_expansion:
            current_best = max((m for _, m in self._map_history), default=0.0)
            if current_best >= self.cfg.expansion_target_map - 0.05 and \
               self._expansion_round < self.cfg.max_expansion_rounds:
                decision.level = 3
                decision.action = "data_expansion"
                decision.adjustment = {
                    "current_best": current_best,
                    "target": self.cfg.expansion_target_map,
                    "round": self._expansion_round + 1,
                }
            else:
                # Close to target but no expansion possible — trigger LLM diagnosis
                decision.level = 0
                decision.action = "llm_diagnosis"
                decision.adjustment = {
                    "current_best": current_best,
                    "target": self.cfg.expansion_target_map,
                    "mAP50_history": [m for _, m in self._map_history[-50:]],
                    "current_epoch": current_epoch,
                    "total_epochs": total_epochs,
                }
        else:
            decision.action = "none"

        return decision

    def _apply_decision(self, decision: PlateauDecision) -> None:
        """Apply a plateau decision: update state and notify."""
        level = decision.level
        action = decision.action

        logger.warning(
            f"[{self.task_id}][PLATEAU] Level-{level} {action} triggered: "
            f"improvement={decision.improvement:.5f}, avg_mAP50={decision.avg_recent_mAP50:.5f}"
        )

        self._triggered_strategies.append({
            "epoch": self._last_reported_epoch,
            "level": level,
            "action": action,
            "mAP50": decision.avg_recent_mAP50,
            "adjustment": decision.adjustment,
        })

        if level == 1:
            self._lr_reduction_count += 1
            new_lr = decision.adjustment.get("new_lr", self._current_lr)
            self._current_lr = new_lr
            self._in_stage_restarts += 1
        elif level == 2:
            self._augment_boost_active = True
            self._augment_boost_remaining = self.cfg.augmentation_boost_epochs
            self._current_augment.update({
                "mixup": decision.adjustment.get("mixup", self.cfg.boosted_mixup),
                "copy_paste": decision.adjustment.get("copy_paste", self.cfg.boosted_copy_paste),
                "degrees": decision.adjustment.get("degrees", self.cfg.boosted_degrees),
            })
            self._in_stage_restarts += 1
        elif level == 3:
            self._signaled_expansion = True
            self._expansion_round += 1

        # Update cache
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id].update({
                    "lr_decay_triggered": level == 1,
                    "lr_decay_count": self._lr_reduction_count,
                    "augment_boost_active": self._augment_boost_active,
                    "augment_boost_remaining": self._augment_boost_remaining,
                    "data_expansion_requested": level == 3,
                    "data_expansion_round": self._expansion_round,
                    "in_stage_restarts": self._in_stage_restarts,
                    "strategies_triggered": self._triggered_strategies,
                })
                if level == 1:
                    _tasks_cache[self.task_id]["lr_decay_signal"] = decision.adjustment
                if level == 2:
                    _tasks_cache[self.task_id]["augment_boost_signal"] = decision.adjustment
                if level == 3:
                    _tasks_cache[self.task_id]["data_expansion_signal"] = decision.adjustment

        # Notify callback
        if self._on_decision:
            self._on_decision(decision)

    def _end_augment_boost(self) -> None:
        """Signal augmentation boost ended."""
        self._augment_boost_active = False
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id].update({
                    "augment_boost_active": False,
                    "augment_boost_signal": None,
                })
        logger.info(f"[{self.task_id}][PLATEAU] Augmentation boost ended. Resuming normal augmentation.")

    def set_llm_diagnosis(self, diagnosis: Dict[str, Any]) -> None:
        """Set LLM diagnosis result after plateau is confirmed."""
        self._llm_diagnosis = diagnosis
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["llm_diagnosis"] = diagnosis
        if self._on_llm_diagnosis:
            self._on_llm_diagnosis(diagnosis)

    def get_status(self) -> Dict[str, Any]:
        """Return current plateau detection status."""
        return {
            "enabled": self.cfg.enabled,
            "lr_reduction_count": self._lr_reduction_count,
            "augment_boost_active": self._augment_boost_active,
            "augment_boost_remaining": self._augment_boost_remaining,
            "expansion_round": self._expansion_round,
            "signaled_expansion": self._signaled_expansion,
            "in_stage_restarts": self._in_stage_restarts,
            "max_in_stage_restarts": self._max_in_stage_restarts,
            "current_best_mAP50": max((m for _, m in self._map_history), default=0.0),
            "recent_mAP50": self._map_history[-1][1] if self._map_history else 0.0,
            "strategies_triggered": self._triggered_strategies,
            "llm_diagnosis": self._llm_diagnosis,
        }

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Return the path to the best checkpoint seen so far (for resume)."""
        if not self._map_history:
            return None
        # The caller is responsible for tracking best checkpoint path
        # This is stored in cache by the training loop
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                return _tasks_cache[self.task_id].get("best_checkpoint_path")
        return None

    def set_best_checkpoint_path(self, path: str) -> None:
        """Set the best checkpoint path for future resume."""
        with _tasks_lock:
            if self.task_id in _tasks_cache:
                _tasks_cache[self.task_id]["best_checkpoint_path"] = path
