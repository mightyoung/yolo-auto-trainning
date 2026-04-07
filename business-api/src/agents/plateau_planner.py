"""Bounded plateau-recovery planner.

This module provides a small Tree-of-Thoughts-like decision step for plateau
handling only. It builds a small set of candidate recovery actions, scores
them deterministically from the current signal bundle, and returns the best
candidate plus a compact rationale.
"""

from __future__ import annotations

from typing import Any

from .worker_memory import build_attempt_record

MAX_PLATEAU_CANDIDATES = 4


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _summarize_adjustments(adjustments_triggered: list[dict[str, Any]] | None) -> dict[str, Any]:
    lr_counts = {
        item.get("decay_count")
        for item in adjustments_triggered or []
        if item.get("level") == 1 and item.get("decay_count") is not None
    }
    data_expansion_done = any(item.get("level") == 3 for item in adjustments_triggered or [])
    return {
        "lr_decay_counts": sorted(lr_counts),
        "data_expansion_done": data_expansion_done,
    }


def _score_lr_decay(status_summary: dict[str, Any]) -> tuple[float, str]:
    signal = dict(status_summary.get("lr_decay_signal") or {})
    decay_count = _as_int(signal.get("lr_decay_count"), 1)
    current_map = _as_float(signal.get("current_mAP50"), _as_float(status_summary.get("live_mAP50")))
    factor = _as_float(signal.get("factor"), 0.5)
    target_map = _as_float(signal.get("target_mAP50"), 0.9)

    gap = max(target_map - current_map, 0.0)
    score = 0.68
    score += 0.05 if decay_count <= 2 else 0.0
    score += 0.04 if factor <= 0.5 else 0.0
    score -= min(gap, 0.25) * 0.35
    score += 0.03 if current_map >= target_map - 0.05 else 0.0
    score = max(0.0, min(score, 0.95))

    reason = (
        f"lr_decay_count={decay_count}, current_mAP50={current_map:.4f}, "
        f"target_mAP50={target_map:.4f}, factor={factor:.4f}"
    )
    return score, reason


def _score_data_expansion(status_summary: dict[str, Any]) -> tuple[float, str]:
    signal = dict(status_summary.get("data_expansion_signal") or {})
    current_map = _as_float(signal.get("current_mAP50"), _as_float(status_summary.get("live_mAP50")))
    target_map = _as_float(signal.get("target_mAP50"), 0.9)
    recommendation = str(signal.get("recommendation", "") or "")

    gap = max(target_map - current_map, 0.0)
    score = 0.62
    score += min(gap, 0.25) * 0.55
    score += 0.05 if recommendation else 0.0
    score += 0.03 if current_map < target_map - 0.10 else 0.0
    score = max(0.0, min(score, 0.95))

    reason = (
        f"current_mAP50={current_map:.4f}, target_mAP50={target_map:.4f}, "
        f"gap={gap:.4f}, recommendation={'yes' if recommendation else 'no'}"
    )
    return score, reason


def build_plateau_candidates(
    status_summary: dict[str, Any],
    adjustments_triggered: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a bounded candidate set for plateau recovery."""
    adjustment_state = _summarize_adjustments(adjustments_triggered)
    candidates: list[dict[str, Any]] = []

    lr_decay_triggered = bool(status_summary.get("lr_decay_triggered"))
    lr_decay_signal = status_summary.get("lr_decay_signal") or {}
    if lr_decay_triggered and lr_decay_signal:
        decay_count = _as_int(lr_decay_signal.get("lr_decay_count"), 1)
        if decay_count <= 3 and decay_count not in adjustment_state["lr_decay_counts"]:
            score, reason = _score_lr_decay(status_summary)
            candidates.append(
                {
                    "action": "lr_decay",
                    "stage": "lr_decay",
                    "score": score,
                    "reason": reason,
                    "details": {
                        "decay_count": decay_count,
                        "signal": lr_decay_signal,
                    },
                }
            )

    data_expansion_requested = bool(status_summary.get("data_expansion_requested"))
    data_expansion_signal = status_summary.get("data_expansion_signal") or {}
    if data_expansion_requested and data_expansion_signal and not adjustment_state["data_expansion_done"]:
        score, reason = _score_data_expansion(status_summary)
        candidates.append(
            {
                "action": "data_expansion",
                "stage": "data_expansion",
                "score": score,
                "reason": reason,
                "details": {
                    "signal": data_expansion_signal,
                },
            }
        )

    return sorted(candidates, key=lambda item: (item["score"], item["action"]), reverse=True)[:MAX_PLATEAU_CANDIDATES]


def select_plateau_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the best candidate from a bounded candidate set."""
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["score"], item["action"]))


def build_plateau_decision(
    status_summary: dict[str, Any],
    adjustments_triggered: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build a compact plateau-recovery decision trace."""
    candidates = build_plateau_candidates(status_summary, adjustments_triggered)
    selected = select_plateau_candidate(candidates)
    if selected is None:
        return None

    rejected = [candidate for candidate in candidates if candidate is not selected]
    rationale = (
        f"selected={selected['action']} score={selected['score']:.2f}; "
        f"rejected={len(rejected)}"
    )
    return {
        "selected": selected,
        "rejected": rejected,
        "candidate_count": len(candidates),
        "rationale": rationale,
        "signal_bundle": {
            "live_mAP50": status_summary.get("live_mAP50"),
            "lr_decay_triggered": status_summary.get("lr_decay_triggered"),
            "data_expansion_requested": status_summary.get("data_expansion_requested"),
            "augment_boost_active": status_summary.get("augment_boost_active"),
            "curriculum_stage": status_summary.get("curriculum_stage"),
        },
    }


def build_plateau_attempt_record(
    *,
    task_id: str,
    training_task_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Create a typed attempt record for a plateau-recovery decision."""
    selected = decision["selected"]
    return build_attempt_record(
        attempt_type="plateau_search",
        stage="plateau_recovery",
        outcome="selected",
        source="plateau_planner",
        action=selected["action"],
        training_task_id=training_task_id,
        details={
            "task_id": task_id,
            "candidate_count": decision["candidate_count"],
            "selected": selected,
            "rejected": decision["rejected"],
            "rationale": decision["rationale"],
            "signal_bundle": decision["signal_bundle"],
        },
    )
