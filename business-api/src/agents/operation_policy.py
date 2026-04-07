"""Centralized policy for risky agent-side operations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


ALLOW = "allow"
ASK = "ask"
DENY = "deny"

HIGH_RISK_OPERATIONS = {
    "ssh_dataset_check",
    "ssh_dataset_download",
    "ssh_dataset_yaml",
    "gpu_training_submit",
    "gpu_scheduler_start",
    "model_export",
    "dataset_download",
}


@dataclass(frozen=True)
class OperationDecision:
    """Structured allow/deny/ask decision."""

    operation: str
    behavior: str
    reason: str
    context: dict[str, Any]


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _missing_envs(*names: str) -> list[str]:
    return [name for name in names if not os.getenv(name)]


def evaluate_operation(operation: str, *, context: dict[str, Any] | None = None) -> OperationDecision:
    """Return a bounded policy decision for the requested operation."""
    context = dict(context or {})
    mode = os.getenv("OPERATION_POLICY_MODE", ALLOW).strip().lower()

    if mode == DENY:
        return OperationDecision(
            operation=operation,
            behavior=DENY,
            reason="operation policy is set to deny",
            context=context,
        )
    if mode == ASK and operation in HIGH_RISK_OPERATIONS:
        return OperationDecision(
            operation=operation,
            behavior=ASK,
            reason="operation policy requires confirmation for risky actions",
            context=context,
        )

    if operation in {"ssh_dataset_check", "ssh_dataset_download", "ssh_dataset_yaml"}:
        missing = _missing_envs("GPU_SERVER_HOST", "GPU_SERVER_USER", "GPU_SERVER_PASS")
        if missing:
            return OperationDecision(
                operation=operation,
                behavior=DENY,
                reason=f"missing SSH credentials: {', '.join(missing)}",
                context=context,
            )

    if operation == "dataset_download":
        source = str(context.get("source") or "").lower()
        if source != "coco_builtin":
            missing = _missing_envs("GPU_SERVER_HOST", "GPU_SERVER_USER", "GPU_SERVER_PASS", "ROBOFLOW_API_KEY")
            if missing:
                return OperationDecision(
                    operation=operation,
                    behavior=DENY,
                    reason=f"missing dataset-download credentials: {', '.join(missing)}",
                    context=context,
                )

    if operation in {"gpu_training_submit", "model_export"}:
        missing = _missing_envs("TRAINING_API_URL", "TRAINING_API_KEY")
        if missing:
            return OperationDecision(
                operation=operation,
                behavior=DENY,
                reason=f"missing training API settings: {', '.join(missing)}",
                context=context,
            )

    if operation == "gpu_scheduler_start" and _env_truthy("DISABLE_GPU_SCHEDULER"):
        return OperationDecision(
            operation=operation,
            behavior=DENY,
            reason="GPU scheduler is disabled by environment",
            context=context,
        )

    return OperationDecision(
        operation=operation,
        behavior=ALLOW,
        reason="allowed",
        context=context,
    )


def require_operation_allowed(operation: str, *, context: dict[str, Any] | None = None) -> OperationDecision:
    """Raise if the decision is not allow."""
    decision = evaluate_operation(operation, context=context)
    if decision.behavior != ALLOW:
        raise PermissionError(f"{operation} {decision.behavior}: {decision.reason}")
    return decision
