"""Compact summary helpers for coordinator and worker histories."""

from __future__ import annotations

from typing import Any


def _compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def build_compact_summary(
    *,
    kind: str,
    stage: str | None = None,
    outcome: str | None = None,
    action: str | None = None,
    subject: str | None = None,
    detail: str | None = None,
    limit: int = 160,
) -> str:
    """Build a short human-readable summary for attempt or tool history."""
    parts: list[str] = []
    if subject:
        parts.append(_compact_text(subject, 32))
    if kind:
        parts.append(_compact_text(kind, 24))
    if stage:
        parts.append(_compact_text(stage, 24))
    if outcome:
        parts.append(_compact_text(outcome, 24))
    if action:
        parts.append(_compact_text(action, 24))

    summary = " ".join(parts).strip()
    if detail:
        summary = f"{summary}: {_compact_text(detail, 80)}" if summary else _compact_text(detail, 80)

    return _compact_text(summary or kind, limit)


def summarize_tool_batch(
    tool_name: str,
    *,
    outcome: str,
    detail: str | None = None,
    subject: str | None = None,
    limit: int = 120,
) -> str:
    """Summarize a tool batch in the style of a short commit subject."""
    return build_compact_summary(
        kind="tool batch",
        stage=tool_name,
        outcome=outcome,
        subject=subject or tool_name,
        detail=detail,
        limit=limit,
    )


def summarize_attempt(
    attempt_type: str,
    stage: str,
    outcome: str,
    *,
    action: str | None = None,
    error: str | None = None,
    detail: str | None = None,
    limit: int = 140,
) -> str:
    """Summarize a retry / failure / success attempt record."""
    extra = error or detail
    return build_compact_summary(
        kind=attempt_type,
        stage=stage,
        outcome=outcome,
        action=action,
        detail=extra,
        limit=limit,
    )
