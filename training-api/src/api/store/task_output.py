"""Bounded output spool helpers for training tasks."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any


MAX_TASK_OUTPUT_BYTES = 5 * 1024 * 1024
OUTPUT_TEXT_LIMIT = 240

_ROOT_DIR = Path(tempfile.gettempdir()) / "yolo-auto-training" / "training-api" / "task-output"
_TASK_OUTPUTS: dict[str, "TaskOutputSpool"] = {}
_TASK_OUTPUTS_LOCK = Lock()


def _compact_text(value: Any, limit: int = OUTPUT_TEXT_LIMIT) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _open_append_handle(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "a", encoding="utf-8")


class TaskOutputSpool:
    """File-backed spool for long-running training/export output."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.path = _ROOT_DIR / f"{task_id}.log"
        self._offset = 0
        self._summary = ""
        self._capped = False
        self._lock = Lock()

    def append(self, text: str, *, summary: str | None = None) -> dict[str, Any]:
        chunk = text if text.endswith("\n") else f"{text}\n"
        encoded = chunk.encode("utf-8")
        with self._lock:
            if self._capped:
                return self.snapshot()
            if self._offset + len(encoded) > MAX_TASK_OUTPUT_BYTES:
                chunk = "\n[output truncated: max task output size reached]\n"
                encoded = chunk.encode("utf-8")
                self._capped = True
            with _open_append_handle(self.path) as handle:
                handle.write(chunk)
            self._offset += len(encoded)
            self._summary = _compact_text(summary or chunk.strip() or self._summary)
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "output_path": str(self.path),
            "output_offset": self._offset,
            "output_summary": self._summary or None,
            "output_capped": self._capped,
        }


def get_task_output_spool(task_id: str) -> TaskOutputSpool:
    with _TASK_OUTPUTS_LOCK:
        spool = _TASK_OUTPUTS.get(task_id)
        if spool is None:
            spool = TaskOutputSpool(task_id)
            _TASK_OUTPUTS[task_id] = spool
        return spool


def append_task_output(
    task_id: str,
    text: str,
    *,
    summary: str | None = None,
) -> dict[str, Any]:
    """Append output to the training spool and return the bounded snapshot."""
    spool = get_task_output_spool(task_id)
    return spool.append(text, summary=summary)


def apply_output_snapshot(record: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Attach a bounded output snapshot to a task record."""
    updated = dict(record)
    updated.update(snapshot)
    return updated
