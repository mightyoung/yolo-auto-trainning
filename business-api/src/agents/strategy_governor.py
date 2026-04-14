"""Bounded autonomous-governor helpers for Business-side single-writer commits."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


DEFAULT_STRATEGY_BUDGET = 10
DEFAULT_LEASE_TTL_MS = 900_000

_COMMIT_GATE_LUA = """
local hash_key = KEYS[1]
local proposal_id = ARGV[1]
local budget_limit = tonumber(ARGV[2])
local record_json = ARGV[3]
local child_training_task_id = ARGV[4]
local proposal_field = "proposal:" .. proposal_id
local terminal = redis.call("HGET", hash_key, "terminal")
if terminal == "1" then
  return cjson.encode({ok=false, error="terminal"})
end
if redis.call("HEXISTS", hash_key, proposal_field) == 1 then
  local existing = redis.call("HGET", hash_key, proposal_field)
  return cjson.encode({ok=false, error="duplicate", record=existing})
end
local commit_count = tonumber(redis.call("HGET", hash_key, "strategy_commit_count") or "0")
if commit_count >= budget_limit then
  return cjson.encode({ok=false, error="budget_exhausted", count=commit_count})
end
local sequence = tonumber(redis.call("HGET", hash_key, "strategy_sequence") or "0") + 1
local new_count = commit_count + 1
redis.call("HSET", hash_key,
  "strategy_commit_count", tostring(new_count),
  "strategy_sequence", tostring(sequence),
  "current_child_training_task_id", child_training_task_id,
  proposal_field, record_json
)
return cjson.encode({
  ok=true,
  sequence=sequence,
  commit_count=new_count,
  child_training_task_id=child_training_task_id
})
"""


def _governor_hash(parent_run_id: str) -> str:
    return f"governor:{parent_run_id}"


def _lease_key(parent_run_id: str) -> str:
    return f"lock:governor:{parent_run_id}"


def _proposal_field() -> str:
    return "pending_strategy_proposal"


def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _from_json(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def build_strategy_proposal(
    *,
    parent_run_id: str,
    child_training_task_id: str,
    decision: dict[str, Any],
    proposal_id: str,
) -> dict[str, Any]:
    """Build a canonical strategy proposal payload."""
    selected = dict(decision.get("selected") or {})
    signal = dict(selected.get("details", {}).get("signal") or {})
    return {
        "proposal_id": proposal_id,
        "parent_run_id": parent_run_id,
        "child_training_task_id": child_training_task_id,
        "action": selected.get("action"),
        "stage": selected.get("stage"),
        "rationale": decision.get("rationale", ""),
        "trigger_signal": signal,
        "decision": selected.get("action"),
        "change_set": {
            "action": selected.get("action"),
            "stage": selected.get("stage"),
            "score": selected.get("score"),
            "details": selected.get("details") or {},
        },
        "timestamp": datetime.now().isoformat(),
    }


def store_pending_strategy_proposal(redis_client, parent_run_id: str, proposal: dict[str, Any]) -> None:
    """Persist the newest proposal for the parent run."""
    if redis_client is None:
        return
    payload = _to_json(proposal)
    redis_client.hset(
        f"agent:{parent_run_id}",
        mapping={_proposal_field(): payload},
    )
    redis_client.hset(
        f"autoadjust:{parent_run_id}",
        mapping={_proposal_field(): payload},
    )


def get_pending_strategy_proposal(redis_client, parent_run_id: str) -> dict[str, Any] | None:
    """Load the pending proposal if present."""
    if redis_client is None:
        return None
    return _from_json(redis_client.hget(f"agent:{parent_run_id}", _proposal_field()))


def clear_pending_strategy_proposal(redis_client, parent_run_id: str, proposal_id: str | None = None) -> None:
    """Delete the pending proposal once it is committed or rejected."""
    if redis_client is None:
        return
    current = get_pending_strategy_proposal(redis_client, parent_run_id)
    if proposal_id and current and current.get("proposal_id") != proposal_id:
        return
    redis_client.hdel(f"agent:{parent_run_id}", _proposal_field())
    redis_client.hdel(f"autoadjust:{parent_run_id}", _proposal_field())


def acquire_governor_lease(redis_client, parent_run_id: str, owner: str, ttl_ms: int = DEFAULT_LEASE_TTL_MS) -> bool:
    """Acquire the single-writer lease for a parent run."""
    if redis_client is None:
        return False
    return bool(redis_client.set(_lease_key(parent_run_id), owner, nx=True, px=ttl_ms))


def refresh_governor_lease(redis_client, parent_run_id: str, owner: str, ttl_ms: int = DEFAULT_LEASE_TTL_MS) -> bool:
    """Refresh the single-writer lease only if still owned."""
    if redis_client is None:
        return False
    key = _lease_key(parent_run_id)
    if redis_client.get(key) != owner:
        return False
    return bool(redis_client.pexpire(key, ttl_ms))


def release_governor_lease(redis_client, parent_run_id: str, owner: str) -> None:
    """Release the lease if the caller still owns it."""
    if redis_client is None:
        return
    key = _lease_key(parent_run_id)
    if redis_client.get(key) == owner:
        redis_client.delete(key)


def commit_strategy_proposal(
    redis_client,
    *,
    parent_run_id: str,
    proposal: dict[str, Any],
    child_training_task_id: str,
    budget_limit: int = DEFAULT_STRATEGY_BUDGET,
) -> dict[str, Any]:
    """Run the atomic Lua budget gate and persist a pending commit record."""
    proposal_id = proposal["proposal_id"]
    record = {
        **proposal,
        "commit_id": proposal_id,
        "status": "pending_effects",
        "sequence": None,
        "child_training_task_id": child_training_task_id,
    }
    raw = redis_client.eval(
        _COMMIT_GATE_LUA,
        1,
        _governor_hash(parent_run_id),
        proposal_id,
        str(budget_limit),
        _to_json(record),
        child_training_task_id,
    )
    result = _from_json(raw) or {"ok": False, "error": "invalid_gate_response"}
    if result.get("ok"):
        record["sequence"] = result["sequence"]
        record["status"] = "pending_effects"
        redis_client.hset(
            _governor_hash(parent_run_id),
            mapping={f"proposal:{proposal_id}": _to_json(record)},
        )
        result["record"] = record
    elif result.get("record"):
        result["record"] = _from_json(result["record"])
    return result


def finalize_commit_record(
    redis_client,
    *,
    parent_run_id: str,
    proposal_id: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Update the durable commit record after side effects finish."""
    if redis_client is None:
        return None
    key = _governor_hash(parent_run_id)
    field = f"proposal:{proposal_id}"
    record = _from_json(redis_client.hget(key, field))
    if record is None:
        return None
    record["status"] = status
    record["updated_at"] = datetime.now().isoformat()
    if extra:
        record.update(extra)
    redis_client.hset(key, mapping={field: _to_json(record)})
    return record


def allocate_strategy_sequence(redis_client, parent_run_id: str) -> int:
    """Reserve the next sequence number for non-commit terminal strategy events."""
    return int(redis_client.hincrby(_governor_hash(parent_run_id), "strategy_sequence", 1))


def recover_pending_commit_records(redis_client, parent_run_id: str) -> list[dict[str, Any]]:
    """List still-pending commit records for recovery/fail-closed handling."""
    if redis_client is None:
        return []
    records: list[dict[str, Any]] = []
    for field, value in redis_client.hgetall(_governor_hash(parent_run_id)).items():
        if not field.startswith("proposal:"):
            continue
        record = _from_json(value)
        if record and record.get("status") == "pending_effects":
            records.append(record)
    records.sort(key=lambda item: (item.get("sequence") or 0, item.get("timestamp") or ""))
    return records


def mark_governor_terminal(redis_client, parent_run_id: str, reason: str) -> None:
    """Prevent further commits for a terminal parent run."""
    if redis_client is None:
        return
    redis_client.hset(
        _governor_hash(parent_run_id),
        mapping={
            "terminal": "1",
            "terminal_reason": reason,
            "terminal_at": datetime.now().isoformat(),
        },
    )
