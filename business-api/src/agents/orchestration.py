"""
CrewAI Agents - Multi-agent orchestration for YOLO training system.

Based on CrewAI best practices:
- https://docs.crewai.com/en/concepts/processes

Uses lazy imports for crewai so the module loads even if crewai is not installed.
When unavailable, falls back to direct DatasetDiscovery.

This module is a thin facade that re-exports from specialized modules:
- tools.py: get_llm() and tool classes
- agent_factories.py: CrewAI agent factory functions
- ssh_ops.py: SSH operations for GPU server interactions
- auto_adjust_agent.py: AutoAdjustAgent for plateau handling
"""

import os
import sys
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import path setup
_project_root = Path(__file__).parent.parent.parent.parent  # project root (contains src/)
_biz_api_root = Path(__file__).parent.parent  # business-api/src/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_biz_api_root) not in sys.path:
    sys.path.insert(0, str(_biz_api_root))

# Re-export tools and factories for backward compatibility
from .tools import (
    get_llm,
    DatasetSearchTool,
    DatasetDownloadTool,
    TrainModelTool,
    ExportModelTool,
)
from .agent_factories import (
    create_dataset_discovery_agent,
    create_training_agent,
    create_deployment_agent,
)
from .ssh_ops import (
    check_dataset_exists,
    download_dataset_ssh,
    download_coco_builtin_ssh,
    generate_data_yaml_ssh,
)
from .auto_adjust_agent import AutoAdjustAgent
from .coordinator_summary import build_compact_summary, summarize_attempt, summarize_tool_batch
from .operation_policy import require_operation_allowed
from .task_output import append_agent_output
from .worker_memory import (
    append_agent_attempt,
    build_attempt_record,
    sanitize_training_status,
)
from .strategy_governor import (
    DEFAULT_STRATEGY_BUDGET,
    acquire_governor_lease,
    allocate_strategy_sequence,
    clear_pending_strategy_proposal,
    commit_strategy_proposal,
    finalize_commit_record,
    get_pending_strategy_proposal,
    mark_governor_terminal,
    recover_pending_commit_records,
    refresh_governor_lease,
    release_governor_lease,
)
try:
    from ..api.task_registry import append_task_event
except ImportError:  # pragma: no cover - compatibility with direct package imports
    from api.task_registry import append_task_event

# Lazy import for optional crewai dependency
CREWAI_AVAILABLE = False
_Agent = _Task = _Crew = _Process = _BaseTool = _LLM = None


def _try_import_crewai():
    global CREWAI_AVAILABLE, _Agent, _Task, _Crew, _Process, _BaseTool, _LLM
    if CREWAI_AVAILABLE:
        return True
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import BaseTool
        from crewai.llm import LLM
        _Agent = Agent
        _Task = Task
        _Crew = Crew
        _Process = Process
        _BaseTool = BaseTool
        _LLM = LLM
        CREWAI_AVAILABLE = True
        return True
    except ImportError:
        CREWAI_AVAILABLE = False
        return False


# Try importing crewai now (will succeed if installed)
_try_import_crewai()

from src.data.discovery import DatasetDiscovery, DatasetInfo
from .gpu_scheduler import start_scheduler


class YOLOTrainingOrchestrator:
    """Orchestrates CrewAI + Pipeline execution with HiTL confirmation gates."""

    def __init__(self):
        # Don't init LLM here - only init when crewai is available
        pass

    def _get_redis(self):
        try:
            from src.api.redis_client import get_redis_client
            return get_redis_client()
        except ImportError:
            import redis
            return redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=0,
                password=os.getenv("REDIS_PASSWORD"),  # No default - must be configured
                decode_responses=True,
            )

    def run_phase1(self, task_description: str, user_id: str, task_id: str) -> None:
        """Phase 1: Run dataset discovery agent, then await human confirmation."""
        r = self._get_redis()
        r.hset(f"agent:{task_id}", mapping={
            "status": "running", "user_id": user_id,
            "task_description": task_description,
            "progress": "10.0", "current_agent": "Dataset Curator",
            "created_at": datetime.now().isoformat(),
        })

        try:
            discovery = DatasetDiscovery()
            results = discovery.search(query=task_description, max_results=5)

            if not results:
                # Fallback: curated fire/smoke datasets (no API key configured)
                results = [
                    DatasetInfo(source="roboflow",
                        name="fire-and-smoke-dataset",
                        url="https://universe.roboflow.com/workspace-fwkuns/fire-and-smoke-dataset",
                        license="CC BY 4.0", annotations="bounding-box", images=2800,
                        categories=["fire","smoke"], relevance_score=0.92),
                    DatasetInfo(source="roboflow",
                        name="fire-detection-ymonk",
                        url="https://universe.roboflow.com/ymonk/fire-detection-ymonk",
                        license="CC BY 4.0", annotations="bounding-box", images=1200,
                        categories=["fire"], relevance_score=0.85),
                    DatasetInfo(source="roboflow",
                        name="roboflow-universe/fire-detection",
                        url="https://universe.roboflow.com/roboflow-universe/fire-detection",
                        license="CC BY 4.0", annotations="bounding-box", images=5600,
                        categories=["fire","smoke"], relevance_score=0.88),
                    DatasetInfo(source="roboflow",
                        name="forest-fire-detection",
                        url="https://universe.roboflow.com/workspace-fwkuns/forest-fire-detection",
                        license="CC BY 4.0", annotations="bounding-box", images=3800,
                        categories=["fire","smoke"], relevance_score=0.80),
                    # COCO Person built-in fallback: no API key needed
                    DatasetInfo(source="coco_builtin",
                        name="COCO-Person-BuiltIn",
                        url="https://cocodataset.org",
                        license="CC BY 4.0",
                        annotations="coco",
                        images=0,
                        categories=["person"],
                        relevance_score=0.95),
                ]

            # Build result string for display
            lines = [f"Found {len(results)} datasets:"]
            for ds in results:
                lines.append(f"  - {ds.name} ({ds.source})")
                lines.append(f"    Relevance: {ds.relevance_score:.2f}, Images: {ds.images}, URL: {ds.url}")

            # Pick best dataset as recommendation
            best = max(results, key=lambda d: d.relevance_score)
            lines.append(f"\nRecommended: {best.name} (score={best.relevance_score:.2f}) from {best.source}")

            if CREWAI_AVAILABLE:
                lines.append(f"\n[CrewAI available - full agentic pipeline enabled]")
            else:
                lines.append(f"\n[CrewAI unavailable - using direct discovery fallback]")

            result_str = "\n".join(lines)
            append_agent_output(
                r,
                task_id,
                result_str,
                summary=summarize_tool_batch(
                    "dataset discovery",
                    outcome="completed",
                    detail=f"{len(results)} candidates",
                    subject=best.name,
                ),
            )

            r.hset(f"agent:{task_id}", mapping={
                "status": "awaiting_confirmation",
                "current_agent": "Dataset Curator",
                "progress": "30.0",
                "phase1_result": result_str,
                "confirmed_running": "false",
            })

            # Autonomous mode: bypass human gate and auto-confirm Phase 1
            r_check = self._get_redis()
            if r_check.hget(f"agent:{task_id}", "auto_confirm") == "true":
                # Set Phase 1 confirmation overrides so run_phase2 reads the right dataset
                if best.source == "coco_builtin":
                    dataset_path = "/home/wangxin/data/coco_person"
                else:
                    dataset_path = f"/home/wangxin/data/{best.name.replace('/', '_').replace('.', '_')}"
                default_overrides = {
                    "dataset_name": best.name,
                    "dataset_path": dataset_path,
                    "source": best.source,
                    "user_id": user_id,
                }
                r.hset(f"agent:{task_id}", mapping={
                    "overrides_awaiting_confirmation": json.dumps(default_overrides),
                    "status": "running",
                    "confirmed_running": "true",
                    "progress": "32.0",
                })
                # Chain to phase2 in background thread
                def _auto_phase2():
                    try:
                        import time
                        time.sleep(1)
                        orch2 = YOLOTrainingOrchestrator()
                        orch2.run_phase2(task_id, user_id)
                    except Exception as e:
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "failed", "error": f"Auto Phase2 error: {e}",
                        })
                t = threading.Thread(target=_auto_phase2, daemon=True)
                t.start()
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            })

    def run_phase2(self, task_id: str, user_id: str) -> None:
        """Phase 2: Download dataset + generate data.yaml, then await training param confirmation."""
        r = self._get_redis()

        # Read Phase 1 confirmation overrides (dataset choice)
        overrides_json = r.hget(f"agent:{task_id}", "overrides_awaiting_confirmation")
        overrides = json.loads(overrides_json) if overrides_json else {}

        dataset_name = overrides.get("dataset_name", "workspace-fwkuns/fire-and-smoke-dataset")
        dataset_path = overrides.get("dataset_path", "/home/wangxin/data/fire-smoke")
        source = overrides.get("source", "roboflow")

        # Update status: downloading
        r.hset(f"agent:{task_id}", mapping={
            "status": "downloading_dataset",
            "current_agent": "Data Engineer",
            "progress": "35.0",
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "source": source,
            "confirmed_training": "false",
        })

        try:
            # Check if dataset already exists with images
            require_operation_allowed(
                "ssh_dataset_check",
                context={"dataset_name": dataset_name, "dataset_path": dataset_path, "source": source},
            )
            skip_download = check_dataset_exists(dataset_path, source)

            if not skip_download:
                # Download dataset to GPU server via SSH
                require_operation_allowed(
                    "dataset_download",
                    context={"dataset_name": dataset_name, "dataset_path": dataset_path, "source": source},
                )
                if source == "coco_builtin":
                    download_coco_builtin_ssh(dataset_path)
                else:
                    download_dataset_ssh(dataset_name, dataset_path, source)

            # Generate data.yaml on GPU server (if missing or stale)
            require_operation_allowed("ssh_dataset_yaml", context={"dataset_path": dataset_path})
            generate_data_yaml_ssh(dataset_path)
            append_agent_output(
                r,
                task_id,
                f"Dataset ready: {dataset_name} -> {dataset_path}",
                summary=summarize_tool_batch(
                    "dataset preparation",
                    outcome="completed",
                    detail="download skipped" if skip_download else "downloaded",
                    subject=dataset_name,
                ),
            )

            r.hset(f"agent:{task_id}", mapping={
                "status": "awaiting_training_confirmation",
                "current_agent": "ML Engineer",
                "progress": "50.0",
                "download_status": "completed" if not skip_download else "skipped_existing",
                "confirmed_training": "false",
            })

            # Autonomous mode: bypass training confirmation gate and auto-chain to Phase 3
            r_check = self._get_redis()
            if r_check.hget(f"agent:{task_id}", "auto_confirm") == "true":
                r.hset(f"agent:{task_id}", mapping={
                    "status": "running",
                    "confirmed_training": "true",
                    "progress": "52.0",
                })
                def _auto_phase3():
                    try:
                        import time
                        time.sleep(1)
                        orch3 = YOLOTrainingOrchestrator()
                        orch3.run_phase3(task_id, "auto")
                    except Exception as e:
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "failed", "error": f"Auto Phase3 error: {e}",
                        })
                t = threading.Thread(target=_auto_phase3, daemon=True)
                t.start()
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Dataset download failed: {e}",
                "completed_at": datetime.now().isoformat(),
            })

    def run_phase3(self, task_id: str, user_id: str) -> None:
        """Phase 3: Submit actual training job to GPU server."""
        r = self._get_redis()
        data = r.hgetall(f"agent:{task_id}")
        confirmed_training = data.get("confirmed_training") == "true"

        overrides_json = r.hget(f"agent:{task_id}", "overrides_training_confirmation")
        overrides = json.loads(overrides_json) if overrides_json else {}

        model = overrides.get("model", "yolo11x")     # Upgraded: yolo11n → yolo11x for mAP target
        epochs = overrides.get("epochs", 200)          # Upgraded: 50 → 200 for mAP target
        imgsz = overrides.get("imgsz", 1280)          # Upgraded: 640 → 1280 for small object detection
        batch = overrides.get("batch", 8)            # Halved from 16 due to 1280px images
        device = overrides.get("device", "cuda:0")
        augmentation_preset = overrides.get("augmentation_preset", "strong")  # Strong for 90%+ mAP
        resume_from = overrides.get("resume_from", None)

        r.hset(f"agent:{task_id}", mapping={
            "status": "training",
            "current_agent": "ML Engineer",
            "progress": "55.0",
            "training_model": model,
            "training_epochs": str(epochs),
            "training_imgsz": str(imgsz),
            "submission": json.dumps({
                "model": model,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": device,
                "augmentation_preset": augmentation_preset,
                "resume_from": resume_from,
                "data_yaml": data.get("dataset_path", "/home/wangxin/data/fire-smoke") + "/data.yaml",
            }),
        })

        try:
            require_operation_allowed(
                "gpu_training_submit",
                context={"task_id": task_id, "device": device, "model": model},
            )
            from src.api.training_client import TrainingAPIClient
            client = TrainingAPIClient(
                base_url=os.getenv("TRAINING_API_URL", "http://localhost:8001"),
                api_key=os.getenv("TRAINING_API_KEY", ""),
            )

            # Use curriculum (3-stage progressive) training for HiTL
            stage1 = {
                "name": "rapid_validation",
                "epochs": 50,
                "imgsz": 640,
                "batch": 16,
                "model": "yolo11m",
                "augmentation_preset": "balanced",
            }
            stage2 = {
                "name": "deep_training",
                "epochs": 150,
                "imgsz": 1280,
                "batch": 8,
                "model": "yolo11x",
                "augmentation_preset": "strong",
                "mosaic": 1.0,
                "mixup": 0.3,
                "copy_paste": 0.4,
                "degrees": 15.0,
                "translate": 0.2,
                "scale": 0.7,
                "close_mosaic": 15,
            }
            stage3 = {
                "name": "fine_tuning",
                "epochs": 100,
                "imgsz": 1280,
                "batch": 8,
                "model": "yolo11x",
                "augmentation_preset": "strong",
                "mosaic": 0.0,       # Disable mosaic for fine detail
                "mixup": 0.1,
                "copy_paste": 0.1,
                "degrees": 5.0,
                "translate": 0.1,
                "scale": 0.5,
                "close_mosaic": 100,  # Mosaic disabled entirely
            }

            # dataset_path is a directory; curriculum needs the actual YAML file path
            import posixpath
            _dataset_dir = data.get("dataset_path", "/home/wangxin/data/fire-smoke")
            _yaml_path = posixpath.join(_dataset_dir, "data.yaml")
            result = client.start_curriculum_sync(
                task_id=task_id,
                data_yaml=_yaml_path,
                output_dir="/home/wangxin/runs",
                device=device,
                auto_export=True,
                stage1_min_map=0.50,
                stage2_target_map=0.90,
                stage2_min_for_stage3=0.80,
                stage1_overrides=stage1,
                stage2_overrides=stage2,
                stage3_overrides=stage3,
            )
            training_task_id = result.get("task_id", task_id)
            r.hset(f"agent:{task_id}", mapping={
                "status": "training",
                "training_type": "curriculum",
                "progress": "10.0",  # Stage 1 just started
                "training_task_id": training_task_id,
            })
            event_state = append_task_event(
                r.hgetall(f"agent:{task_id}"),
                source=task_id,
                target=training_task_id,
                relation="training_started",
                node_type="task",
                label="training task",
                metadata={"training_type": "curriculum"},
            )
            r.hset(f"agent:{task_id}", mapping={"event_graph": json.dumps(event_state.get("event_graph", {}))})
            append_agent_output(
                r,
                task_id,
                f"Training submitted: task_id={training_task_id}, model={model}, epochs={epochs}, imgsz={imgsz}",
                summary=summarize_tool_batch(
                    "training submission",
                    outcome="completed",
                    detail=f"{training_task_id} {model}",
                    subject=task_id,
                ),
            )

            # Poll Training API for completion (runs in background thread)
            self._poll_training(task_id, training_task_id, client)

            # Start the GPU task queue scheduler for autonomous dispatch
            require_operation_allowed("gpu_scheduler_start", context={"task_id": task_id})
            start_scheduler()
        except Exception as e:
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Training submission failed: {e}",
                "completed_at": datetime.now().isoformat(),
            })
            append_agent_output(
                r,
                task_id,
                f"Training submission failed: {e}",
                summary=summarize_attempt(
                    "training",
                    "submission",
                    "failed",
                    action="start_curriculum_sync",
                    error=str(e),
                ),
            )

    def _build_governor_child_id(self, task_id: str, proposal_id: str) -> str:
        """Build a deterministic child training task id for a committed strategy change."""
        return f"train_{task_id[:8]}_{proposal_id[:8]}"

    def _record_strategy_event(
        self,
        redis_client,
        task_id: str,
        *,
        relation: str,
        training_task_id: str,
        label: str,
        metadata: dict,
    ) -> None:
        """Append canonical strategy events and mirror durable ledger into agent state."""
        event_state = append_task_event(
            redis_client.hgetall(f"agent:{task_id}"),
            source=training_task_id,
            target=task_id,
            relation=relation,
            node_type="strategy",
            label=label,
            metadata=metadata,
        )
        redis_client.hset(
            f"agent:{task_id}",
            mapping={
                "event_graph": json.dumps(event_state.get("event_graph", {})),
                "strategy_ledger": json.dumps(event_state.get("strategy_ledger", [])),
            },
        )

    def _record_strategy_stop(
        self,
        redis_client,
        task_id: str,
        *,
        training_task_id: str,
        label: str,
        stop_reason: str,
        proposal: dict | None = None,
    ) -> None:
        """Append the final durable stop event required by the approved plan."""
        sequence = allocate_strategy_sequence(redis_client, task_id)
        metadata = {
            "proposal_id": (proposal or {}).get("proposal_id"),
            "commit_id": (proposal or {}).get("commit_id") or (proposal or {}).get("proposal_id"),
            "parent_run_id": task_id,
            "child_training_task_id": (proposal or {}).get("child_training_task_id") or training_task_id,
            "sequence": sequence,
            "trigger_signal": (proposal or {}).get("trigger_signal") or {},
            "rationale": (proposal or {}).get("rationale", ""),
            "change_set": (proposal or {}).get("change_set") or {},
            "decision": "stopped",
            "stop_reason": stop_reason,
            "timestamp": datetime.now().isoformat(),
        }
        self._record_strategy_event(
            redis_client,
            task_id,
            relation="strategy_stop",
            training_task_id=training_task_id,
            label=label,
            metadata=metadata,
        )

    def _recover_pending_governor_effects(self, redis_client, task_id: str) -> None:
        """Fail closed if the parent run has stranded pending commit records."""
        pending_records = recover_pending_commit_records(redis_client, task_id)
        if not pending_records:
            return
        record = pending_records[0]
        finalize_commit_record(
            redis_client,
            parent_run_id=task_id,
            proposal_id=record["proposal_id"],
            status="finalized",
            extra={"decision": "fail_closed", "stop_reason": "pending_effects_recovery"},
        )
        self._record_strategy_event(
            redis_client,
            task_id,
            relation="strategy_stop",
            training_task_id=record.get("child_training_task_id") or task_id,
            label="pending_effects_recovery",
            metadata={
                **record,
                "sequence": allocate_strategy_sequence(redis_client, task_id),
                "decision": "fail_closed",
                "stop_reason": "pending_effects_recovery",
                "timestamp": datetime.now().isoformat(),
            },
        )
        redis_client.hset(
            f"agent:{task_id}",
            mapping={
                "status": "failed",
                "error": "Governor recovery found pending_effects and failed closed",
                "completed_at": datetime.now().isoformat(),
            },
        )
        mark_governor_terminal(redis_client, task_id, "pending_effects_recovery")

    def _commit_strategy_action(
        self,
        *,
        task_id: str,
        training_task_id: str,
        proposal: dict,
        auto_adjust_agent: AutoAdjustAgent,
        client,
        redis_client,
        lease_owner: str,
    ) -> str | None:
        """Commit a pending strategy proposal as the single Business-side writer."""
        action = proposal.get("action")
        proposal_id = proposal["proposal_id"]
        next_training_task_id = self._build_governor_child_id(task_id, proposal_id)
        gate = commit_strategy_proposal(
            redis_client,
            parent_run_id=task_id,
            proposal=proposal,
            child_training_task_id=next_training_task_id,
            budget_limit=DEFAULT_STRATEGY_BUDGET,
        )
        if not gate.get("ok"):
            reason = gate.get("error", "rejected")
            if reason in {"budget_exhausted", "terminal"}:
                self._record_strategy_stop(
                    redis_client,
                    task_id,
                    training_task_id=training_task_id,
                    label=action or "stopped",
                    stop_reason=reason,
                    proposal={**proposal, "commit_id": proposal_id},
                )
                mark_governor_terminal(redis_client, task_id, reason)
                redis_client.hset(
                    f"agent:{task_id}",
                    mapping={
                        "status": "failed",
                        "error": f"Governor stopped strategy changes: {reason}",
                        "completed_at": datetime.now().isoformat(),
                    },
                )
            clear_pending_strategy_proposal(redis_client, task_id, proposal_id)
            return None

        commit_record = gate["record"]
        commit_metadata = {
            **proposal,
            "commit_id": proposal_id,
            "sequence": commit_record["sequence"],
            "child_training_task_id": next_training_task_id,
            "decision": action,
            "timestamp": datetime.now().isoformat(),
        }
        self._record_strategy_event(
            redis_client,
            task_id,
            relation="strategy_change_committed",
            training_task_id=training_task_id,
            label=action or "committed",
            metadata=commit_metadata,
        )
        redis_client.hset(
            f"agent:{task_id}",
            mapping={
                "strategy_commit_count": str(gate["commit_count"]),
                "strategy_sequence": str(commit_record["sequence"]),
                "current_child_training_task_id": next_training_task_id,
                "latest_strategy_commit": json.dumps(commit_metadata),
            },
        )
        try:
            client.cancel_task_sync(training_task_id)
            self._record_strategy_event(
                redis_client,
                task_id,
                relation="training_cancel_requested",
                training_task_id=training_task_id,
                label=action or "cancel",
                metadata=commit_metadata,
            )

            def _lease_refresh() -> None:
                if not refresh_governor_lease(redis_client, task_id, lease_owner, DEFAULT_LEASE_TTL_MS):
                    raise RuntimeError("Governor lease lost during side effects")

            if action == "lr_decay":
                result = auto_adjust_agent._trigger_lr_adjustment(
                    lr_decay_signal=proposal.get("trigger_signal") or {},
                    decay_count=((proposal.get("trigger_signal") or {}).get("lr_decay_count") or 1),
                    base_url=client.base_url,
                    headers=client._get_headers(),
                    r=redis_client,
                    new_task_id=next_training_task_id,
                    proposal=commit_metadata,
                    raise_on_error=True,
                    lease_refresh=_lease_refresh,
                )
            elif action == "data_expansion":
                result = auto_adjust_agent._trigger_data_expansion(
                    data_expansion_signal=proposal.get("trigger_signal") or {},
                    base_url=client.base_url,
                    headers=client._get_headers(),
                    r=redis_client,
                    new_task_id=next_training_task_id,
                    proposal=commit_metadata,
                    raise_on_error=True,
                    lease_refresh=_lease_refresh,
                )
            else:
                raise RuntimeError(f"Unsupported strategy action: {action}")
            if not result:
                raise RuntimeError(f"side_effect_failed: empty result for {action}")
        except Exception as exc:
            finalize_commit_record(
                redis_client,
                parent_run_id=task_id,
                proposal_id=proposal_id,
                status="finalized",
                extra={"decision": "fail_closed", "error": str(exc)},
            )
            self._record_strategy_stop(
                redis_client,
                task_id,
                training_task_id=training_task_id,
                label=action or "failed",
                stop_reason="side_effect_failed",
                proposal=commit_metadata,
            )
            mark_governor_terminal(redis_client, task_id, "side_effect_failed")
            clear_pending_strategy_proposal(redis_client, task_id, proposal_id)
            raise RuntimeError(f"side_effect_failed: {exc}") from exc

        finalize_commit_record(
            redis_client,
            parent_run_id=task_id,
            proposal_id=proposal_id,
            status="effects_done",
            extra={
                "decision": action,
                "child_training_task_id": (result or {}).get("new_training_task_id", next_training_task_id),
            },
        )
        finalize_commit_record(
            redis_client,
            parent_run_id=task_id,
            proposal_id=proposal_id,
            status="finalized",
            extra={
                "decision": action,
                "child_training_task_id": (result or {}).get("new_training_task_id", next_training_task_id),
            },
        )
        clear_pending_strategy_proposal(redis_client, task_id, proposal_id)
        effective_training_task_id = (result or {}).get("new_training_task_id", next_training_task_id)
        auto_adjust_agent.update_training_task_id(effective_training_task_id)
        return effective_training_task_id

    def _poll_training(self, task_id: str, training_task_id: str, client) -> None:
        """Poll Training API for training completion and update Redis.

        Dynamically reacts to plateau signals from the Training API.
        """
        import time
        max_wait = 7200  # 2 hours max
        start = time.time()
        r = self._get_redis()
        lease_owner = f"{task_id}:{uuid.uuid4().hex[:8]}"
        _expansion_notified = False  # Track if we've already triggered dataset search for this request

        if not acquire_governor_lease(r, task_id, lease_owner):
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": "Governor lease acquisition failed",
                "completed_at": datetime.now().isoformat(),
            })
            return

        self._recover_pending_governor_effects(r, task_id)

        # Spawn AutoAdjustAgent for proposal-only plateau detection
        auto_adjust_agent = AutoAdjustAgent(task_id=task_id, training_task_id=training_task_id, proposal_only=True)
        auto_adjust_agent.start()

        try:
            while time.time() - start < max_wait:
                time.sleep(30)
                try:
                    if not refresh_governor_lease(r, task_id, lease_owner):
                        raise RuntimeError("Governor lease lost during poll loop")

                    # Use sync client to avoid asyncio.run() in thread
                    status_data = client.get_task_status_sync(training_task_id)
                    status_summary = sanitize_training_status(status_data)
                    status = status_data.get("status", "unknown")
                    progress = status_data.get("progress", 60)

                    # Extract plateau signals
                    live_mAP50 = status_data.get("live_mAP50")
                    lr_decay_triggered = status_data.get("lr_decay_triggered", False)
                    lr_decay_signal = status_data.get("lr_decay_signal") or {}
                    augment_boost_active = status_data.get("augment_boost_active", False)
                    augment_boost_signal = status_data.get("augment_boost_signal")
                    data_expansion_requested = status_data.get("data_expansion_requested", False)
                    data_expansion_signal = status_data.get("data_expansion_signal") or {}
                    strategies_triggered = status_data.get("strategies_triggered") or []
                    # Curriculum-specific fields
                    curriculum_stage = status_data.get("curriculum_stage", "")
                    curriculum_stage_mAP = status_data.get("curriculum_stage_mAP")
                    curriculum_history = status_data.get("curriculum_stage_history") or []

                    # Build mapping with plateau info + curriculum info
                    redis_mapping = {
                        "status": "training",
                        "progress": str(progress),
                        "training_status": status,
                        "training_summary": json.dumps(status_summary),
                        "live_mAP50": str(live_mAP50) if live_mAP50 is not None else "",
                        "lr_decay_triggered": str(lr_decay_triggered),
                        "augment_boost_active": str(augment_boost_active),
                        "data_expansion_requested": str(data_expansion_requested),
                        "strategies_triggered": str(strategies_triggered),
                        "curriculum_stage": curriculum_stage,
                        "curriculum_stage_mAP": str(curriculum_stage_mAP) if curriculum_stage_mAP is not None else "",
                        "curriculum_stage_history": str(curriculum_history),
                    }

                    # React to LR decay signal
                    if lr_decay_triggered and lr_decay_signal:
                        count = lr_decay_signal.get("lr_decay_count", "?")
                        factor = lr_decay_signal.get("factor", 0.5)
                        epoch = lr_decay_signal.get("epoch", "?")
                        mAP = lr_decay_signal.get("current_mAP50", 0)
                        print(f"[PLATEAU ALERT] Task {task_id}: LR decay #{count} triggered "
                              f"at epoch {epoch}, mAP50={mAP:.4f}, factor={factor}")

                    # React to data expansion request
                    if data_expansion_requested and not _expansion_notified:
                        _expansion_notified = True
                        rec = data_expansion_signal.get("recommendation", "")
                        print(f"[PLATEAU ALERT] Task {task_id}: Data expansion triggered! "
                              f"mAP50={data_expansion_signal.get('current_mAP50', 0):.4f}, "
                              f"target={data_expansion_signal.get('target_mAP50', 0):.2f}")
                        print(f"[PLATEAU] Recommendation: {rec}")
                        redis_mapping["plateau_action"] = "dataset_search_triggered"
                        redis_mapping["expansion_recommendation"] = rec

                    # Log augmentation boost
                    if augment_boost_active and augment_boost_signal:
                        ep = augment_boost_signal.get("start_epoch", "?")
                        rem = status_data.get("augment_boost_remaining", "?")
                        mix = augment_boost_signal.get("mixup", "?")
                        cp = augment_boost_signal.get("copy_paste", "?")
                        print(f"[PLATEAU ALERT] Task {task_id}: Augmentation boost active "
                              f"at EP {ep}, remaining={rem}, mixup={mix}, copy_paste={cp}")

                    # Log curriculum stage progress
                    if curriculum_stage:
                        stage_mAP_str = f", mAP50={curriculum_stage_mAP:.4f}" if curriculum_stage_mAP else ""
                        print(f"[CURRICULUM] Task {task_id}: {curriculum_stage}{stage_mAP_str}, progress={progress:.1f}%")

                    pending_proposal = get_pending_strategy_proposal(r, task_id)
                    if pending_proposal and pending_proposal.get("child_training_task_id") == training_task_id:
                        new_training_task_id = self._commit_strategy_action(
                            task_id=task_id,
                            training_task_id=training_task_id,
                            proposal=pending_proposal,
                            auto_adjust_agent=auto_adjust_agent,
                            client=client,
                            redis_client=r,
                            lease_owner=lease_owner,
                        )
                        if new_training_task_id:
                            training_task_id = new_training_task_id
                            _expansion_notified = False
                            continue

                    r.hset(f"agent:{task_id}", mapping=redis_mapping)

                    if status in ("completed", "success"):
                        auto_adjust_agent.stop()
                        self._record_strategy_stop(
                            r,
                            task_id,
                            training_task_id=training_task_id,
                            label="training_completed",
                            stop_reason="training_completed",
                        )
                        mark_governor_terminal(r, task_id, "training_completed")
                        model_path = status_data.get("model_path", "/home/wangxin/runs/train/weights/best.pt")
                        append_agent_attempt(
                            r,
                            task_id,
                            build_attempt_record(
                                attempt_type="training_completion",
                                stage=curriculum_stage or "training",
                                outcome="completed",
                                source="training_poller",
                                action="auto_export",
                                training_task_id=training_task_id,
                                details=status_summary,
                            ),
                        )
                        event_state = append_task_event(
                            r.hgetall(f"agent:{task_id}"),
                            source=training_task_id,
                            target=task_id,
                            relation="training_completed",
                            node_type="training",
                            label="completed",
                            metadata={"status": "completed"},
                        )
                        r.hset(f"agent:{task_id}", mapping={"event_graph": json.dumps(event_state.get("event_graph", {}))})
                        agent_data = r.hgetall(f"agent:{task_id}")
                        project_name = agent_data.get("project_name", task_id)
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "training_completed",
                            "progress": "90.0",
                            "model_path": model_path,
                        })
                        append_agent_output(
                            r,
                            task_id,
                            f"Training completed: model_path={model_path}",
                            summary=summarize_attempt(
                                "training",
                                curriculum_stage or "training",
                                "completed",
                                action="auto_export",
                                detail=model_path,
                            ),
                        )
                        # Trigger auto export + deploy
                        self._auto_export_and_deploy(task_id, model_path, project_name)
                        return
                    elif status in ("failed", "error"):
                        auto_adjust_agent.stop()
                        self._record_strategy_stop(
                            r,
                            task_id,
                            training_task_id=training_task_id,
                            label="training_failed",
                            stop_reason="training_failed",
                        )
                        mark_governor_terminal(r, task_id, "training_failed")
                        append_agent_attempt(
                            r,
                            task_id,
                            build_attempt_record(
                                attempt_type="training_completion",
                                stage=curriculum_stage or "training",
                                outcome="failed",
                                source="training_poller",
                                error=status_data.get("error", status),
                                training_task_id=training_task_id,
                                details=status_summary,
                            ),
                        )
                        event_state = append_task_event(
                            r.hgetall(f"agent:{task_id}"),
                            source=training_task_id,
                            target=task_id,
                            relation="training_failed",
                            node_type="training",
                            label="failed",
                            metadata={"status": "failed"},
                        )
                        r.hset(f"agent:{task_id}", mapping={"event_graph": json.dumps(event_state.get("event_graph", {}))})
                        r.hset(f"agent:{task_id}", mapping={
                            "status": "failed",
                            "error": f"GPU training failed: {status_data.get('error', status)}",
                            "completed_at": datetime.now().isoformat(),
                        })
                        append_agent_output(
                            r,
                            task_id,
                            f"Training failed: {status_data.get('error', status)}",
                            summary=summarize_attempt(
                                "training",
                                curriculum_stage or "training",
                                "failed",
                                action="poll_training",
                                error=status_data.get("error", status),
                            ),
                        )
                        return
                except Exception as e:
                    if "Governor lease lost" in str(e) or "side_effect_failed" in str(e) or "Unsupported strategy action" in str(e):
                        raise
                    append_agent_attempt(
                        r,
                        task_id,
                        build_attempt_record(
                            attempt_type="training_poll",
                            stage="training",
                            outcome="error",
                            source="training_poller",
                            error=str(e),
                            training_task_id=training_task_id,
                        ),
                    )
                    r.hset(f"agent:{task_id}", mapping={
                        "training_poll_error": str(e),
                    })
                    append_agent_output(
                        r,
                        task_id,
                        f"Training poll error: {e}",
                        summary=summarize_attempt(
                            "training",
                            "poll",
                            "error",
                            action="status_check",
                            error=str(e),
                        ),
                    )

            # Timeout
            auto_adjust_agent.stop()
            self._record_strategy_stop(
                r,
                task_id,
                training_task_id=training_task_id,
                label="training_timeout",
                stop_reason="training_timeout",
            )
            mark_governor_terminal(r, task_id, "training_timeout")
            append_agent_attempt(
                r,
                task_id,
                build_attempt_record(
                    attempt_type="training_completion",
                    stage="training",
                    outcome="timeout",
                    source="training_poller",
                    error="Training timeout (>2h)",
                    training_task_id=training_task_id,
                ),
            )
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": "Training timeout (>2h)",
                "completed_at": datetime.now().isoformat(),
            })
            append_agent_output(
                r,
                task_id,
                "Training timeout after 2 hours",
                summary=summarize_attempt(
                    "training",
                    "poll",
                    "timeout",
                    action="status_check",
                    error="Training timeout (>2h)",
                ),
            )
        except Exception as e:
            # Top-level: prevent thread from crashing silently
            auto_adjust_agent.stop()
            self._record_strategy_stop(
                r,
                task_id,
                training_task_id=training_task_id,
                label="polling_crashed",
                stop_reason="polling_crashed",
            )
            mark_governor_terminal(r, task_id, "polling_crashed")
            append_agent_attempt(
                r,
                task_id,
                build_attempt_record(
                    attempt_type="training_poll",
                    stage="training",
                    outcome="crashed",
                    source="training_poller",
                    error=str(e),
                    training_task_id=training_task_id,
                ),
            )
            r.hset(f"agent:{task_id}", mapping={
                "status": "failed",
                "error": f"Polling crashed: {e}",
                "completed_at": datetime.now().isoformat(),
            })
            append_agent_output(
                r,
                task_id,
                f"Training polling crashed: {e}",
                summary=summarize_attempt(
                    "training",
                    "poll",
                    "crashed",
                    action="status_check",
                    error=str(e),
                ),
            )
        finally:
            release_governor_lease(r, task_id, lease_owner)

    def _auto_export_and_deploy(self, task_id: str, model_path: str, project_name: str) -> None:
        """Automatically export and deploy after training completes."""
        def _do_export_deploy():
            try:
                require_operation_allowed(
                    "model_export",
                    context={"task_id": task_id, "model_path": model_path, "project_name": project_name},
                )
                from src.api.training_client import TrainingAPIClient
                client = TrainingAPIClient(
                    base_url=os.getenv("TRAINING_API_URL", "http://localhost:8001"),
                    api_key=os.getenv("TRAINING_API_KEY", ""),
                )
                r = self._get_redis()

                export_task_id = f"{task_id}_export"
                print(f"[{task_id}] Auto-export triggered: model={model_path}")
                append_agent_output(
                    r,
                    task_id,
                    f"Auto-export triggered for {model_path}",
                    summary=summarize_tool_batch(
                        "auto export",
                        outcome="started",
                        detail=project_name,
                        subject=task_id,
                    ),
                )

                # Step 1: Submit export
                export_resp = client.start_export_sync(
                    task_id=export_task_id,
                    model_path=model_path,
                    platform="jetson_orin",
                    formats=["onnx"],
                    imgsz=640,
                )
                print(f"[{task_id}] Export submitted: {export_resp}")
                r.hset(f"agent:{task_id}", mapping={
                    "export_status": "submitted",
                    "export_task_id": export_task_id,
                })

                # Step 2: Poll export status (up to 5 minutes, 10s intervals)
                export_done = False
                for attempt in range(30):
                    import time
                    time.sleep(10)
                    try:
                        status_resp = client.get_export_status_sync(export_task_id)
                        export_status = status_resp.get("status", "unknown")
                        print(f"[{task_id}] Export status check {attempt+1}/30: {export_status}")
                        if export_status in ("completed", "exported", "done", "success"):
                            export_done = True
                            break
                        if export_status in ("failed", "error"):
                            print(f"[{task_id}] Export failed: {status_resp.get('error', export_status)}")
                            r.hset(f"agent:{task_id}", mapping={
                                "export_status": "failed",
                                "export_error": str(status_resp.get("error", export_status)),
                            })
                            append_agent_output(
                                r,
                                task_id,
                                f"Export failed: {status_resp.get('error', export_status)}",
                                summary=summarize_attempt(
                                    "export",
                                    "poll",
                                    "failed",
                                    action="get_export_status_sync",
                                    error=str(status_resp.get("error", export_status)),
                                ),
                            )
                            return
                    except Exception as poll_err:
                        print(f"[{task_id}] Export poll error: {poll_err}")

                if not export_done:
                    print(f"[{task_id}] Export timed out after 5 minutes")
                    r.hset(f"agent:{task_id}", mapping={"export_status": "timeout"})
                    append_agent_output(
                        r,
                        task_id,
                        "Export timed out after 5 minutes",
                        summary=summarize_attempt(
                            "export",
                            "poll",
                            "timeout",
                            action="get_export_status_sync",
                            error="Export timed out after 5 minutes",
                        ),
                    )
                    return

                print(f"[{task_id}] Export completed successfully")
                r.hset(f"agent:{task_id}", mapping={
                    "export_status": "completed",
                    "exported_model_path": status_resp.get("export_path", model_path),
                })
                append_agent_output(
                    r,
                    task_id,
                    f"Export completed: {status_resp.get('export_path', model_path)}",
                    summary=summarize_attempt(
                        "export",
                        "poll",
                        "completed",
                        action="start_export_sync",
                        detail=status_resp.get("export_path", model_path),
                    ),
                )

                # Step 3: Deploy
                try:
                    require_operation_allowed(
                        "gpu_training_submit",
                        context={"task_id": task_id, "platform": "jetson_orin"},
                    )
                    deploy_resp = client.submit_deployment(
                        model_path=model_path,
                        platform="jetson_orin",
                        imgsz=640,
                    )
                    print(f"[{task_id}] Auto-deploy submitted: {deploy_resp}")
                    r.hset(f"agent:{task_id}", mapping={
                        "deployment_status": "deployed",
                        "deploy_id": deploy_resp.get("deploy_id", ""),
                        "deploy_platform": deploy_resp.get("platform", "jetson_orin"),
                        "deployment_config": json.dumps(deploy_resp.get("config", {})),
                    })
                    append_agent_output(
                        r,
                        task_id,
                        f"Deployment submitted: {deploy_resp.get('deploy_id', '')}",
                        summary=summarize_tool_batch(
                            "deployment",
                            outcome="completed",
                            detail=deploy_resp.get("platform", "jetson_orin"),
                            subject=project_name,
                        ),
                    )
                except Exception as deploy_err:
                    print(f"[{task_id}] Auto-deploy failed: {deploy_err}")
                    r.hset(f"agent:{task_id}", mapping={
                        "deployment_status": "failed",
                        "deployment_error": str(deploy_err),
                    })
                    append_agent_output(
                        r,
                        task_id,
                        f"Deployment failed: {deploy_err}",
                        summary=summarize_attempt(
                            "deployment",
                            "deploy",
                            "failed",
                            action="submit_deployment",
                            error=str(deploy_err),
                        ),
                    )

            except Exception as e:
                print(f"[{task_id}] Auto-export/deploy failed: {e}")
                r.hset(f"agent:{task_id}", mapping={
                    "export_status": "failed",
                    "export_error": str(e),
                })
                append_agent_output(
                    r,
                    task_id,
                    f"Auto-export/deploy failed: {e}",
                    summary=summarize_attempt(
                        "export",
                        "deploy",
                        "failed",
                        action="start_export_sync",
                        error=str(e),
                    ),
                )

        import threading
        t = threading.Thread(target=_do_export_deploy, daemon=True)
        t.start()

    def _trigger_dataset_search(self, task_id: str, data_expansion_signal: dict) -> None:
        """Trigger dataset search for plateau-breaking."""
        try:
            discovery = DatasetDiscovery()
            rec = data_expansion_signal.get("recommendation", "fire smoke detection")
            results = discovery.search(query=rec, max_results=3)
            print(f"[Plateau] Dataset search results: {len(results)} datasets found")
            for ds in results:
                print(f"  - {ds.name} ({ds.source}): relevance={ds.relevance_score:.2f}")
        except Exception as e:
            print(f"[Plateau] Dataset search failed: {e}")

    def confirm(self, task_id: str, approved: bool, overrides: dict = None) -> bool:
        """Record human confirmation for HiTL gates."""
        r = self._get_redis()
        if approved:
            if overrides:
                r.hset(f"agent:{task_id}", mapping={
                    f"overrides_{r.hget(f'agent:{task_id}', 'status')}": json.dumps(overrides)
                })
            return True
        r.hset(f"agent:{task_id}", mapping={
            "status": "cancelled",
            "completed_at": datetime.now().isoformat(),
        })
        return True

    def get_status(self, task_id: str) -> dict | None:
        """Get full task status from Redis."""
        r = self._get_redis()
        data = r.hgetall(f"agent:{task_id}")
        if not data:
            return None
        data["progress"] = float(data.get("progress", "0.0"))
        return data

    def get_pipeline_status(self, task_id: str) -> dict:
        """Get pipeline_id and pipeline_status."""
        r = self._get_redis()
        data = r.hgetall(f"agent:{task_id}")
        return {
            "pipeline_id": data.get("pipeline_id", ""),
            "pipeline_status": data.get("pipeline_status", ""),
        }

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        r = self._get_redis()
        r.hset(f"agent:{task_id}", mapping={
            "status": "cancelled",
            "completed_at": datetime.now().isoformat(),
        })
        return True
