"""
Auto-adjust agent for training plateau handling.
Location: business-api/src/agents/auto_adjust_agent.py

Background agent that monitors training plateau and auto-triggers adjustments:
- Level 1: Cancel current task + restart with halved lr0 + resume_from best.pt
- Level 2: Log augmentation boost (already handled by Training API)
- Level 3: Run ActiveLearning + SemiSupervised pipeline to expand dataset
"""

import os
import json
import threading
import uuid
from datetime import datetime
from typing import Optional

import httpx

from .worker_memory import (
    append_agent_attempt,
    append_autoadjust_attempt,
    build_attempt_record,
    extract_agent_submission,
    sanitize_training_status,
)
from .coordinator_summary import summarize_attempt, summarize_tool_batch
from .operation_policy import require_operation_allowed
from .task_output import append_agent_output
try:
    from ..api.task_registry import append_task_event
except ImportError:  # pragma: no cover - compatibility with direct package imports
    from api.task_registry import append_task_event
from .plateau_planner import (
    build_plateau_attempt_record,
    build_plateau_decision,
)
from .strategy_governor import (
    build_strategy_proposal,
    store_pending_strategy_proposal,
)


def _get_redis_for_adjust():
    """Get Redis client for AutoAdjustAgent."""
    import redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_password = os.getenv("REDIS_PASSWORD", None)
    try:
        if redis_password:
            return redis.from_url(redis_url, password=redis_password, decode_responses=True)
        return redis.from_url(redis_url, decode_responses=True)
    except Exception:
        # Fallback: try direct connection
        try:
            return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True, password=redis_password)
        except Exception:
            return None


class AutoAdjustAgent:
    """Background agent that monitors training plateau and auto-triggers adjustments.

    Runs in a background thread, polling the Training API every 60s. When plateau
    signals are detected (lr_decay, augment_boost, data_expansion), it autonomously:
      Level 1: Cancel current task + restart with halved lr0 + resume_from best.pt
      Level 2: Log augmentation boost (already handled by Training API)
      Level 3: Run ActiveLearning + SemiSupervised pipeline to expand dataset

    Usage:
        agent = AutoAdjustAgent(task_id, training_task_id)
        agent.start()   # Spawns background thread
        agent.stop()    # Terminates gracefully
    """

    def __init__(self, task_id: str, training_task_id: str, *, proposal_only: bool = False):
        self.task_id = task_id
        self.training_task_id = training_task_id
        self.proposal_only = proposal_only
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._adjustments_triggered: list[dict] = []
        self._lr0_history: list[float] = [0.01]  # Start with default

    def start(self) -> None:
        """Start the auto-adjustment background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"AutoAdjust-{self.task_id[:8]}")
        self._thread.start()
        print(f"[AutoAdjust] Started for task {self.task_id}")

    def stop(self) -> None:
        """Stop the background thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        print(f"[AutoAdjust] Stopped for task {self.task_id}")

    def update_training_task_id(self, training_task_id: str) -> None:
        """Switch the active child training task once the committer starts a new run."""
        self.training_task_id = training_task_id

    def _emit_strategy_proposal(self, r, decision: dict, *, selected_action: str) -> None:
        """Persist a proposal for the Business-side committer to consume."""
        proposal = build_strategy_proposal(
            parent_run_id=self.task_id,
            child_training_task_id=self.training_task_id,
            decision=decision,
            proposal_id=f"proposal_{uuid.uuid4().hex[:12]}",
        )
        store_pending_strategy_proposal(r, self.task_id, proposal)
        append_agent_output(
            r,
            self.task_id,
            f"Strategy proposal emitted: {selected_action}",
            summary=summarize_tool_batch(
                "strategy proposal",
                outcome="proposed",
                detail=selected_action,
                subject=self.task_id,
            ),
        )
        event_state = append_task_event(
            r.hgetall(f"agent:{self.task_id}"),
            source=self.training_task_id,
            target=self.task_id,
            relation="strategy_change_proposed",
            node_type="strategy",
            label=selected_action,
            metadata=proposal,
        )
        r.hset(
            f"agent:{self.task_id}",
            mapping={
                "event_graph": json.dumps(event_state.get("event_graph", {})),
                "strategy_ledger": json.dumps(event_state.get("strategy_ledger", [])),
                "latest_strategy_proposal": json.dumps(proposal),
            },
        )

    def _run(self) -> None:
        """Main polling loop."""
        import time

        r = _get_redis_for_adjust()
        base_url = os.getenv("TRAINING_API_URL", "http://localhost:8001")
        api_key = os.getenv("TRAINING_API_KEY", "")

        headers = {"X-API-Key": api_key}

        while not self._stop_event.is_set():
            try:
                # Poll Training API status
                with httpx.Client(timeout=15.0) as client:
                    resp = client.get(
                        f"{base_url}/api/v1/internal/train/status/{self.training_task_id}",
                        headers=headers,
                    )
                    if resp.status_code != 200:
                        time.sleep(60)
                        continue
                    status_data = resp.json()

                status = status_data.get("status", "unknown")
                live_mAP50 = status_data.get("live_mAP50")
                lr_decay_triggered = status_data.get("lr_decay_triggered", False)
                lr_decay_signal = status_data.get("lr_decay_signal") or {}
                data_expansion_requested = status_data.get("data_expansion_requested", False)
                data_expansion_signal = status_data.get("data_expansion_signal") or {}
                augment_boost_active = status_data.get("augment_boost_active", False)
                strategies_triggered = status_data.get("strategies_triggered") or []
                status_summary = sanitize_training_status(status_data)
                plateau_decision = build_plateau_decision(status_summary, self._adjustments_triggered)

                # Update Redis with current state
                redis_mapping = {
                    "status": status,
                    "live_mAP50": str(live_mAP50) if live_mAP50 is not None else "",
                    "lr_decay_triggered": str(lr_decay_triggered),
                    "data_expansion_requested": str(data_expansion_requested),
                    "augment_boost_active": str(augment_boost_active),
                    "strategies_triggered": str(strategies_triggered),
                    "status_summary": json.dumps(status_summary),
                    "last_checked": datetime.now().isoformat(),
                }
                if plateau_decision:
                    redis_mapping["plateau_decision"] = json.dumps(plateau_decision)
                    redis_mapping["plateau_selected_action"] = plateau_decision["selected"]["action"]
                    redis_mapping["plateau_decision_rationale"] = plateau_decision["rationale"]
                r.hset(f"autoadjust:{self.task_id}", mapping=redis_mapping)

                if plateau_decision:
                    record = build_plateau_attempt_record(
                        task_id=self.task_id,
                        training_task_id=self.training_task_id,
                        decision=plateau_decision,
                    )
                    append_agent_attempt(r, self.task_id, record)
                    append_autoadjust_attempt(r, self.task_id, record)
                    event_state = append_task_event(
                        r.hgetall(f"agent:{self.task_id}"),
                        source=self.training_task_id,
                        target=self.task_id,
                        relation="plateau_decision",
                        node_type="plateau",
                        label=plateau_decision["selected"]["action"],
                        metadata={
                            "selected_action": plateau_decision["selected"]["action"],
                            "candidate_count": plateau_decision["candidate_count"],
                        },
                    )
                    r.hset(f"agent:{self.task_id}", mapping={"event_graph": json.dumps(event_state.get("event_graph", {}))})

                # Don't act if training already finished
                if status in ("completed", "failed", "cancelled"):
                    break

                selected_action = plateau_decision["selected"]["action"] if plateau_decision else None

                # Level 1: LR decay — auto-restart with halved lr0 and resume
                if selected_action == "lr_decay" and lr_decay_signal:
                    append_agent_output(
                        r,
                        self.task_id,
                        f"Plateau decision selected lr_decay at count={lr_decay_signal.get('lr_decay_count', 1)}",
                        summary=summarize_tool_batch(
                            "plateau decision",
                            outcome="selected",
                            detail="lr_decay",
                            subject=self.task_id,
                        ),
                    )
                    decay_count = lr_decay_signal.get("lr_decay_count", 1)
                    if self.proposal_only:
                        print(f"[AutoAdjust] Proposal-only: LR decay #{decay_count}")
                        self._emit_strategy_proposal(r, plateau_decision, selected_action=selected_action)
                    else:
                        print(f"[AutoAdjust] Level 1: LR decay #{decay_count} — triggering auto-adjust")
                        self._trigger_lr_adjustment(
                            lr_decay_signal=lr_decay_signal,
                            decay_count=decay_count,
                            base_url=base_url,
                            headers=headers,
                            r=r,
                        )

                # Level 3: Data expansion — run ActiveLearning + SemiSupervised pipeline
                if selected_action == "data_expansion" and data_expansion_signal:
                    append_agent_output(
                        r,
                        self.task_id,
                        "Plateau decision selected data_expansion",
                        summary=summarize_tool_batch(
                            "plateau decision",
                            outcome="selected",
                            detail="data_expansion",
                            subject=self.task_id,
                        ),
                    )
                    if self.proposal_only:
                        print("[AutoAdjust] Proposal-only: data expansion")
                        self._emit_strategy_proposal(r, plateau_decision, selected_action=selected_action)
                    else:
                        print(f"[AutoAdjust] Level 3: Data expansion triggered")
                        self._trigger_data_expansion(
                            data_expansion_signal=data_expansion_signal,
                            base_url=base_url,
                            headers=headers,
                            r=r,
                        )

            except Exception as e:
                print(f"[AutoAdjust] Error in polling loop: {e}")

            # Poll every 60 seconds
            self._stop_event.wait(timeout=60)

    def _trigger_lr_adjustment(
        self,
        lr_decay_signal: dict,
        decay_count: int,
        base_url: str,
        headers: dict,
        r,
        *,
        new_task_id: str | None = None,
        proposal: dict | None = None,
        raise_on_error: bool = False,
        lease_refresh=None,
    ) -> dict | None:
        """Cancel current task and restart with halved lr0 + resume_from best.pt."""
        try:
            if lease_refresh:
                lease_refresh()
            current_lr = self._lr0_history[-1]
            new_lr = max(current_lr * 0.5, 1e-6)
            self._lr0_history.append(new_lr)

            # Get current task params from Redis
            task_data = r.hgetall(f"agent:{self.task_id}")
            params = extract_agent_submission(task_data)

            model = params.get("model", "yolo11m")
            data_yaml = params.get("data_yaml", "/home/wangxin/data/fire-smoke/data.yaml")
            original_epochs = int(params.get("epochs", 100))
            imgsz = int(params.get("imgsz", 640))
            device = params.get("device", "cuda:0")

            # Find the best.pt from current run
            current_output_dir = task_data.get("output_dir", f"/home/wangxin/runs/{self.task_id}")
            best_pt = f"{current_output_dir}/weights/best.pt"

            # Generate new task ID
            new_task_id = new_task_id or f"train_{uuid.uuid4().hex[:8]}"

            # Cancel current training via Training API
            if lease_refresh:
                lease_refresh()
            with httpx.Client(timeout=15.0) as client:
                client.post(
                    f"{base_url}/api/v1/internal/train/cancel/{self.training_task_id}",
                    headers=headers,
                )

            # Start new training with halved lr0 + resume
            hpo_params = {"lr0": new_lr}

            require_operation_allowed(
                "gpu_training_submit",
                context={"task_id": self.task_id, "training_task_id": self.training_task_id, "device": device},
            )
            if lease_refresh:
                lease_refresh()
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{base_url}/api/v1/internal/train/start",
                    json={
                        "task_id": new_task_id,
                        "model": model,
                        "data_yaml": data_yaml,
                        "epochs": original_epochs + 50,  # Add extra epochs
                        "imgsz": imgsz,
                        "batch": 16,
                        "device": device,
                        "output_dir": f"/home/wangxin/runs/{new_task_id}",
                        "augmentation_preset": "strong",
                        "resume_from": best_pt,
                        "lr0": new_lr,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                result = resp.json()

            new_training_task_id = result.get("task_id", new_task_id)

            # Update Redis
            r.hset(f"agent:{self.task_id}", mapping={
                "status": "auto_adjusting",
                "adjusted_task_id": new_task_id,
                "adjusted_training_id": new_training_task_id,
                "lr_adjustment": str({
                    "old_lr": current_lr,
                    "new_lr": new_lr,
                    "decay_count": decay_count,
                    "resume_from": best_pt,
                    "timestamp": datetime.now().isoformat(),
                }),
            })

            self._adjustments_triggered.append({
                "level": 1,
                "decay_count": decay_count,
                "old_lr": current_lr,
                "new_lr": new_lr,
                "new_task_id": new_task_id,
                "timestamp": datetime.now().isoformat(),
            })
            record = build_attempt_record(
                attempt_type="auto_adjust",
                stage="lr_decay",
                outcome="completed",
                source="auto_adjust_agent",
                action="restart_with_halved_lr",
                training_task_id=new_training_task_id,
                details={
                    "old_lr": current_lr,
                    "new_lr": new_lr,
                    "decay_count": decay_count,
                },
            )
            append_agent_attempt(r, self.task_id, record)
            append_autoadjust_attempt(r, self.task_id, record)
            append_agent_output(
                r,
                self.task_id,
                f"LR decay restart submitted: {new_training_task_id}",
                summary=summarize_attempt(
                    "auto_adjust",
                    "lr_decay",
                    "completed",
                    action="restart_with_halved_lr",
                    detail=f"{new_training_task_id} lr={new_lr:.6f}",
                ),
            )
            event_state = append_task_event(
                r.hgetall(f"agent:{self.task_id}"),
                source=self.training_task_id,
                target=new_training_task_id,
                relation="auto_adjust_retry",
                node_type="auto_adjust",
                label="lr_decay",
                metadata={
                    "proposal_id": proposal.get("proposal_id") if proposal else None,
                    "commit_id": proposal.get("commit_id") if proposal else None,
                    "decay_count": decay_count,
                    "old_lr": current_lr,
                    "new_lr": new_lr,
                },
            )
            r.hset(f"agent:{self.task_id}", mapping={"event_graph": json.dumps(event_state.get("event_graph", {}))})

            print(f"[AutoAdjust] Level 1 complete: new task={new_task_id}, lr={new_lr:.6f}, resume={best_pt}")
            return {
                "new_task_id": new_task_id,
                "new_training_task_id": new_training_task_id,
                "new_lr": new_lr,
                "resume_from": best_pt,
            }

        except Exception as e:
            print(f"[AutoAdjust] Level 1 failed: {e}")
            record = build_attempt_record(
                attempt_type="auto_adjust",
                stage="lr_decay",
                outcome="failed",
                source="auto_adjust_agent",
                action="restart_with_halved_lr",
                error=str(e),
            )
            append_agent_attempt(r, self.task_id, record)
            append_autoadjust_attempt(r, self.task_id, record)
            append_agent_output(
                r,
                self.task_id,
                f"LR decay adjustment failed: {e}",
                summary=summarize_attempt(
                    "auto_adjust",
                    "lr_decay",
                    "failed",
                    action="restart_with_halved_lr",
                    error=str(e),
                ),
            )
            r.hset(f"agent:{self.task_id}", mapping={
                "lr_adjustment_error": str(e),
            })
            if raise_on_error:
                raise
            return None

    def _trigger_data_expansion(
        self,
        data_expansion_signal: dict,
        base_url: str,
        headers: dict,
        r,
        *,
        new_task_id: str | None = None,
        proposal: dict | None = None,
        raise_on_error: bool = False,
        lease_refresh=None,
    ) -> dict | None:
        """Run ActiveLearning + SemiSupervised pipeline to expand dataset."""
        try:
            if lease_refresh:
                lease_refresh()
            target_mAP = data_expansion_signal.get("target_mAP50", 0.90)
            current_mAP = data_expansion_signal.get("current_mAP50", 0)
            expansion_round = sum(1 for a in self._adjustments_triggered if a.get("level") == 3) + 1

            print(f"[AutoAdjust] Data expansion round {expansion_round}: "
                  f"mAP50={current_mAP:.4f} → target={target_mAP:.2f}")

            # Get model path
            task_data = r.hgetall(f"agent:{self.task_id}")
            model_path = task_data.get("model_path")
            if not model_path:
                # Try to find best.pt in output dir
                output_dir = task_data.get("output_dir", f"/home/wangxin/runs/{self.task_id}")
                model_path = f"{output_dir}/weights/best.pt"

            if not model_path:
                print("[AutoAdjust] No model path found, skipping data expansion")
                if raise_on_error:
                    raise RuntimeError("No model path found for data expansion")
                return None

            # Step 1: ActiveLearning — select most uncertain samples via SSH on GPU server
            unlabeled_dirs = [
                "/home/wangxin/data/unlabeled_fire",
                "/home/wangxin/data/raw",
                "/home/wangxin/data/fire-smoke/unlabeled",
            ]

            selected_samples = []
            # SSH bridge: check GPU server filesystem via paramiko
            try:
                if lease_refresh:
                    lease_refresh()
                import paramiko
                ssh_host = os.getenv("GPU_SERVER_HOST")
                ssh_user = os.getenv("GPU_SERVER_USER")
                ssh_pass = os.getenv("GPU_SERVER_PASS")
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)

                for img_dir in unlabeled_dirs:
                    # Check via SSH that the directory exists AND has image files
                    check_script = (
                        f"import sys; from pathlib import Path; p=Path(r'{img_dir}'); "
                        f"imgs=list(p.glob('*.jpg'))+list(p.glob('*.png'))+list(p.glob('*.jpeg'))+list(p.glob('*.bmp')); "
                        f"print(f'EXISTS {{len(imgs)}}')"
                    )
                    stdin, stdout, stderr = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{check_script}"',
                        timeout=15
                    )
                    stdout.channel.recv_exit_status()
                    line = stdout.read().decode(errors='replace').strip()
                    if not line.startswith("EXISTS "):
                        continue
                    count = int(line.split()[1])
                    if count == 0:
                        continue
                    print(f"[AutoAdjust] Found {count} unlabeled images in {img_dir}")

                    # Run ActiveLearning on GPU server via SSH
                    require_operation_allowed(
                        "ssh_dataset_check",
                        context={"dataset_path": img_dir, "source": "active_learning"},
                    )
                    al_script = (
                        "import sys, json\n"
                        "sys.path.insert(0, '/home/wangxin/yolo-auto-training/training-api/src')\n"
                        "from training.active_learner import ActiveLearningPipeline, ActiveLearningConfig\n"
                        f"al = ActiveLearningPipeline(config=ActiveLearningConfig(strategy='entropy', top_k=200, batch_size=16))\n"
                        f"result = al.select_uncertain_samples(model_path='{model_path}', image_dir='{img_dir}')\n"
                        "print(json.dumps(result, default=str))"
                    )
                    stdin2, stdout2, stderr2 = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{al_script}"',
                        timeout=120
                    )
                    stdout2.channel.recv_exit_status()
                    output = stdout2.read().decode(errors='replace')
                    if output.strip():
                        try:
                            result = json.loads(output)
                            if result.get("selected"):
                                selected_samples.extend(result["selected"])
                                print(f"[AutoAdjust] AL: found {len(result['selected'])} uncertain samples in {img_dir}")
                        except json.JSONDecodeError:
                            print(f"[AutoAdjust] AL parse error for {img_dir}: {output[:200]}")
                ssh.close()
            except Exception as e:
                print(f"[AutoAdjust] SSH-based AL failed: {e}")

            if not selected_samples:
                print("[AutoAdjust] No unlabeled images found — data expansion not possible")
                r.hset(f"agent:{self.task_id}", mapping={
                    "expansion_status": "no_unlabeled_data",
                    "expansion_note": "No unlabeled image directories found on GPU server",
                })
                self._adjustments_triggered.append({
                    "level": 3,
                    "round": expansion_round,
                    "status": "no_data",
                    "timestamp": datetime.now().isoformat(),
                })
                record = build_attempt_record(
                    attempt_type="auto_adjust",
                    stage="data_expansion",
                    outcome="skipped",
                    source="auto_adjust_agent",
                    action="expand_dataset",
                    error="No unlabeled image directories found on GPU server",
                    details={"round": expansion_round},
                )
                append_agent_attempt(r, self.task_id, record)
                append_autoadjust_attempt(r, self.task_id, record)
                append_agent_output(
                    r,
                    self.task_id,
                    "No unlabeled data found for expansion",
                    summary=summarize_attempt(
                        "auto_adjust",
                        "data_expansion",
                        "skipped",
                        action="expand_dataset",
                        error="No unlabeled image directories found on GPU server",
                    ),
                )
                if raise_on_error:
                    raise RuntimeError("No unlabeled image directories found on GPU server")
                return None

            # Step 2: SemiSupervised — generate pseudo-labels via SSH on GPU server
            pseudo_labels = []
            sample_paths = [s["path"] for s in selected_samples[:500]]
            if sample_paths:
                try:
                    if lease_refresh:
                        lease_refresh()
                    import paramiko
                    ssh_host = os.getenv("GPU_SERVER_HOST")
                    ssh_user = os.getenv("GPU_SERVER_USER")
                    ssh_pass = os.getenv("GPU_SERVER_PASS")
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)

                    # Build Python list string for sample paths
                    paths_repr = repr(sample_paths)
                    require_operation_allowed(
                        "ssh_dataset_download",
                        context={"dataset_path": model_path, "source": "semi_supervised"},
                    )
                    ss_script = (
                        "import sys, json\n"
                        "sys.path.insert(0, '/home/wangxin/yolo-auto-training/training-api/src')\n"
                        "from training.semi_supervised import SemiSupervisedPipeline\n"
                        f"ss = SemiSupervisedPipeline(confidence_threshold=0.75)\n"
                        f"pseudo_labels = ss.generate_pseudo_labels(\n"
                        f"    teacher_model_path='{model_path}',\n"
                        f"    unlabeled_images={paths_repr},\n"
                        f"    method='yolo_teacher',\n"
                        ")\n"
                        "print(json.dumps([{'path': p.path, 'boxes': p.boxes, 'confidence': p.confidence} for p in pseudo_labels], default=str))"
                    )
                    stdin, stdout, stderr = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{ss_script}"',
                        timeout=600
                    )
                    stdout.channel.recv_exit_status()
                    output = stdout.read().decode(errors='replace')
                    stderr_err = stderr.read().decode(errors='replace')
                    if stderr_err:
                        print(f"[AutoAdjust] SS stderr: {stderr_err[:300]}")
                    if output.strip():
                        try:
                            pseudo_labels = json.loads(output)
                            print(f"[AutoAdjust] SS: generated {len(pseudo_labels)} pseudo-labels")
                        except json.JSONDecodeError:
                            print(f"[AutoAdjust] SS parse error: {output[:200]}")
                    ssh.close()
                except Exception as e:
                    print(f"[AutoAdjust] SSH-based SS failed: {e}")

            if not pseudo_labels:
                print("[AutoAdjust] No pseudo-labels generated")
                append_agent_output(
                    r,
                    self.task_id,
                    "No pseudo-labels generated",
                    summary=summarize_attempt(
                        "auto_adjust",
                        "data_expansion",
                        "skipped",
                        action="generate_pseudo_labels",
                    ),
                )
                if raise_on_error:
                    raise RuntimeError("No pseudo-labels generated for data expansion")
                return None

            # Step 3: Filter and create expanded dataset via SSH
            filtered = []
            expanded_yaml = None
            if pseudo_labels:
                try:
                    if lease_refresh:
                        lease_refresh()
                    import paramiko
                    ssh_host = os.getenv("GPU_SERVER_HOST")
                    ssh_user = os.getenv("GPU_SERVER_USER")
                    ssh_pass = os.getenv("GPU_SERVER_PASS")
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)

                    pseudo_repr = repr(pseudo_labels[:500])
                    require_operation_allowed(
                        "ssh_dataset_yaml",
                        context={"dataset_path": expanded_yaml or f"/home/wangxin/data/expanded_{self.task_id}"},
                    )
                    filter_script = (
                        "import sys, json\n"
                        "sys.path.insert(0, '/home/wangxin/yolo-auto-training/training-api/src')\n"
                        "from training.semi_supervised import SemiSupervisedPipeline, PseudoLabel\n"
                        f"ss = SemiSupervisedPipeline(confidence_threshold=0.75)\n"
                        f"pseudo_labels = [PseudoLabel(**p) for p in {pseudo_repr}]\n"
                        f"filtered = ss.filter_pseudo_labels(pseudo_labels, min_boxes=1, max_boxes=50)\n"
                        f"expanded_dir = '/home/wangxin/data/expanded_{new_task_id or self.task_id}_{proposal.get('commit_id', 'draft') if proposal else 'draft'}'\n"
                        f"expanded_yaml = ss.create_pseudo_dataset(filtered, output_dir=expanded_dir, class_names=['fire', 'smoke'])\n"
                        "print(json.dumps({'filtered': len(filtered), 'yaml': expanded_yaml}))"
                    )
                    stdin, stdout, stderr = ssh.exec_command(
                        f'/home/wangxin/yolo-auto-training/training-venv/bin/python -c "{filter_script}"',
                        timeout=300
                    )
                    stdout.channel.recv_exit_status()
                    output = stdout.read().decode(errors='replace')
                    if output.strip():
                        try:
                            result = json.loads(output)
                            filtered_count = result.get("filtered", 0)
                            expanded_yaml = result.get("yaml")
                            filtered = pseudo_labels[:filtered_count]
                            print(f"[AutoAdjust] Filtered to {filtered_count} quality pseudo-labels")
                            if expanded_yaml:
                                print(f"[AutoAdjust] Expanded dataset created: {expanded_yaml}")
                        except json.JSONDecodeError:
                            print(f"[AutoAdjust] Filter/create error: {output[:200]}")
                    ssh.close()
                except Exception as e:
                    print(f"[AutoAdjust] SSH-based filter/create failed: {e}")

            if not filtered or not expanded_yaml:
                if raise_on_error:
                    raise RuntimeError("Expanded dataset was not materialized")
                return None

            # Step 4: Submit new training with expanded dataset
            task_data = r.hgetall(f"agent:{self.task_id}")
            params = extract_agent_submission(task_data)
            model = params.get("model", "yolo11m")
            imgsz = int(params.get("imgsz", 640))
            device = params.get("device", "cuda:0")

            new_task_id = new_task_id or f"train_{uuid.uuid4().hex[:8]}"

            if lease_refresh:
                lease_refresh()
            with httpx.Client(timeout=15.0) as client:
                require_operation_allowed(
                    "gpu_training_submit",
                    context={"task_id": self.task_id, "training_task_id": self.training_task_id, "device": device},
                )
                resp = client.post(
                    f"{base_url}/api/v1/internal/train/start",
                    json={
                        "task_id": new_task_id,
                        "model": model,
                        "data_yaml": expanded_yaml,
                        "epochs": 100,
                        "imgsz": imgsz,
                        "batch": 16,
                        "device": device,
                        "output_dir": f"/home/wangxin/runs/{new_task_id}",
                        "augmentation_preset": "strong",
                        "resume_from": model_path,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                result = resp.json()

            new_training_task_id = result.get("task_id", new_task_id)

            r.hset(f"agent:{self.task_id}", mapping={
                "status": "expanding_data",
                "expansion_task_id": new_task_id,
                "expansion_training_id": new_training_task_id,
                "expansion_pseudo_labels": str(len(filtered)),
                "expansion_dataset": expanded_yaml,
            })

            self._adjustments_triggered.append({
                "level": 3,
                "round": expansion_round,
                "status": "success",
                "pseudo_labels": len(filtered),
                "new_task_id": new_task_id,
                "timestamp": datetime.now().isoformat(),
            })
            record = build_attempt_record(
                attempt_type="auto_adjust",
                stage="data_expansion",
                outcome="completed",
                source="auto_adjust_agent",
                action="expand_dataset",
                training_task_id=new_training_task_id,
                details={
                    "round": expansion_round,
                    "pseudo_labels": len(filtered),
                    "expanded_yaml": expanded_yaml,
                },
            )
            append_agent_attempt(r, self.task_id, record)
            append_autoadjust_attempt(r, self.task_id, record)
            append_agent_output(
                r,
                self.task_id,
                f"Data expansion submitted: {new_training_task_id}",
                summary=summarize_attempt(
                    "auto_adjust",
                    "data_expansion",
                    "completed",
                    action="expand_dataset",
                    detail=f"{new_training_task_id} pseudo_labels={len(filtered)}",
                ),
            )
            event_state = append_task_event(
                r.hgetall(f"agent:{self.task_id}"),
                source=self.training_task_id,
                target=new_training_task_id,
                relation="data_expansion_retry",
                node_type="auto_adjust",
                label="data_expansion",
                metadata={
                    "proposal_id": proposal.get("proposal_id") if proposal else None,
                    "commit_id": proposal.get("commit_id") if proposal else None,
                    "round": expansion_round,
                    "pseudo_labels": len(filtered),
                    "expanded_yaml": expanded_yaml,
                },
            )
            r.hset(f"agent:{self.task_id}", mapping={"event_graph": json.dumps(event_state.get("event_graph", {}))})

            print(f"[AutoAdjust] Level 3 complete: new task={new_task_id}, "
                  f"pseudo_labels={len(filtered)}, yaml={expanded_yaml}")
            return {
                "new_task_id": new_task_id,
                "new_training_task_id": new_training_task_id,
                "expanded_yaml": expanded_yaml,
                "pseudo_labels": len(filtered),
            }

        except Exception as e:
            print(f"[AutoAdjust] Level 3 failed: {e}")
            record = build_attempt_record(
                attempt_type="auto_adjust",
                stage="data_expansion",
                outcome="failed",
                source="auto_adjust_agent",
                action="expand_dataset",
                error=str(e),
            )
            append_agent_attempt(r, self.task_id, record)
            append_autoadjust_attempt(r, self.task_id, record)
            append_agent_output(
                r,
                self.task_id,
                f"Data expansion failed: {e}",
                summary=summarize_attempt(
                    "auto_adjust",
                    "data_expansion",
                    "failed",
                    action="expand_dataset",
                    error=str(e),
                ),
            )
            event_state = append_task_event(
                r.hgetall(f"agent:{self.task_id}"),
                source=self.training_task_id,
                target=self.task_id,
                relation="plateau_action_failed",
                node_type="plateau",
                label="failed",
                metadata={"stage": "data_expansion"},
            )
            r.hset(f"agent:{self.task_id}", mapping={"event_graph": json.dumps(event_state.get("event_graph", {}))})
            r.hset(f"agent:{self.task_id}", mapping={
                "expansion_status": "failed",
                "expansion_error": str(e),
            })
            self._adjustments_triggered.append({
                "level": 3,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            if raise_on_error:
                raise
            return None
