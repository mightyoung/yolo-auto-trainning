#!/usr/bin/env python3
"""HiTL full-flow integration test: Phase 1 -> Confirm -> Phase 2 -> Confirm -> Phase 3."""
import urllib.request, urllib.error, json, time, sys

BUSINESS_API = "http://localhost:8002"
API_KEY = "default-business-api-key"  # Business API service auth

def api_get(path, timeout=10):
    req = urllib.request.Request(
        BUSINESS_API + path,
        headers={"X-API-Key": API_KEY}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def api_post(path, data, timeout=10):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BUSINESS_API + path,
        data=body,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def poll_until(key_statuses, path, timeout_s, poll_interval=10, key="status"):
    """Poll an API until status matches expected values."""
    start = time.time()
    last = ""
    while time.time() - start < timeout_s:
        try:
            resp = api_get(path, timeout=15)
            status = resp.get(key, "?")
            elapsed = time.time() - start
            if status != last:
                print(f"  [{elapsed:.0f}s] status={status}")
                last = status
            if status in key_statuses:
                return status, resp
            time.sleep(poll_interval)
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(poll_interval)
    return "TIMEOUT", {}

# === STEP 1: Health check ===
print("=== 1. Business API health ===")
try:
    health = api_get("/health", timeout=5)
    print(f"  {health}")
except Exception as e:
    print(f"  DOWN: {e}")
    sys.exit(1)

# === STEP 2: Submit CrewAI task (Phase 1) ===
print("\n=== 2. Submit CrewAI task (HiTL Phase 1) ===")
task_payload = {
    "task": "火焰和烟雾检测：检测图像中的火焰和烟雾，适用于室内火灾预警和室外森林火灾监控场景"
}
try:
    submit_resp = api_post("/api/v1/agent/task", task_payload, timeout=30)
    print(f"  Submit: {submit_resp}")
    task_id = submit_resp.get("task_id", "")
    if not task_id:
        print("  ERROR: No task_id returned"); sys.exit(1)
    print(f"  Task ID: {task_id}")
except Exception as e:
    print(f"  Submit failed: {e}")
    sys.exit(1)

# === STEP 3: Poll until Phase 1 done (awaiting_confirmation) ===
print("\n=== 3. Poll until Phase 1 complete (awaiting_confirmation) ===")
status, resp = poll_until(
    ["awaiting_confirmation", "running"],
    f"/api/v1/agent/task/{task_id}",
    timeout_s=600,
    poll_interval=15,
    key="status"
)
print(f"  Phase 1 result status: {status}")
if status == "TIMEOUT":
    print("  TIMEOUT waiting for Phase 1")
    sys.exit(1)
elif status == "awaiting_confirmation":
    print("  Phase 1 COMPLETE - dataset discovery ready for confirmation")
    # Show discovery results
    print(f"  Current agent: {resp.get('current_agent', '?')}")
    print(f"  Progress: {resp.get('progress', '?')}")
elif status == "running":
    print("  Still running, will continue polling...")
    # Keep polling
    status2, resp2 = poll_until(
        ["awaiting_confirmation"],
        f"/api/v1/agent/task/{task_id}",
        timeout_s=600,
        poll_interval=15,
        key="status"
    )
    status = status2
    resp = resp2
    print(f"  Phase 1 final status: {status}")

# === STEP 4: Confirm Phase 1 (dataset choice) ===
print("\n=== 4. Confirm Phase 1 (approve dataset) ===")
try:
    confirm_payload = {
        "approved": True,
        "overrides": {}
    }
    confirm_resp = api_post(
        f"/api/v1/agent/task/{task_id}/confirm",
        confirm_payload,
        timeout=30
    )
    print(f"  Confirm: {confirm_resp}")
except Exception as e:
    print(f"  Confirm failed: {e}")
    # Print current status
    try:
        status_resp = api_get(f"/api/v1/agent/task/{task_id}")
        print(f"  Current status: {status_resp.get('status')}")
    except:
        pass
    sys.exit(1)

# === STEP 5: Poll until Phase 2 done (awaiting_training_confirmation) ===
print("\n=== 5. Poll until Phase 2 complete (awaiting_training_confirmation) ===")
status, resp = poll_until(
    ["awaiting_training_confirmation", "training", "training_completed", "failed"],
    f"/api/v1/agent/task/{task_id}",
    timeout_s=300,
    poll_interval=15,
    key="status"
)
print(f"  Phase 2 result status: {status}")
if status == "TIMEOUT":
    print("  TIMEOUT waiting for Phase 2")
    # Show what we got
    try:
        r = api_get(f"/api/v1/agent/task/{task_id}")
        print(f"  Current status: {r.get('status')}")
        print(f"  Current agent: {r.get('current_agent')}")
        print(f"  Training status: {r.get('training_status')}")
    except:
        pass
    sys.exit(1)
elif status == "failed":
    print(f"  FAILED: {resp.get('error', resp)}")
    sys.exit(1)
elif status in ("training", "training_completed"):
    print(f"  Phase 2 COMPLETE - training in progress")
elif status == "awaiting_training_confirmation":
    print("  Phase 2 COMPLETE - training params ready for confirmation")
    # Show training params
    print(f"  Model: {resp.get('training_model', '?')}")
    print(f"  Epochs: {resp.get('training_epochs', '?')}")

# === STEP 6: Confirm training params (Phase 2 confirmation) ===
if status == "awaiting_training_confirmation":
    print("\n=== 6. Confirm Phase 2 (approve training params) ===")
    try:
        confirm_payload = {
            "approved": True,
            "overrides": {
                "epochs": 50,
                "model": "yolo11n",
                "batch": 16,
                "device": "cuda:0"
            }
        }
        confirm_resp = api_post(
            f"/api/v1/agent/task/{task_id}/confirm",
            confirm_payload,
            timeout=30
        )
        print(f"  Confirm: {confirm_resp}")
    except Exception as e:
        print(f"  Confirm failed: {e}")
        sys.exit(1)

# === STEP 7: Poll until training complete ===
print("\n=== 7. Poll until training complete ===")
status, resp = poll_until(
    ["training_completed", "completed", "failed"],
    f"/api/v1/agent/task/{task_id}",
    timeout_s=1800,
    poll_interval=30,
    key="status"
)
print(f"  Training result status: {status}")
if status == "TIMEOUT":
    print("  TIMEOUT waiting for training")
    sys.exit(1)
elif status == "failed":
    print(f"  FAILED: {resp.get('error', resp)}")
    sys.exit(1)
elif status in ("training_completed", "completed"):
    print(f"  TRAINING COMPLETE!")
    print(f"  Model path: {resp.get('model_path', '?')}")
    print(f"  mAP50: {resp.get('metrics', {}).get('mAP50', '?')}")

# === STEP 8: Final status ===
print("\n=== 8. Final status ===")
try:
    final = api_get(f"/api/v1/agent/task/{task_id}")
    print(f"  Status: {final.get('status')}")
    print(f"  Progress: {final.get('progress')}")
    print(f"  Model path: {final.get('model_path', 'N/A')}")
    print(f"  Error: {final.get('error', 'None')}")
except Exception as e:
    print(f"  Failed to get final status: {e}")

print("\n=== DONE ===")
