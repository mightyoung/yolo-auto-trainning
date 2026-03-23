"""Restart Business API server and run E2E test for the training pipeline."""
import subprocess
import time
import socket
import sys
import os
import json
import urllib.request
import urllib.error

WORKING_DIR = r"E:\yolo-auto-training"
BIZ_API_DIR = os.path.join(WORKING_DIR, "business-api")
LOG_FILE = os.path.join(WORKING_DIR, "biz_server.log")
TOKEN_FILE = os.path.join(WORKING_DIR, ".token.txt")
E2E_LOG = os.path.join(WORKING_DIR, "e2e_test.log")

def is_port_open(host, port, timeout=2):
    try:
        s = socket.socket()
        s.settimeout(timeout)
        r = s.connect_ex((host, port))
        s.close()
        return r == 0
    except:
        return False

def kill_server_on_port(port):
    """Kill process listening on port using Python."""
    import subprocess
    # Use netstat to find PID, then kill it
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"],
            capture_output=True, text=True, timeout=15
        )
        print(f"  Kill result: {result.stdout.strip()[:100]}")
    except Exception as e:
        print(f"  Kill attempt: {e}")
    time.sleep(3)
    if not is_port_open("localhost", port):
        print(f"  Port {port} is now free")
    else:
        print(f"  Warning: Port {port} may still be in use")

def start_server(log_file_path):
    """Start Business API server."""
    logf = open(log_file_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.gateway:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=BIZ_API_DIR,
        stdout=logf,
        stderr=subprocess.STDOUT,
        env=os.environ.copy()
    )
    print(f"  Server started with PID {proc.pid}")
    return proc, logf

def wait_for_server(host, port, max_wait=30):
    start = time.time()
    while time.time() - start < max_wait:
        if is_port_open(host, port):
            print(f"  Server ready at {host}:{port}")
            return True
        time.sleep(1)
    print(f"  Server failed to start within {max_wait}s")
    return False

def login_and_get_token():
    """Login and save fresh token to .token.txt, return the token."""
    url = "http://localhost:8000/api/v1/data/auth/login"
    data = json.dumps({"username": "admin", "password": "admin123"}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            token = result.get("access_token", "")
            with open(TOKEN_FILE, "w") as f:
                f.write("Token loaded from login\n" + token)
            return token
    except Exception as e:
        log(f"  Login failed: {e}")
        return None

def load_token():
    # Try loading from file first
    try:
        with open(TOKEN_FILE, "r") as f:
            content = f.read().strip()
        for line in reversed(content.split("\n")):
            line = line.strip()
            if line and not line.startswith("Token loaded"):
                return line
    except:
        pass
    return None

def api_get(path, token):
    url = f"http://localhost:8000{path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code

def api_post(path, token, data=None):
    url = f"http://localhost:8000{path}"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode() if data else None
    if body:
        req.data = body
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read()), e.code
        except:
            return {"error": str(e)}, e.code

def log(msg):
    print(msg)
    with open(E2E_LOG, "a") as f:
        f.write(msg + "\n")

def main():
    # Clear old log
    open(E2E_LOG, "w").close()

    log("=== STEP 1: Kill old server ===")
    kill_server_on_port(8000)
    time.sleep(2)

    log("=== STEP 2: Start new server ===")
    proc, logf = start_server(LOG_FILE)

    log("=== STEP 3: Wait for server ready ===")
    if not wait_for_server("localhost", 8000, 30):
        log("FATAL: Server did not start")
        proc.terminate()
        return
    time.sleep(2)  # Extra settle time

    log("=== STEP 4: Load auth token ===")
    token = load_token()
    if not token:
        log("  No cached token, logging in...")
        token = login_and_get_token()
    if not token:
        log("FATAL: No auth token found")
        proc.terminate()
        return
    log(f"  Token loaded: {len(token)} chars")

    log("=== STEP 5: Submit task ===")
    result, status = api_post("/api/v1/agent/task", token, {
        "task": "Train a fire and smoke detection model using YOLO11",
        "context": None,
        "agents": None
    })
    log(f"  Status: {status}")
    log(f"  Response: {json.dumps(result)}")
    if status != 200:
        log("FATAL: Task submission failed")
        proc.terminate()
        return

    task_id = result.get("task_id")
    log(f"  Task ID: {task_id}")

    # Poll Phase 1
    log("=== STEP 6: Poll Phase 1 (dataset discovery) ===")
    for i in range(30):
        time.sleep(2)
        data, status = api_get(f"/api/v1/agent/task/{task_id}", token)
        s = data.get("status", "")
        log(f"  [{i*2}s] status={s}")
        if s == "awaiting_confirmation":
            log("  Phase 1: awaiting_confirmation - GOOD")
            break
        if s in ("failed", "error"):
            log(f"  Phase 1 FAILED: {data.get('error')}")
            proc.terminate()
            return
    else:
        log("  Phase 1 timeout waiting for confirmation")
        proc.terminate()
        return

    log("=== STEP 7: Confirm Phase 1 (dataset) ===")
    result, status = api_post(f"/api/v1/agent/task/{task_id}/confirm", token, {
        "approved": True,
        "overrides": None
    })
    log(f"  Status: {status}")
    log(f"  Response: {json.dumps(result)}")

    # Poll Phase 2
    log("=== STEP 8: Poll Phase 2 (training params gate) ===")
    for i in range(30):
        time.sleep(2)
        data, status = api_get(f"/api/v1/agent/task/{task_id}", token)
        s = data.get("status", "")
        log(f"  [{i*2}s] status={s}")
        if s == "awaiting_training_confirmation":
            log("  Phase 2: awaiting_training_confirmation - GOOD")
            break
        if s in ("failed", "error"):
            log(f"  Phase 2 FAILED: {data.get('error')}")
            proc.terminate()
            return
    else:
        log("  Phase 2 timeout waiting for training confirmation")
        proc.terminate()
        return

    log("=== STEP 9: Confirm Phase 2 (training) ===")
    result, status = api_post(f"/api/v1/agent/task/{task_id}/confirm", token, {
        "approved": True,
        "overrides": {"model": "yolo11n", "epochs": 50, "imgsz": 640, "batch": 16}
    })
    log(f"  Status: {status}")
    log(f"  Response: {json.dumps(result)}")

    # Poll Phase 3 (training on GPU)
    log("=== STEP 10: Poll Phase 3 (GPU training) ===")
    last_status = ""
    for i in range(120):  # up to 10 minutes
        time.sleep(5)
        data, status = api_get(f"/api/v1/agent/task/{task_id}", token)
        s = data.get("status", "")
        p = data.get("progress", 0)
        ts = data.get("training_status", "N/A")
        log(f"  [{i*5}s] status={s} progress={p}% training_status={ts}")
        if s == "training_completed":
            log(f"  SUCCESS! Model at: {data.get('model_path')}")
            break
        if s in ("failed", "error"):
            err = data.get('error', 'unknown')
            log(f"  Training FAILED: {err}")
            break
        last_status = s
    else:
        if last_status == "training":
            log("  Timeout after 10 minutes - training still in progress")
        else:
            log(f"  Did not reach completed state, last status: {last_status}")

    log("=== E2E TEST COMPLETE ===")
    proc.terminate()
    logf.close()
    msg = f"\nAll logs saved to {E2E_LOG}"
    print(msg.encode('ascii', errors='replace').decode('ascii'))

if __name__ == "__main__":
    main()
