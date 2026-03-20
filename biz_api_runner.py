#!/usr/bin/env python3
"""Start Business API, log to file, and run tests."""
import subprocess, sys, os, time, urllib.request, json

# Load .env file from project root BEFORE copying environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"E:\yolo-auto-training\.env")

PORT = 8002  # 8000 may be zombie-socket'd on Windows; use 8002 if 8000 is blocked
CWD = r"E:\yolo-auto-training\business-api"
LOGFILE = r"E:\yolo-auto-training\biz_server.log"
ENV = os.environ.copy()

def get_pids(port):
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique"],
            capture_output=True, text=True, timeout=10
        )
        return [int(x.strip()) for x in r.stdout.strip().split('\n') if x.strip().isdigit()]
    except:
        return []

# Kill existing
for pid in get_pids(PORT):
    try:
        subprocess.run(["powershell", "-Command", f"Stop-Process -Id {pid} -Force"],
                       capture_output=True, timeout=10)
    except:
        pass

time.sleep(2)

# Start server, log everything
logf = open(LOGFILE, "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.gateway:app",
     "--host", "0.0.0.0", "--port", str(PORT)],
    cwd=CWD, env=ENV,
    stdout=logf, stderr=subprocess.STDOUT,
)

# Wait for startup
time.sleep(8)

# Check health
try:
    r = urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5)
    data = json.loads(r.read())
    result = json.dumps(data)
    print(f"HEALTH: {result}")
    logf.write(f"HEALTH: {result}\n")
except Exception as e:
    print(f"HEALTH FAILED: {e}")
    logf.write(f"HEALTH FAILED: {e}\n")

# Test login
try:
    req = urllib.request.Request(
        f"http://localhost:{PORT}/api/v1/data/auth/login",
        data=json.dumps({"username": "admin", "password": "admin123"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    r = urllib.request.urlopen(req, timeout=10)
    data = json.loads(r.read())
    print(f"LOGIN SUCCESS: {data}")
    logf.write(f"LOGIN SUCCESS: {json.dumps(data)}\n")
    with open(r"E:\yolo-auto-training\.token.txt", "w") as f:
        f.write(data.get("access_token", ""))
    print("Token saved.")
except urllib.error.HTTPError as e:
    body = e.read().decode(errors='replace')
    print(f"LOGIN HTTP {e.code}: {body}")
    logf.write(f"LOGIN HTTP {e.code}: {body}\n")
except Exception as e:
    print(f"LOGIN ERROR: {e}")
    logf.write(f"LOGIN ERROR: {e}\n")

logf.close()
print(f"Server PID: {proc.pid}")
print(f"Log file: {LOGFILE}")
print("Press Ctrl+C to stop server...")
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
