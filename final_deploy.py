#!/usr/bin/env python3
"""
Final deployment: kill old process on port 8001, upload ALL files,
restart Training API on GPU with CUDA_VISIBLE_DEVICES=1 (Tesla T4 GPU 1),
then run full E2E verification with Business API.
"""
import paramiko, time, httpx, subprocess, sys, os

HOST, USER, PASS = '192.168.11.3', 'wangxin', '123123'

def do_ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)
    return c

def upload(client, local, remote):
    with open(local, 'rb') as f:
        sftp = client.open_sftp()
        sftp.putfo(f, remote)
        sftp.close()

def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode(), stderr.read().decode()

client = do_ssh()

# === Kill by port 8001 ===
print("=== Kill process on port 8001 ===")
ks = b'import os,signal\nfor l in open("/proc/net/tcp"):p=l.split()\nif len(p)>1 and "1F41" in p[1]:\n i=p[9]\n for pid in os.listdir("/proc"):\n  if not pid.isdigit():continue\n  try:\n   for fd in os.listdir("/proc/"+pid+"/fd"):\n    try:\n     if "socket:["+i+"]" in os.readlink("/proc/"+pid+"/fd/"+fd):\n      print("Kill",pid)\n      os.kill(int(pid),signal.SIGKILL)\n    except:pass\n  except:pass\n  except:pass\nprint("Done")'
sftp = client.open_sftp()
f = sftp.file('/tmp/kp.py','wb',-1); f.write(ks); f.close(); sftp.close()
out,_=run_cmd(client, '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/kp.py')
print(out.strip())
time.sleep(4)

# Verify port free
out,_ = run_cmd(client, 'ss -tlnp')
free = True
for l in out.split('\n'):
    if ':8001' in l:
        print("WARNING port still:", l); free = False
if free: print("Port 8001 is free")

# === Upload ALL files ===
print("\n=== Upload all files ===")
files = [
    # Training API files (on GPU at /home/wangxin/yolo-auto-training/training-api/)
    (r'E:\yolo-auto-training\startup_gpu.py', '/home/wangxin/yolo-auto-training/training-api/startup.py'),
    (r'E:\yolo-auto-training\src\training\runner.py', '/home/wangxin/yolo-auto-training/training-api/src/training/runner.py'),
    (r'E:\yolo-auto-training\src\training\config.py', '/home/wangxin/yolo-auto-training/training-api/src/training/config.py'),
    (r'E:\yolo-auto-training\training-api\src\api\routes.py', '/home/wangxin/yolo-auto-training/training-api/src/api/routes.py'),
    (r'E:\yolo-auto-training\training-api\src\api\gateway.py', '/home/wangxin/yolo-auto-training/training-api/src/api/gateway.py'),
    (r'E:\yolo-auto-training\training-api\src\training\__init__.py', '/home/wangxin/yolo-auto-training/training-api/src/training/__init__.py'),
    # Deployment modules
    (r'E:\yolo-auto-training\training-api\src\deployment\exporter.py', '/home/wangxin/yolo-auto-training/training-api/src/deployment/exporter.py'),
    (r'E:\yolo-auto-training\training-api\src\deployment\validator.py', '/home/wangxin/yolo-auto-training/training-api/src/deployment/validator.py'),
    # Inference modules (required to avoid import errors)
    (r'E:\yolo-auto-training\training-api\src\inference\__init__.py', '/home/wangxin/yolo-auto-training/training-api/src/inference/__init__.py'),
    (r'E:\yolo-auto-training\training-api\src\inference\engine.py', '/home/wangxin/yolo-auto-training/training-api/src/inference/engine.py'),
    (r'E:\yolo-auto-training\training-api\src\inference\batch.py', '/home/wangxin/yolo-auto-training/training-api/src/inference/batch.py'),
    # MLflow tracker (required by runner.py)
    (r'E:\yolo-auto-training\training-api\src\training\mlflow_tracker.py', '/home/wangxin/yolo-auto-training/training-api/src/training/mlflow_tracker.py'),
    # Legacy src (fix for NameError: 'List' not defined in EdgeDeployer)
    (r'E:\yolo-auto-training\src\deployment\exporter.py', '/home/wangxin/yolo-auto-training/src/deployment/exporter.py'),
]
for local, remote in files:
    upload(client, local, remote)
    print(f"  {os.path.basename(local)} -> OK")

# === Clear cache ===
print("\n=== Clear bytecode cache ===")
cs = b'import os,shutil\nfrom pathlib import Path\nfor pyc in Path("/home/wangxin/yolo-auto-training/training-api").rglob("*.pyc"):\n try:pyc.unlink()\n except:pass\nfor c in Path("/home/wangxin/yolo-auto-training/training-api").rglob("__pycache__"):\n shutil.rmtree(c,ignore_errors=True)\nprint("Done")'
sftp = client.open_sftp()
f = sftp.file('/tmp/cc3.py','wb',-1); f.write(cs); f.close(); sftp.close()
out,_=run_cmd(client, '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/cc3.py')
print(out.strip())

# === Start Training API (background) ===
print("\n=== Starting Training API ===")
VENV = '/home/wangxin/yolo-auto-training/training-venv/bin/python'
startup = (
    "cd /home/wangxin/yolo-auto-training && "
    "CUDA_VISIBLE_DEVICES=1 "
    "PYTHONPATH=/home/wangxin/yolo-auto-training/training-api/src "
    "JWT_SECRET_KEY=yolo-training-secret-key-2024 "
    "INTERNAL_API_KEY=5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE "
    "nohup "
    + VENV + " -m uvicorn training-api.src.api.gateway:app "
    "--host 0.0.0.0 --port 8001 "
    "> /tmp/training-api.log 2>&1 &"
)
client.exec_command(startup)
print("  Startup command sent")
print("Start command sent")
time.sleep(8)

# === Health check ===
print("\n=== Health check ===")
try:
    r = httpx.get('http://192.168.11.3:8001/health', timeout=10)
    print(f"  {r.status_code} - {r.text}")
except Exception as e:
    print(f"  FAILED: {e}")

out,_ = run_cmd(client, 'tail -10 /tmp/training-api.log')
print("\n=== Training API Log ===")
print(out)
client.close()

# === E2E ===
print("\n=== Running E2E ===")
env = os.environ.copy()
env['JWT_SECRET_KEY'] = '48ef2bj3k0HQ_afGMXtRTzCevYxdAHu9mzkKgW7rmdI'
env['REDIS_URL'] = 'redis://192.168.11.134:6379/0'
env['REDIS_PASSWORD'] = '123456'
env['TRAINING_API_URL'] = 'http://192.168.11.3:8001'
env['TRAINING_API_KEY'] = '5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE'
env['BUSINESS_API_KEY'] = '5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE'
env['DEEPSEEK_API_KEY'] = 'sk-689dfd47f63b4f99a04b8e14958bb1f5'
env['DEEPSEEK_BASE_URL'] = 'https://api.deepseek.com/v1'
env['DEEPSEEK_MODEL'] = 'deepseek-reasoner'

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'src.api.gateway:app', '--host', '0.0.0.0', '--port', '8002'],
    env=env, cwd=r'E:\yolo-auto-training\business-api',
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
print(f"Business API PID: {proc.pid}")
time.sleep(6)

try:
    resp = httpx.get('http://localhost:8002/health', timeout=15)
    print(f"Health: {resp.status_code}")

    resp = httpx.post('http://localhost:8002/api/v1/train/submit',
        json={
            'project_name': 'coco8-e2e', 'model': 'yolo11n', 'epochs': 3,
            'imgsz': 320, 'data_yaml': 'coco8.yaml',
        },
        headers={'X-API-Key': '5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE'},
        timeout=30
    )
    print(f"Submit: {resp.status_code}")
    if resp.status_code == 200:
        task_id = resp.json().get('task_id')
        print(f"Task ID: {task_id}")
        for i in range(60):
            time.sleep(10)
            resp = httpx.get(f'http://localhost:8002/api/v1/train/status/{task_id}',
                headers={'X-API-Key': '5M2oDsEfm0KxwSwFhLDtsq77FGztUY9DapuwQPx0fSE'},
                timeout=10
            )
            if resp.status_code != 200:
                print(f"[{i+1}] Status fail: {resp.status_code}")
                break
            data = resp.json()
            s = data.get('status', '?')
            p = data.get('progress', 0)
            print(f"[{i+1}] {s} {p}%")
            if s in ('completed', 'failed'):
                print(f"\n=== {s.upper()} ===")
                import json
                print(json.dumps(data, indent=2))
                break
        else:
            print("Timed out after 10 min")
finally:
    c2 = do_ssh()
    out,_ = run_cmd(c2, 'tail -100 /tmp/training-api.log')
    print("\n=== GPU Log ===")
    try: print(out)
    except: pass
    c2.close()
    proc.terminate()
    try: proc.wait(5)
    except: proc.kill()
    print("Done.")
