"""
SSH Operations for GPU server interactions.
Location: business-api/src/agents/ssh_ops.py

Contains standalone SSH helper methods for:
- Dataset existence checking
- Roboflow dataset download
- COCO built-in dataset download and YOLO format conversion
- data.yaml generation
"""

import os

from .operation_policy import require_operation_allowed


def get_ssh_credentials() -> tuple[str, str, str]:
    """Get SSH credentials from environment.

    Returns:
        tuple: (host, username, password)

    Raises:
        ValueError: If credentials are not configured
    """
    ssh_host = os.getenv("GPU_SERVER_HOST")
    ssh_user = os.getenv("GPU_SERVER_USER")
    ssh_pass = os.getenv("GPU_SERVER_PASS")

    if not ssh_host or not ssh_user or not ssh_pass:
        raise ValueError(
            "SSH credentials not configured. Set GPU_SERVER_HOST, GPU_SERVER_USER, "
            "and GPU_SERVER_PASS environment variables."
        )

    return ssh_host, ssh_user, ssh_pass


def check_dataset_exists(dataset_path: str, source: str = "roboflow") -> bool:
    """Check if dataset already exists at the given path with train images.

    For coco_builtin source, tries multiple known path variants to find existing data.
    """
    import paramiko

    require_operation_allowed("ssh_dataset_check", context={"dataset_path": dataset_path, "source": source})
    ssh_host, ssh_user, ssh_pass = get_ssh_credentials()

    # For coco_builtin, try multiple known path variants
    paths_to_try = [dataset_path]
    if source == "coco_builtin":
        paths_to_try.extend([
            "/home/wangxin/data/coco_person",
            "/home/wangxin/data/COCO_Person_BuiltIn",
            "/home/wangxin/data/COCO-Person-BuiltIn",
        ])

    for path in paths_to_try:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=10)
            stdin, stdout, stderr = client.exec_command(
                '/home/wangxin/yolo-auto-training/training-venv/bin/python -c "'
                'import pathlib; p=pathlib.Path(r\\\"' + path + '\\\"); '
                'train=list((p/\\\"train\\\"/\\\"images\\\").glob(\\\"*.jpg\\\"))+list((p/\\\"train\\\"/\\\"images\\\").glob(\\\"*.png\\\")); '
                'print(len(train))'
                '"',
                timeout=15
            )
            stdout.channel.recv_exit_status()
            count = int(stdout.read().decode().strip())
            print(f"[Dataset check] Found {count} train images at {path}")
            client.close()
            if count > 0:
                return True
        except Exception as e:
            print(f"[Dataset check] Error for {path}: {e}")
    return False


def download_dataset_ssh(dataset_name: str, dataset_path: str, source: str) -> None:
    """Download Roboflow dataset to GPU server via SSH."""
    import paramiko

    require_operation_allowed(
        "ssh_dataset_download",
        context={"dataset_name": dataset_name, "dataset_path": dataset_path, "source": source},
    )
    ssh_host, ssh_user, ssh_pass = get_ssh_credentials()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not set in environment")

    script = (
        "import urllib.request, urllib.error, json, zipfile, os, sys\n"
        "from pathlib import Path\n\n"
        "output_path = Path(r'" + dataset_path + "')\n"
        "output_path.mkdir(parents=True, exist_ok=True)\n\n"
        "api_key = '" + api_key + "'\n"
        "name = '" + dataset_name + "'\n\n"
        "parts = name.split('/')\n"
        "workspace = parts[0] if len(parts) > 0 else None\n"
        "project = parts[1] if len(parts) > 1 else parts[0]\n"
        "version = parts[2] if len(parts) > 2 else None\n\n"
        "if not version:\n"
        "    try:\n"
        "        meta_url = 'https://api.roboflow.com/' + workspace + '/' + project + '/info?api_key=' + api_key\n"
        "        req = urllib.request.Request(meta_url)\n"
        "        with urllib.request.urlopen(req, timeout=30) as resp:\n"
        "            meta = json.loads(resp.read())\n"
        "        versions = meta.get('versions', [])\n"
        "        if versions:\n"
        "            version = versions[-1]['id']\n"
        "            print('Latest version: ' + version)\n"
        "    except Exception as e:\n"
        "        print('Could not get version: ' + str(e))\n"
        "        raise RuntimeError('Cannot determine dataset version for ' + name)\n\n"
        "if not version:\n"
        "    raise RuntimeError('No version found for ' + name)\n\n"
        "download_url = 'https://app.roboflow.com/' + workspace + '/' + project + '/' + version + '/download?api_key=' + api_key + '&format=yolov8'\n"
        "print('Downloading ' + workspace + '/' + project + '/' + version + ' to ' + str(output_path) + '...')\n"
        "req = urllib.request.Request(download_url)\n"
        "with urllib.request.urlopen(req, timeout=600) as resp:\n"
        "    data = resp.read()\n\n"
        "zip_path = output_path / 'dataset.zip'\n"
        "with open(zip_path, 'wb') as f:\n"
        "    f.write(data)\n\n"
        "print('Extracting...')\n"
        "with zipfile.ZipFile(zip_path, 'r') as z:\n"
        "    z.extractall(output_path)\n"
        "zip_path.unlink()\n\n"
        "items = list(output_path.iterdir())\n"
        "print('Contents: ' + str([i.name for i in items]))\n"
        "for item in items:\n"
        "    if item.is_dir():\n"
        "        subdirs = [s.name for s in item.iterdir()]\n"
        "        print('Subdir ' + item.name + ' contains: ' + str(subdirs))\n\n"
        "print('Download complete!')\n"
    )

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=15)

        sftp = client.open_sftp()
        f = sftp.file('/tmp/dl_roboflow.py', 'wb', -1)
        f.write(script.encode())
        f.close()
        sftp.close()

        stdin, stdout, stderr = client.exec_command(
            '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/dl_roboflow.py 2>&1',
            timeout=700
        )
        output = stdout.read().decode(errors='replace')
        error = stderr.read().decode(errors='replace')
        client.close()

        print(f"[Download] stdout: {output}")
        if error:
            print(f"[Download] stderr: {error}")

        if 'Download complete!' not in output:
            raise RuntimeError(f"Download script failed. Output: {output[:500]}")
    except Exception as e:
        raise RuntimeError(f"SSH download failed: {e}")


def download_coco_builtin_ssh(dataset_path: str) -> None:
    """
    Download COCO val2017, filter to person class, and convert to YOLO format
    on the GPU server.  No API keys needed.
    """
    import paramiko

    require_operation_allowed(
        "dataset_download",
        context={"dataset_path": dataset_path, "source": "coco_builtin"},
    )
    ssh_host, ssh_user, ssh_pass = get_ssh_credentials()

    script = (
        "import urllib.request, zipfile, json, shutil, os\n"
        "from pathlib import Path\n\n"
        "output_path = Path(r'" + dataset_path + "')\n"
        "output_path.mkdir(parents=True, exist_ok=True)\n\n"
        "coco_cache = Path.home() / '.cache' / 'ultralytics' / 'coco'\n"
        "coco_cache.mkdir(parents=True, exist_ok=True)\n\n"
        "val_img_dir = coco_cache / 'images' / 'val2017'\n"
        "ann_dir = coco_cache / 'annotations'\n\n"
        "# Download COCO annotations (trainval2017, ~11 MB)\n"
        "ann_zip = coco_cache / 'annotations_trainval2017.zip'\n"
        "if not (ann_dir / 'instances_val2017.json').exists():\n"
        "    print('[COCO] Downloading annotations...')\n"
        "    if not ann_zip.exists():\n"
        "        urllib.request.urlretrieve(\n"
        "            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',\n"
        "            ann_zip)\n"
        "    print('[COCO] Extracting annotations...')\n"
        "    with zipfile.ZipFile(ann_zip, 'r') as z:\n"
        "        z.extractall(coco_cache)\n"
        "    ann_zip.unlink()\n\n"
        "# Download COCO val images (~300 MB)\n"
        "val_zip = coco_cache / 'val2017.zip'\n"
        "if not val_img_dir.exists() or len(list(val_img_dir.glob('*.jpg'))) < 100:\n"
        "    print('[COCO] Downloading val images (~300 MB)...')\n"
        "    urllib.request.urlretrieve(\n"
        "        'http://images.cocodataset.org/zips/val2017.zip',\n"
        "        val_zip)\n"
        "    print('[COCO] Extracting val images...')\n"
        "    with zipfile.ZipFile(val_zip, 'r') as z:\n"
        "        z.extractall(coco_cache)\n"
        "    val_zip.unlink()\n\n"
        "# Parse annotations, filter to person (cat_id=1)\n"
        "ann_file = ann_dir / 'instances_val2017.json'\n"
        "with open(ann_file) as f:\n"
        "    coco = json.load(f)\n\n"
        "img_map = {img['id']: img for img in coco['images']}\n"
        "person_bboxes = {}\n"
        "for ann in coco['annotations']:\n"
        "    if ann['category_id'] == 1 and ann.get('bbox'):\n"
        "        img_id = ann['image_id']\n"
        "        person_bboxes.setdefault(img_id, []).append(ann['bbox'])\n\n"
        "valid_ids = sorted(person_bboxes.keys())\n"
        "n = len(valid_ids)\n"
        "n_train = int(n * 0.8)\n"
        "train_ids = set(valid_ids[:n_train])\n"
        "val_ids = set(valid_ids[n_train:])\n\n"
        "train_img_d = output_path / 'train' / 'images'\n"
        "train_lbl_d = output_path / 'train' / 'labels'\n"
        "val_img_d = output_path / 'val' / 'images'\n"
        "val_lbl_d = output_path / 'val' / 'labels'\n"
        "for d in [train_img_d, train_lbl_d, val_img_d, val_lbl_d]:\n"
        "    d.mkdir(parents=True, exist_ok=True)\n\n"
        "copied = {'train': 0, 'val': 0}\n"
        "for img_id, bboxes in person_bboxes.items():\n"
        "    img_meta = img_map[img_id]\n"
        "    src = val_img_dir / img_meta['file_name']\n"
        "    if not src.exists():\n"
        "        continue\n"
        "    split = 'train' if img_id in train_ids else 'val'\n"
        "    img_d = train_img_d if split == 'train' else val_img_d\n"
        "    lbl_d = train_lbl_d if split == 'train' else val_lbl_d\n"
        "    dst_img = img_d / img_meta['file_name']\n"
        "    dst_lbl = (lbl_d / img_meta['file_name']).with_suffix('.txt')\n"
        "    shutil.copy2(src, dst_img)\n"
        "    W, H = img_meta['width'], img_meta['height']\n"
        "    lines = []\n"
        "    for x, y, w, h in bboxes:\n"
        "        xc = max(0.0, min(1.0, (x + w / 2) / W))\n"
        "        yc = max(0.0, min(1.0, (y + h / 2) / H))\n"
        "        nw = max(0.0, min(1.0, w / W))\n"
        "        nh = max(0.0, min(1.0, h / H))\n"
        "        lines.append(f'0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}')\n"
        "    with open(dst_lbl, 'w') as f:\n"
        "        f.write('\\n'.join(lines))\n"
        "    copied[split] += 1\n\n"
        "print(f'[COCO] Wrote YOLO dataset: {copied[\"train\"]} train / {copied[\"val\"]} val images')\n\n"
        "# Write data.yaml\n"
        "yaml_content = (\n"
        "    f'# COCO Person Detection (auto-generated)\\n'\n"
        "    f'# Source: http://cocodataset.org  |  License: CC BY 4.0\\n\\n'\n"
        "    f'path: {output_path.resolve()}\\n'\n"
        "    f'train: train/images\\n'\n"
        "    f'val: val/images\\n\\n'\n"
        "    f'nc: 1\\n'\n"
        "    f'names: [person]\\n'\n"
        ")\n"
        "with open(output_path / 'data.yaml', 'w') as f:\n"
        "    f.write(yaml_content)\n"
        "print('[COCO] data.yaml written.')\n"
        "print('[COCO] Download complete!')\n"
    )

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=15)

        sftp = client.open_sftp()
        f = sftp.file('/tmp/dl_coco_person.py', 'wb', -1)
        f.write(script.encode())
        f.close()
        sftp.close()

        stdin, stdout, stderr = client.exec_command(
            '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/dl_coco_person.py 2>&1',
            timeout=900
        )
        output = stdout.read().decode(errors='replace')
        error = stderr.read().decode(errors='replace')
        client.close()

        print(f"[COCO Download] stdout: {output}")
        if error:
            print(f"[COCO Download] stderr: {error}")

        if 'Download complete!' not in output:
            raise RuntimeError(f"COCO download script failed. Output: {output[:500]}")
    except Exception as e:
        raise RuntimeError(f"SSH COCO download failed: {e}")


def generate_data_yaml_ssh(dataset_path: str) -> None:
    """Generate data.yaml on GPU server based on actual dataset structure."""
    import paramiko

    require_operation_allowed("ssh_dataset_yaml", context={"dataset_path": dataset_path})
    ssh_host, ssh_user, ssh_pass = get_ssh_credentials()

    script = (
        "import os, json\n"
        "from pathlib import Path\n\n"
        "base = Path(r'" + dataset_path + "')\n\n"
        "# Find actual dataset root\n"
        "dataset_root = base\n"
        "for item in base.iterdir():\n"
        "    if item.is_dir():\n"
        "        if (item / 'train').exists() or (item / 'data.yaml').exists():\n"
        "            dataset_root = item\n"
        "            break\n\n"
        "train_dir = dataset_root / 'train' / 'images'\n"
        "val_dir = dataset_root / 'val' / 'images'\n"
        "if not val_dir.exists():\n"
        "    val_dir = dataset_root / 'valid' / 'images'\n\n"
        "yaml_path = dataset_root / 'data.yaml'\n"
        "if yaml_path.exists():\n"
        "    print('data.yaml already exists, skipping generation.')\n"
        "else:\n"
        "    # Detect classes from label files\n"
        "    import glob\n"
        "    label_files = glob.glob(str(dataset_root / 'train' / 'labels' / '*.txt'))\n"
        "    if not label_files:\n"
        "        label_files = glob.glob(str(dataset_root / 'train' / 'labels' / '*.txt'))\n\n"
        "    class_ids = set()\n"
        "    for lf in label_files[:500]:\n"
        "        with open(lf) as f:\n"
        "            for line in f:\n"
        "                parts = line.strip().split()\n"
        "                if parts:\n"
        "                    class_ids.add(int(parts[0]))\n\n"
        "    num_classes = max(class_ids) + 1 if class_ids else 1\n\n"
        "    # Generate default class names\n"
        "    names = {i: f'class_{i}' for i in range(num_classes)}\n\n"
        "    yaml_content = (\n"
        "        f'path: {dataset_root.resolve()}\\n'\n"
        "        f'train: train/images\\n'\n"
        "        f'val: val/images\\n'\n"
        "        f'nc: {num_classes}\\n'\n"
        "        f'names: {names}\\n'\n"
        "    )\n"
        "    with open(yaml_path, 'w') as f:\n"
        "        f.write(yaml_content)\n"
        "    print('Generated data.yaml: ' + yaml_content)\n"
    )

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ssh_host, username=ssh_user, password=ssh_pass, timeout=15)

        sftp = client.open_sftp()
        f = sftp.file('/tmp/gen_yaml.py', 'wb', -1)
        f.write(script.encode())
        f.close()
        sftp.close()

        stdin, stdout, stderr = client.exec_command(
            '/home/wangxin/yolo-auto-training/training-venv/bin/python /tmp/gen_yaml.py 2>&1',
            timeout=60
        )
        output = stdout.read().decode(errors='replace')
        error = stderr.read().decode(errors='replace')
        client.close()

        print(f"[YAML] stdout: {output}")
        if error:
            print(f"[YAML] stderr: {error}")
    except Exception as e:
        print(f"[YAML] Warning: {e}")
