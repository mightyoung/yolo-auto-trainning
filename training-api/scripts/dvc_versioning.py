"""
DVC-based dataset versioning for YOLO Auto-Training.

Tracks dataset versions alongside training runs for reproducibility.
Gracefully degrades when DVC is not available.
"""
import json
import logging
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def is_dvc_initialized(path: Path) -> bool:
    """
    Check if DVC is initialized in the given directory or any parent.

    Args:
        path: Directory path to check.

    Returns:
        True if .dvc directory exists in path or a parent directory.
    """
    try:
        p = Path(path).resolve()
        while p != p.parent:
            if (p / ".dvc").is_dir():
                return True
            p = p.parent
        return False
    except Exception as e:
        logger.warning(f"[dvc_versioning] is_dvc_initialized failed for {path}: {e}")
        return False


def _run_cmd(cmd: list, cwd: Optional[str] = None) -> tuple[str, int]:
    """
    Run a shell command and return (stdout, returncode).

    Args:
        cmd: Command as list of strings.
        cwd: Working directory for the command.

    Returns:
        Tuple of (stdout text, return code).
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            shell=False,
            timeout=30,
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        logger.warning(f"[dvc_versioning] Command timed out: {' '.join(cmd)}")
        return "", -1
    except Exception as e:
        logger.warning(f"[dvc_versioning] Command failed: {' '.join(cmd)}: {e}")
        return "", -1


def get_git_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        Git commit SHA (short), or "unknown" on failure.
    """
    stdout, rc = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
    if rc == 0 and stdout:
        return stdout
    return "unknown"


def get_dataset_hash(dataset_path: str) -> str:
    """
    Get a content hash for a dataset directory.

    If the directory is DVC-initialized, runs 'dvc diff --json' to get
    a DVC-specific hash. Falls back to hashing the directory listing
    for non-DVC datasets.

    Args:
        dataset_path: Path to the dataset directory (containing data.yaml).

    Returns:
        A hex hash string representing the dataset version.
    """
    ds_path = Path(dataset_path).resolve()

    # Try DVC metrics/status first
    if is_dvc_initialized(ds_path):
        # Use 'dvc metrics diff' or 'dvc status --json' for version hash
        stdout, rc = _run_cmd(
            ["dvc", "status", "--json"],
            cwd=str(ds_path),
        )
        if rc == 0 and stdout:
            try:
                # Hash the status JSON output for a stable version identifier
                return hashlib.md5(stdout.encode()).hexdigest()[:12]
            except Exception:
                pass

        # Fallback: use 'dvc repro' hash or .dvc files
        stdout2, rc2 = _run_cmd(
            ["git", "ls-files", "*.dvc"],
            cwd=str(ds_path),
        )
        if rc2 == 0 and stdout2:
            return hashlib.md5(stdout2.encode()).hexdigest()[:12]

    # Fallback for non-DVC: hash the listing of image dirs (train/val)
    try:
        hashes: list[str] = []
        for split in ["train", "val", "test"]:
            for subdir in ["images", "labels"]:
                split_path = ds_path / split / subdir
                if split_path.exists():
                    files = sorted([f.name for f in split_path.iterdir() if f.is_file()])
                    hashes.append(f"{split}/{subdir}:" + "|".join(files[:100]))
        if hashes:
            combined = ";".join(hashes)
            return hashlib.md5(combined.encode()).hexdigest()[:12]
    except Exception as e:
        logger.warning(f"[dvc_versioning] Fallback hash failed: {e}")

    return "untracked"


def init_dvc(dataset_path: str) -> bool:
    """
    Initialize DVC tracking for a dataset directory.

    Runs 'dvc init' inside the dataset directory and commits the
    resulting .dvc and .dvcignore files to git.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        True if DVC was initialized successfully, False otherwise.
    """
    ds_path = Path(dataset_path).resolve()

    if is_dvc_initialized(ds_path):
        logger.info(f"[dvc_versioning] DVC already initialized in {dataset_path}")
        return True

    # Run dvc init
    stdout, rc = _run_cmd(["dvc", "init"], cwd=str(ds_path))
    if rc != 0:
        logger.error(f"[dvc_versioning] dvc init failed: {stdout}")
        return False

    # Try to commit .dvc to git
    git_add, ga_rc = _run_cmd(["git", "add", ".dvc", ".dvcignore"], cwd=str(ds_path))
    if ga_rc == 0:
        _run_cmd(
            ["git", "commit", "-m", "Initialize DVC for dataset versioning"],
            cwd=str(ds_path),
        )

    logger.info(f"[dvc_versioning] DVC initialized in {dataset_path}")
    return True


def record_version(dataset_path: str, output_file: str) -> bool:
    """
    Record the current dataset version to a JSON file.

    Writes a JSON document with git_hash, dvc_hash, and timestamp
    to the output file. Creates parent directories as needed.

    Args:
        dataset_path: Path to the dataset directory.
        output_file: Path to the output JSON file.

    Returns:
        True if the file was written successfully, False on error.
    """
    try:
        git_hash = get_git_hash()
        dvc_hash = get_dataset_hash(dataset_path)

        version_info: Dict[str, str] = {
            "git_hash": git_hash,
            "dvc_hash": dvc_hash,
            "dataset_path": str(Path(dataset_path).resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(version_info, indent=2), encoding="utf-8")
        logger.info(
            f"[dvc_versioning] Recorded dataset version to {output_file}: "
            f"git={git_hash} dvc={dvc_hash}"
        )
        return True

    except Exception as e:
        logger.warning(f"[dvc_versioning] record_version failed: {e}")
        return False
