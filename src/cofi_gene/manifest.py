"""Manifest helpers: content hashes, git revision, environment fingerprint.

Every processed dataset writes a manifest so a run can be traced back to exact
inputs. Real data, checkpoints and credentials stay out of git; only these small
JSON files are committed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "sha256_file",
    "sha256_array",
    "git_commit",
    "runtime_versions",
    "write_manifest",
    "read_manifest",
    "utc_timestamp",
]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (works for multi-GB raw downloads)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array) -> str:
    """Stable hash of a numpy array's contents and shape (for feature provenance)."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(default: str = "unknown") -> str:
    """Current HEAD sha, or ``default`` outside a git checkout."""
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return default
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and sha else default


def runtime_versions() -> dict:
    """Non-sensitive interpreter/package fingerprint (never hostnames or keys)."""
    import numpy as np

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "executable": sys.executable,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda_build"] = torch.version.cuda
        versions["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - optional dependency
        versions["torch"] = None
    return versions


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(path: str | Path, payload: dict) -> Path:
    """Write ``payload`` as sorted, indented JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
