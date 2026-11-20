"""Small, additive manifests; no scientific output is modified."""
import importlib.metadata
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def source_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_changes(path):
    try:
        diff = subprocess.check_output(
            ["git", "-C", str(path), "diff", "HEAD", "--"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return {"dirty": bool(diff), "diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(path, settings, inputs=()):
    """Record resolved settings and input identities without hashing giant slides."""
    versions = {}
    for name in ("numpy", "pandas", "scanpy", "anndata", "torch", "scikit-learn", "hest", "trident", "gutdecoder"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identities = []
    for item in inputs:
        item = Path(item)
        stat = item.stat()
        identities.append({"path": str(item.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for filename in sorted(source.rglob("*.py")):
        digest.update(str(filename.relative_to(source)).encode())
        digest.update(filename.read_bytes())
    payload = {"python": platform.python_version(), "settings": settings,
               "versions": versions, "source_revision": source_revision(Path(__file__).parent),
               "source_changes": source_changes(Path(__file__).parent),
               "source_sha256": digest.hexdigest(),
               "inputs": identities}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, sort_keys=True, indent=2) + "\n")
    return payload
