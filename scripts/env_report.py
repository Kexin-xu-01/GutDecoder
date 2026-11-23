"""
Report the installed versions gutdecoder depends on, check that tiffslide imports, and print
the dependency pins that reproduce this environment.

    module load python-cbrg/202510
    python scripts/env_report.py
"""
import importlib
import importlib.metadata as md
import importlib.util
import json
import subprocess
from pathlib import Path

PACKAGES = ["tifffile", "tiffslide", "zarr", "hest", "trident", "timm", "torch", "torchvision",
            "numpy", "pandas", "scanpy", "anndata", "openslide-python", "torchstain", "loguru"]
GIT_REPOS = {"hest": "https://github.com/mahmoodlab/HEST.git",
             "trident": "https://github.com/mahmoodlab/TRIDENT.git"}


def installed(name):
    """(version, location, git commit or None), or None if not installed."""
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return None
    direct_url = dist.read_text("direct_url.json")
    commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None
    return dist.version, str(dist.locate_file("")), commit


def import_check(module):
    try:
        importlib.import_module(module)
        return "ok"
    except Exception as e:
        return f"FAILS: {type(e).__name__}: {e}"


print("Installed packages")
info = {}
for name in PACKAGES:
    info[name] = installed(name)
    if info[name] is None:
        print(f"  {name:17s} not installed")
    else:
        version, location, commit = info[name]
        print(f"  {name:17s} {version:14s} {location}" + (f"  git {commit}" if commit else ""))

print("\nImport checks")
for module in ("tiffslide",):
    print(f"  import {module:14s} {import_check(module)}")

def checkout_commit(module):
    """(commit, has local changes) of the git checkout a module is imported from, or None."""
    spec = importlib.util.find_spec(module)  # locates the package without importing it
    if spec is None or spec.origin is None:
        return None
    folder = Path(spec.origin).parent
    run = lambda *args: subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True)
    head = run("rev-parse", "HEAD")
    if head.returncode != 0:  # not inside a git checkout
        return None
    return head.stdout.strip(), bool(run("status", "--porcelain", "--untracked-files=no").stdout.strip())


print("\nPins for pyproject.toml [project.dependencies]")
for name, repo in GIT_REPOS.items():
    commit = info[name][2] if info[name] else None
    dirty = False
    if info[name] and not commit:
        found = checkout_commit(name)
        if found:
            commit, dirty = found
    if commit:
        print(f'  "{name} @ git+{repo}@{commit}",' + ("   # WARNING: checkout has local changes" if dirty else ""))
    else:
        print(f"  # {name}: no git commit found")
