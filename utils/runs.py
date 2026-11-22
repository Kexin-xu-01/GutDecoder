"""Registry of the canonical runs used in figures (metadata/runs.yaml in the hest project)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from gutdecoder.config import RESULTS_ROOT

__all__ = ["RUNS_YAML", "load_runs"]

RUNS_YAML = os.environ.get("GUTDECODER_RUNS_YAML", "/project/gutdecoder/kxu/hest/metadata/runs.yaml")


def load_runs(path: str | Path = RUNS_YAML, check_exists: bool = True) -> dict[str, str]:
    """
    {label: run_id} from the registry. With check_exists, every run folder must exist under
    RESULTS_ROOT; a missing one raises FileNotFoundError listing all of them.
    """
    with open(path) as f:
        runs = yaml.safe_load(f)
    if not isinstance(runs, dict) or not all(isinstance(v, str) for v in runs.values()):
        raise ValueError(f"{path} must map labels to run IDs")
    if check_exists:
        missing = {k: v for k, v in runs.items() if not (Path(RESULTS_ROOT) / v).is_dir()}
        if missing:
            raise FileNotFoundError(f"runs not found under {RESULTS_ROOT}: {missing}")
    return runs
