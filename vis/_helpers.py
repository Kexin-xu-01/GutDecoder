"""
_helpers.py — pure utilities with no internal gutdecoder dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_col_ci(df: pd.DataFrame, name_variants: list[str]) -> str | None:
    """
    Return the actual column name present in df that matches any of the
    case-insensitive name_variants. Returns None if not found.
    """
    if df is None or df.empty:
        return None
    lowmap = {c.lower(): c for c in df.columns}
    for v in name_variants:
        if v is None:
            continue
        key = v.lower()
        if key in lowmap:
            return lowmap[key]
    return None


def _sample_cmap(cmap_name: str, n: int):
    """
    Return `n` RGBA colors sampled evenly from matplotlib colormap `cmap_name`.
    Safer than plt.get_cmap(name, n) on older matplotlib versions.
    """
    cmap = plt.get_cmap(cmap_name)
    # guard: at least 1 color
    if n <= 0:
        n = 1
    colors = cmap(np.linspace(0, 1, n))
    return colors


def _safe_filename(s: str) -> str:
    """Make a string safe for filenames."""
    s = str(s)
    s = re.sub(r"[^\w\-_\. ]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s.strip("_")


def _default_outdir(outdir):
    if outdir is None:
        return Path(".")
    return Path(outdir)


def _assemble_color_map(keys):
    """Return color map dict for given keys combining tab20/tab20b/tab20c."""
    cmap_tab20  = list(plt.get_cmap("tab20").colors)
    cmap_tab20b = list(plt.get_cmap("tab20b").colors)
    cmap_tab20c = list(plt.get_cmap("tab20c").colors)
    combined = cmap_tab20 + cmap_tab20b + cmap_tab20c
    if len(keys) > len(combined):
        # repeat if needed
        repeats = int(np.ceil(len(keys) / len(combined)))
        combined = combined * repeats
    return {k: combined[i] for i, k in enumerate(keys)}


def _format_title(prefix, dataset_name, encoder_name):
    return f"{prefix} | Data: {dataset_name} | Model: {encoder_name}"


def _safe_read_json(path: Path) -> dict:
    import json
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _find_col(df, candidates, required=True, what="column"):
    """Return the first column from `candidates` that exists in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find {what}. Looked for: {candidates}. "
                       f"Available: {list(df.columns)}")
    return None


def detect_patch_col_from_name(name: str) -> str:
    """Detect resolution from a run/dataset name and return a candidate col like 'num_patches_50um'."""
    import re
    # Normalize to lowercase
    s = (name or "").lower()

    # priority: 25um, 50um, 100um
    if re.search(r"25(\W|_|um|$)", s):
        res = 25
    elif re.search(r"50(\W|_|um|$)", s):
        res = 50
    else:
        # default 100um when no explicit mention
        res = 100
    return f"num_patches_{res}um"


def choose_existing_patch_col(candidate: str, meta_cols: list) -> str:
    """Return candidate if present in meta_cols; otherwise try other common names then fallback."""
    import re
    if candidate in meta_cols:
        return candidate

    # try alternative common suffix patterns
    alternatives = []
    # e.g. user might have used 'num_patches_50' or 'num_patches_50_um' etc.
    m = re.match(r"num_patches_(\d+)(um)?", candidate)
    if m:
        n = m.group(1)
        alternatives.extend([
            f"num_patches_{n}",
            f"num_patches_{n}_um",
            f"num_patches_{n}um",  # candidate itself
        ])
    # also try typical defaults
    alternatives.extend(["num_patches_100um", "num_patches_50um", "num_patches_25um", "num_patches"])

    for alt in alternatives:
        if alt in meta_cols:
            return alt

    # final fallback (keep the original candidate even if absent; generate_all_plots should handle missing values)
    return candidate
