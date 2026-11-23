"""
QC & patch-count utilities for HEST-derived Xenium / Visium runs
----------------------------------------------------------------
Purpose
  - Scan slide folders, read aligned_adata.h5ad and patches/*.h5, and produce
    per-sample QC summaries and patch counts that can be merged into metadata.

Key behaviors
  - Infers "in-tissue" spots by exact barcode matching between patch H5 files
    and adata.obs.index (preferred). Falls back to obs['in_tissue'] only if
    no patch barcodes are found.
  - All reported means are rounded to 3 decimal places by default.
  - Recognises runs named exactly: XeniumPR<digits>, XeniumR<digits>, VisiumR<digits>.

Typical layout (example)
  /.../xenium_data/XeniumPR6/slide1/<sample_folder>/
      aligned_adata.h5ad
      patches/patches.h5   (contains 'barcode' and 'coords' datasets)

Quick examples
  # single slide QC summary (saves qc_summary.csv in slide folder)
  df = return_qc("/path/to/XeniumPR6/slide1", save_csv="qc_summary.csv")

  # integrate QC across all runs under xenium_root and visium_root into `metadata`
  out = add_qc_from_all_runs(metadata_df, xenium_root="/.../xenium_data",
                             visium_root="/.../visium_data", save_csv_name="qc_summary.csv")

  # build merged patch-count metrics and merge into metadata
  out = build_merged_counts(metadata_df, specs, count_func=count_patches, auto_prefix=True)

Notes / requirements
  - The functions expect HEST-style patch files created by dump_patches() that include
    an extra 'barcode' asset (fast, exact linking).
  - If your files use different names/structure, inspect the H5 keys and adapt
    `_read_barcodes_from_patch_h5` candidate keys accordingly.
"""


from pathlib import Path
from typing import List, Union, Optional, Sequence, Dict, Any, List, Callable, Set
import pandas as pd
import re
import scanpy as sc
from IPython.display import display
import numpy as np
import h5py
import warnings
from loguru import logger


def _maybe_prefix(sample_id: str, prefix: Optional[str]) -> str:
    """Add prefix to sample_id if not already present."""
    if not prefix:
        return sample_id
    return sample_id if sample_id.startswith(prefix) else f"{prefix}{sample_id}"


def _derive_prefix_from_path(p: Union[str, Path]) -> Optional[str]:
    """
    Heuristic to derive a prefix from a path like:
      /.../XeniumPR1/slide1  -> "XeniumPR1S1"
      /.../XeniumPR1_50um/slide2 -> "XeniumPR1_50umS2"
    Returns None if no reasonable prefix can be derived.
    """
    p = Path(p)
    parts = [pp for pp in p.parts if pp]  # list of path parts
    if not parts:
        return None
    # prefer parent + slide name
    parent = p.parent.name
    name = p.name
    # match slide number from folder name like slide1 or s1 or S1
    m = re.search(r"slide[_\-]?(\d+)|s[_\-]?(\d+)|S[_\-]?(\d+)", name)
    if m:
        # get group non-empty
        slide_num = next(g for g in m.groups() if g is not None)
        if parent:
            return f"{parent}S{slide_num}"
        else:
            return f"S{slide_num}"
    # fallback: if parent contains PR and number, use parent
    if parent:
        m2 = re.search(r"(XeniumPR\w+|XeniumPR\d+|PR\d+|PR\w+)", parent)
        if m2:
            return m2.group(1)
    return None


def _load_and_format_count(
    path: Union[str, Path],
    prefix: Optional[str],
    rename_col: Optional[str],
    count_func,
    save_csv: Optional[Union[str, Path]] = None,
    auto_prefix: bool = False,
) -> pd.DataFrame:
    """
    Call count_func(path), prefix sample_id (safely), and rename numeric column to rename_col.
    """
    if count_func is None:
        raise ValueError("count_func cannot be None")
    df = count_func(str(path), save_csv=save_csv)
    if "sample_id" not in df.columns:
        raise ValueError(f"count_func returned DataFrame without 'sample_id'. Columns: {df.columns.tolist()}")

    # determine effective prefix
    eff_prefix = prefix
    if (not eff_prefix) and auto_prefix:
        eff_prefix = _derive_prefix_from_path(path)

    if eff_prefix:
        df["sample_id"] = df["sample_id"].astype(str).apply(lambda s: _maybe_prefix(s, eff_prefix))

    # rename numeric column to rename_col
    if rename_col:
        if "num_patches" in df.columns:
            df = df.rename(columns={"num_patches": rename_col})
        else:
            # fallback: pick a single numeric column
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) == 1:
                df = df.rename(columns={numeric_cols[0]: rename_col})
            elif len(numeric_cols) == 0:
                raise ValueError(f"No numeric column found to rename in results for path {path}")
            else:
                # if multiple numeric columns, prefer a column named like 'num_patches' patterns
                candidate = None
                for c in numeric_cols:
                    if "patch" in c or "num" in c:
                        candidate = c
                        break
                if candidate:
                    df = df.rename(columns={candidate: rename_col})
                else:
                    raise ValueError(f"Multiple numeric columns found in {path}; please ensure count_func returns 'num_patches' or pass rename_col=None")
    return df


def _concat_counts_for_paths(
    paths: Sequence[Union[str, Path]],
    prefixes: Optional[Sequence[Optional[str]]],
    rename_col: Optional[str],
    count_func,
    save_csv: Optional[Union[str, Path]] = None,
    auto_prefix: bool = False,
) -> pd.DataFrame:
    """
    For a set of folder paths, call count_func for each, format, and concat.
    prefixes may be None (apply derived prefix if auto_prefix True) or a sequence matching paths.
    """
    dfs = []
    if prefixes is None:
        prefixes = [None] * len(paths)
    if len(prefixes) != len(paths):
        raise ValueError("prefixes must have same length as paths (or be None)")

    for p, pref in zip(paths, prefixes):
        p = Path(p)
        if not p.exists():
            # skip and warn
            logger.warning(f"path not found, skipping: {p}")
            continue
        df = _load_and_format_count(p, prefix=pref, rename_col=rename_col, count_func=count_func, save_csv=save_csv, auto_prefix=auto_prefix)
        dfs.append(df)
    if not dfs:
        # empty df with expected columns
        cols = ["sample_id"]
        if rename_col:
            cols.append(rename_col)
        return pd.DataFrame(columns=cols)
    return pd.concat(dfs, ignore_index=True)


def build_merged_counts(
    metadata: pd.DataFrame,
    specs: Sequence[Dict[str, Any]],
    count_func,
    save_csv: Optional[Union[str, Path]] = None,
    auto_prefix: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Build merged dataframe by running count_func over many folder groups (specs) and merging into metadata.

    Args:
      - metadata: DataFrame with 'sample_id' column to merge into.
      - specs: list of dicts. Each dict may contain:
          - 'paths': str | Path | list[str|Path]  (required)
          - 'prefixes': optional list[str|None] same length as paths (or a single str to apply to all)
          - 'rename_col': str (the column name to create in metadata; optional but recommended)
      - count_func: function(path, save_csv=None) -> DataFrame with 'sample_id' and numeric column (default 'num_patches')
      - save_csv: forwarded to count_func (optional)
      - auto_prefix: if True, tries to derive prefix from the folder path when prefix not provided
      - verbose: print progress if True

    Returns:
      dict with:
        - 'merged': merged DataFrame
        - 'per_metric': dict mapping rename_col-> concatenated DataFrame (so you can inspect)
    """
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must have a 'sample_id' column")

    merged = metadata.copy()
    per_metric = {}

    for spec in specs:
        paths = spec.get("paths")
        if paths is None:
            raise ValueError("Each spec must have 'paths'")

        # normalize paths to list
        if isinstance(paths, (str, Path)):
            paths_list = [paths]
        else:
            paths_list = list(paths)

        # prefixes handling: allow single string or list
        prefixes = spec.get("prefixes")
        if isinstance(prefixes, (str, Path)):
            prefixes = [str(prefixes)] * len(paths_list)
        elif prefixes is None:
            prefixes = None
        else:
            prefixes = list(prefixes)

        rename_col = spec.get("rename_col")  # may be None
        df_counts = _concat_counts_for_paths(paths_list, prefixes, rename_col, count_func, save_csv=save_csv, auto_prefix=auto_prefix)

        # store for inspection
        key = rename_col or f"metric_{len(per_metric)+1}"
        per_metric[key] = df_counts

        if df_counts.empty:
            if verbose:
                logger.info(f"no data for metric '{key}' (paths: {paths_list}) -- skipping merge")
            continue

        # merge into merged DataFrame
        merged = pd.merge(merged, df_counts, on="sample_id", how="left")
        if verbose:
            logger.info(f"merged metric '{key}' ({len(df_counts)} rows) into metadata; merged shape now {merged.shape}")

    return {"merged": merged, "per_metric": per_metric}


def _read_barcodes_from_patch_h5(h5_path: Path) -> List[str]:
    """
    Read the barcode dataset from a patch .h5 file as a list of strings.

    The dataset is the first key (top level or one group down) whose name contains
    "barcode", e.g. 'barcode' or 'barcodes'. Values may be bytes or str, shaped (N,) or (N, 1).
    Returns [] with a warning when there is no such dataset, so the caller can fall back to
    the adata `in_tissue` column visibly rather than silently.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return []

    with h5py.File(h5_path, "r") as f:
        keys = []
        for k in f.keys():
            keys.append(k)
            if isinstance(f[k], h5py.Group):
                keys.extend(f"{k}/{kk}" for kk in f[k].keys())

        barcode_key = next((k for k in keys if "barcode" in k.lower()), None)
        if barcode_key is None:
            warnings.warn(f"No barcode dataset in {h5_path} (keys: {keys})")
            return []
        if not isinstance(f[barcode_key], h5py.Dataset):
            warnings.warn(f"'{barcode_key}' in {h5_path} is a group, not a barcode dataset")
            return []
        arr = f[barcode_key][()]

    # flatten shapes like (N,1) -> (N,)
    if arr.ndim > 1 and arr.shape[1] == 1:
        arr = arr[:, 0]

    def to_str(x):
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", errors="ignore")
        # nested sequences like ['007x211']
        if isinstance(x, (list, tuple, np.ndarray)):
            return to_str(x[0]) if len(x) else None
        return str(x)

    return [b.strip() for b in map(to_str, arr) if b is not None]


def _collect_patch_barcodes_from_dir(patches_dir: Path) -> List[str]:
    """
    Read all '*.h5' files under patches_dir and return unique barcodes preserving order.
    """
    patches_dir = Path(patches_dir)
    if not patches_dir.exists() or not patches_dir.is_dir():
        return []
    seen = set()
    out = []
    for p in sorted(patches_dir.glob("*.h5")):
        try:
            barcodes = _read_barcodes_from_patch_h5(p)
        except OSError as e:
            # h5py raises OSError for truncated or unreadable files; skip that file only.
            warnings.warn(f"Could not open {p}, skipping its barcodes: {e}")
            continue
        for b in barcodes:
            if b is None:
                continue
            if b not in seen:
                seen.add(b)
                out.append(b)
    return out


def return_qc(
    root,
    save_csv: Optional[str] = None,
    patch_subdir: str = "patches",
    patch_h5_name: str = "patches.h5",
    round_decimals: int = 3,
):
    """
    Scan `root` (a slide directory) for sample subfolders containing aligned_adata.h5ad and patches/,
    infer in-tissue spots using patch barcodes (preferred), and return QC summary.

    Args:
        root: str or Path to slide directory containing sample subfolders.
        save_csv: optional filename (if relative, saved under `root`). If provided, saves the table.
        patch_subdir: name of the folder with patch files under each sample (default 'patches')
        patch_h5_name: name of patch h5 (default 'patches.h5')
        round_decimals: how many decimals to round means to (default 3)

    Returns:
        pandas.DataFrame with columns:
          sample_id, n_obs, mean_total_counts, mean_n_genes_by_counts,
          mean_log1p_total_counts, mean_log1p_n_genes_by_counts,
          inferred_in_tissue_from_patch_barcodes (bool), num_patch_barcodes,
          n_obs_in_tissue, mean_..._in_tissue, pct_in_tissue
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    results = []

    def safe_mean(series):
        return round(float(pd.Series(series).mean()), round_decimals)
    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue

        # path to aligned_adata.h5ad (keep same filename you use)
        adata_candidates = list(sample_dir.glob("aligned_adata.h5ad")) + list(sample_dir.glob("*.h5ad"))
        # prefer aligned_adata.h5ad if present
        adata_file = None
        for c in adata_candidates:
            if c.name == "aligned_adata.h5ad":
                adata_file = c
                break
        if adata_file is None and adata_candidates:
            adata_file = adata_candidates[0]

        if adata_file is None:
            # nothing to do for this folder
            continue

        sample_id = sample_dir.name
        adata = sc.read_h5ad(adata_file)

        obs = adata.obs

        # collect patch barcodes (fast, exact linking). look under sample_dir/patches/*.h5
        patches_dir = sample_dir / patch_subdir
        barcodes_in_patches = []
        if patches_dir.exists() and patches_dir.is_dir():
            # prefer reading a single 'patches.h5' if it exists, else read all .h5 files
            single = patches_dir / patch_h5_name
            if single.exists():
                barcodes_in_patches = _collect_patch_barcodes_from_dir(patches_dir)
            else:
                barcodes_in_patches = _collect_patch_barcodes_from_dir(patches_dir)

        barcode_set: Set[str] = set(barcodes_in_patches)

        # build in_tissue mask using barcode linking if available
        in_tissue_mask = None
        used_patch_barcodes = False
        if len(barcode_set) > 0:
            used_patch_barcodes = True
            # ensure adata.obs.index are strings
            adata_idx = adata.obs.index.astype(str).to_numpy()
            in_tissue_mask = np.array([ (s in barcode_set) for s in adata_idx ], dtype=bool)
        else:
            # fallback: if adata.obs has in_tissue column, use it; else None
            if "in_tissue" in obs.columns:
                in_tissue_mask = obs["in_tissue"].astype(bool).to_numpy()
            else:
                in_tissue_mask = None

        # compute summary metrics (all spots)
        summary = {
            "sample_id": sample_id,
            "n_obs": int(adata.n_obs),
            "inferred_in_tissue_from_patch_barcodes": bool(used_patch_barcodes),
        }

        # tissue-only metrics
        if in_tissue_mask is None:
            summary.update({
                "n_obs_in_tissue": None,
                "mean_total_counts_in_tissue": None,
                "mean_n_genes_by_counts_in_tissue": None,
                "mean_log1p_total_counts_in_tissue": None,
                "mean_log1p_n_genes_by_counts_in_tissue": None,
                "pct_in_tissue": None,
            })
        else:
            obs_tissue = obs.loc[in_tissue_mask]

            n_in = int(obs_tissue.shape[0])
            pct_in = round((n_in / float(obs.shape[0])) * 100.0, round_decimals) if obs.shape[0] > 0 else None

            summary.update({
                "n_obs_in_tissue": n_in,
                "mean_total_counts_in_tissue": safe_mean(obs_tissue["total_counts"]) if "total_counts" in obs_tissue else None,
                "mean_n_genes_by_counts_in_tissue": safe_mean(obs_tissue["n_genes_by_counts"]) if "n_genes_by_counts" in obs_tissue else None,
                "mean_log1p_total_counts_in_tissue": safe_mean(obs_tissue["log1p_total_counts"]) if "log1p_total_counts" in obs_tissue else None,
                "mean_log1p_n_genes_by_counts_in_tissue": safe_mean(obs_tissue["log1p_n_genes_by_counts"]) if "log1p_n_genes_by_counts" in obs_tissue else None,
                "pct_in_tissue": pct_in,
            })

        results.append(summary)

    df = pd.DataFrame(results).sort_values("sample_id").reset_index(drop=True)

    if save_csv:
        out = Path(save_csv) if Path(save_csv).is_absolute() else (root / save_csv)
        df.to_csv(out, index=False)
        logger.info(f"Saved summary to {out}")

    return df



def add_qc_from_all_runs(
    metadata: pd.DataFrame,
    xenium_root: str | Path,
    visium_root: str | Path,
    qc_func: Optional[Callable] = None,
    save_csv_name: Optional[str] = "qc_summary.csv",
    update_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Scan both xenium_root and visium_root for run folders exactly matching:
        XeniumPR<digits>
        XeniumR<digits>
        VisiumR<digits>

    For each run:
        - process slide1 and slide2
        - call qc_func on slide path
        - prefix sample_id with RunNameS{slide_idx}
        - merge QC columns into metadata

    Returns:
        {
            "metadata": updated_metadata_df,
            "processed_runs": [...],
        }
    """

    xenium_root = Path(xenium_root)
    visium_root = Path(visium_root)

    # strict matching
    patterns = [
        re.compile(r"^XeniumPR\d+$"),
        re.compile(r"^XeniumR\d+$"),
        re.compile(r"^VisiumR\d+$"),
    ]

    # collect all valid run dirs from both roots
    run_dirs: List[Path] = []

    for root in [xenium_root, visium_root]:
        if not root.exists():
            continue
        for d in root.iterdir():
            if d.is_dir() and any(p.match(d.name) for p in patterns):
                run_dirs.append(d)

    if verbose:
        logger.info(f"Found {len(run_dirs)} valid runs across Xenium + Visium: {sorted(str(r) for r in run_dirs)}")

    if qc_func is None:
        qc_func = return_qc

    expected_cols = [
        "n_obs",
        "n_obs_in_tissue",
        "mean_total_counts_in_tissue",
        "mean_n_genes_by_counts_in_tissue",
        "mean_log1p_total_counts_in_tissue",
        "mean_log1p_n_genes_by_counts_in_tissue",
    ]

    meta = metadata.copy(deep=True)

    # ensure QC columns exist
    for col in expected_cols:
        if col not in meta.columns:
            meta[col] = pd.NA

    processed_runs = []
    failures = []
    new_rows = []

    for run_dir in sorted(run_dirs):
        run_name = run_dir.name
        processed_runs.append(run_name)

        for slide_idx, slide_name in enumerate(["slide1", "slide2"], start=1):
            slide_path = run_dir / slide_name
            if not slide_path.exists():
                continue

            if verbose:
                logger.info(f"{run_name} - {slide_name}")

            try:
                if save_csv_name:
                    df_qc = qc_func(str(slide_path), save_csv=save_csv_name)
                else:
                    df_qc = qc_func(str(slide_path))
            except Exception as e:
                failures.append(f"{slide_path}: {type(e).__name__}: {e}")
                continue

            if df_qc is None or df_qc.empty:
                if verbose:
                    logger.info("  No QC rows found.")
                continue

            if "sample_id" not in df_qc.columns:
                df_qc = df_qc.reset_index().rename(columns={"index": "sample_id"})

            # prefix with RunNameS{slide}
            prefix = f"{run_name}S{slide_idx}"
            df_qc["sample_id"] = prefix + df_qc["sample_id"].astype(str)

            for col in expected_cols:
                if col not in df_qc.columns:
                    df_qc[col] = pd.NA

            df_qc = df_qc[["sample_id"] + expected_cols]

            if verbose:
                display(df_qc)

            # merge/update metadata
            for _, row in df_qc.iterrows():
                sid = row["sample_id"]

                if sid in meta["sample_id"].values:
                    if update_existing:
                        for col in expected_cols:
                            meta.loc[meta["sample_id"] == sid, col] = row[col]
                    else:
                        for col in expected_cols:
                            mask = meta["sample_id"] == sid
                            meta.loc[mask, col] = meta.loc[mask, col].fillna(row[col])
                else:
                    new_rows.append({"sample_id": sid, **{col: row[col] for col in expected_cols}})

    if new_rows:
        # Rebuild once instead of concatenating mostly-NA rows, which pandas is deprecating.
        meta = pd.DataFrame(meta.to_dict("records") + new_rows, columns=meta.columns)

    if failures:
        raise RuntimeError(f"QC failed on {len(failures)} slide(s):\n" + "\n".join(failures))

    meta = meta.sort_values("sample_id").reset_index(drop=True)

    logger.info("QC integration complete.")
    return {"metadata": meta, "processed_runs": processed_runs}
