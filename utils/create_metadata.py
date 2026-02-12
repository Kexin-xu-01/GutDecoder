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
            print(f"[warning] path not found, skipping: {p}")
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

        # call and concat counts
        df_counts = _concat_counts_for_paths(paths_list, prefixes, rename_col, count_func, save_csv=save_csv, auto_prefix=auto_prefix)

        # store for inspection
        key = rename_col or f"metric_{len(per_metric)+1}"
        per_metric[key] = df_counts

        if df_counts.empty:
            if verbose:
                print(f"[info] no data for metric '{key}' (paths: {paths_list}) -- skipping merge")
            continue

        # merge into merged DataFrame
        merged = pd.merge(merged, df_counts, on="sample_id", how="left")
        if verbose:
            print(f"[info] merged metric '{key}' ({len(df_counts)} rows) into metadata; merged shape now {merged.shape}")

    return {"merged": merged, "per_metric": per_metric}


def _read_barcodes_from_patch_h5(h5_path: Path) -> List[str]:
    """
    Robustly read a 'barcode' array from a patch .h5 file.
    Handles:
      - datasets named 'barcode', 'barcodes', or nested under groups
      - byte strings, shape (N,1) or (N,) arrays, arrays of arrays
    Returns a list of unicode strings (may be empty).
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return []

    with h5py.File(h5_path, "r") as f:
        # try direct known keys first
        candidate_keys = []
        for k in f.keys():
            candidate_keys.append(k)
            # also consider nested
            if isinstance(f[k], h5py.Group):
                for kk in f[k].keys():
                    candidate_keys.append(f"{k}/{kk}")

        # prefer exact 'barcode' substring matches (case-insensitive)
        barcode_key = None
        for key in candidate_keys:
            if "barcode" in key.lower():
                barcode_key = key
                break

        arr = None
        if barcode_key is not None:
            # access possibly nested key
            try:
                arr = np.array(f[barcode_key])
            except Exception:
                arr = None
        else:
            # fallback: scan for any 1D/2D dataset with string-like dtype or small object dtype
            def walk_find(group):
                for name, item in group.items():
                    if isinstance(item, h5py.Dataset):
                        try:
                            a = np.array(item)
                        except Exception:
                            continue
                        if a is None:
                            continue
                        # Accept 1-D of strings or (N,1) nested structures
                        if (a.ndim == 1) or (a.ndim == 2 and a.shape[1] == 1):
                            # if dtype string-like or object, accept as candidate
                            if a.dtype.kind in ("S", "U", "O", "a"):
                                return a
                        # sometimes stored as (N,1) bytes in dtype 'S'
                        if a.ndim == 2 and a.shape[1] == 1:
                            return a
                    elif isinstance(item, h5py.Group):
                        res = walk_find(item)
                        if res is not None:
                            return res
                return None
            arr = walk_find(f)

        if arr is None:
            return []

        # flatten shapes like (N,1) -> (N,)
        if arr.ndim > 1 and arr.shape[1] == 1:
            arr = arr[:, 0]

        # convert bytes/arrays -> str
        def to_str(x):
            if x is None:
                return None
            # bytes -> decode
            if isinstance(x, (bytes, bytearray)):
                try:
                    return x.decode("utf-8", errors="ignore")
                except Exception:
                    return str(x)
            # numpy.bytes_ etc.
            if isinstance(x, (np.bytes_, np.str_)):
                try:
                    return str(x)
                except Exception:
                    return None
            # nested sequences like ['007x211']
            if isinstance(x, (list, tuple, np.ndarray)):
                if len(x) == 0:
                    return None
                return to_str(x[0])
            # otherwise cast
            return str(x)

        out = []
        for el in arr:
            s = to_str(el)
            if s is None:
                continue
            s = s.strip()
            out.append(s)
        return out


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
        except Exception as e:
            warnings.warn(f"[WARN] Failed to read barcodes from {p}: {e}")
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
        try:
            return round(float(pd.Series(series).mean()), round_decimals)
        except Exception:
            return None

    # iterate sample folders
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

        # load adata
        try:
            adata = sc.read_h5ad(adata_file)
        except Exception as e:
            warnings.warn(f"[WARN] Failed to read {adata_file}: {e}")
            continue

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
            if len(in_tissue_mask) != obs.shape[0]:
                warnings.warn(f"[WARN] in_tissue mask length mismatch for {adata_file} ({len(in_tissue_mask)} vs {obs.shape[0]})")
                obs_tissue = obs.iloc[[]]
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
        print(f"[INFO] Saved summary to {out}")

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
        print(f"Found {len(run_dirs)} valid runs across Xenium + Visium:")
        for r in sorted(run_dirs):
            print("  -", r)

    # autodetect qc function if not provided
    if qc_func is None:
        qc_func = (
            globals().get("return_qc")
            or globals().get("summarize_h5ad_qc_from_adata")
            or globals().get("summarize_h5ad_qc_safe")
        )
        if qc_func is None:
            raise ValueError("qc_func not provided and no default QC function found.")

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

    for run_dir in sorted(run_dirs):
        run_name = run_dir.name
        processed_runs.append(run_name)

        for slide_idx, slide_name in enumerate(["slide1", "slide2"], start=1):
            slide_path = run_dir / slide_name
            if not slide_path.exists():
                continue

            if verbose:
                print(f"\n[QC] {run_name} - {slide_name}")

            try:
                if save_csv_name:
                    df_qc = qc_func(str(slide_path), save_csv=save_csv_name)
                else:
                    df_qc = qc_func(str(slide_path))
            except TypeError:
                df_qc = qc_func(str(slide_path))
            except Exception as e:
                print(f"[ERROR] QC failed on {slide_path}: {e}")
                continue

            if df_qc is None or df_qc.empty:
                if verbose:
                    print("  No QC rows found.")
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
                    new_row = {c: pd.NA for c in meta.columns}
                    new_row["sample_id"] = sid
                    for col in expected_cols:
                        new_row[col] = row[col]
                    meta = pd.concat([meta, pd.DataFrame([new_row])], ignore_index=True)

    meta = meta.sort_values("sample_id").reset_index(drop=True)

    print("\n✅ QC integration complete.")
    return {"metadata": meta, "processed_runs": processed_runs}
