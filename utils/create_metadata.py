from pathlib import Path
from typing import List, Union, Optional, Sequence, Dict, Any, List, Callable
import pandas as pd
import re
import scanpy as sc
from IPython.display import display


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


def return_qc(root, save_csv=None):
    """
    Load .h5ad files under each sample folder and compute mean QC metrics.

    Expected structure:
        root/
            sample1/
                *.h5ad
            sample2/
                *.h5ad

    Args:
        root (str or Path): root directory containing sample folders
        save_csv (str or Path, optional): if given, save summary table to this CSV

    Returns:
        pandas.DataFrame
    """
    root = Path(root)
    results = []

    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue

        for h5ad_file in sample_dir.glob("aligned_adata.h5ad"):
            sample_id = sample_dir.name

            try:
                adata = sc.read_h5ad(h5ad_file)

                obs = adata.obs

                summary = {
                    "sample_id": sample_id,
                    "n_obs": adata.n_obs,
                    "mean_total_counts": obs["total_counts"].mean() if "total_counts" in obs else None,
                    "mean_n_genes_by_counts": obs["n_genes_by_counts"].mean() if "n_genes_by_counts" in obs else None,
                    "mean_log1p_total_counts": obs["log1p_total_counts"].mean() if "log1p_total_counts" in obs else None,
                    "mean_log1p_n_genes_by_counts": obs["log1p_n_genes_by_counts"].mean() if "log1p_n_genes_by_counts" in obs else None,
                }

                results.append(summary)

            except Exception as e:
                print(f"[ERROR] Failed reading {h5ad_file}: {e}")

    df = pd.DataFrame(results).sort_values("sample_id")

    print(df)

    if save_csv:
        df.to_csv(root / save_csv, index=False)
        print(f"[INFO] Saved summary to {root / save_csv}")

    return df


# from pathlib import Path
# import pandas as pd
# from typing import Callable, List, Tuple, Optional, Dict

# import re


# def add_qc_from_selected_runs(
#     metadata: pd.DataFrame,
#     root: str | Path,
#     qc_func: Optional[Callable[[str, Optional[str]], pd.DataFrame]] = None,
#     save_csv_name: Optional[str] = "qc_summary.csv",
#     update_existing: bool = True,
#     verbose: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Scan root for run folders exactly matching:
#       - XeniumPR<digits>  (e.g. XeniumPR1)
#       - XeniumR<digits>   (e.g. XeniumR1)
#       - VisiumR<digits>   (e.g. VisiumR1)

#     For each run found, processes slide1 and slide2 (if present), runs qc_func on the slide path,
#     prefixes sample_id by '{run_name}S{slide_idx}', and merges QC columns into metadata.

#     Args:
#         metadata: pandas.DataFrame containing at least 'sample_id' (will be preserved).
#         root: root directory containing run folders.
#         qc_func: function(path: str, save_csv: Optional[str]) -> pandas.DataFrame.
#                  If None, will try to use `return_qc`, `summarize_h5ad_qc_from_adata`, or
#                  `summarize_h5ad_qc_safe` from globals().
#         save_csv_name: optional filename (relative to each slide) to pass to qc_func so it may save CSV.
#         update_existing: if True, overwrite existing metadata values for matched sample_id rows;
#                          if False, only fill missing values (NaNs).
#         verbose: whether to print progress.

#     Returns:
#         {
#             "metadata": updated_metadata_df,
#             "processed_runs": list_of_run_names_processed,
#             "per_slide": list_of (slide_path_str, df_qc) tuples for debugging
#         }
#     """
#     root = Path(root)

#     # compile regexes for exact matches
#     patterns = [
#         re.compile(r"^XeniumPR\d+$"),
#         re.compile(r"^XeniumR\d+$"),
#         re.compile(r"^VisiumR\d+$"),
#     ]

#     # find run directories that exactly match any pattern
#     run_dirs: List[Path] = []
#     for d in sorted(root.iterdir()):
#         if not d.is_dir():
#             continue
#         if any(p.match(d.name) for p in patterns):
#             run_dirs.append(d)

#     if verbose:
#         print(f"Found {len(run_dirs)} matching runs under {root}:")
#         for r in run_dirs:
#             print("  -", r.name)

#     # autodetect qc_func if not provided
#     if qc_func is None:
#         qc_func = globals().get("return_qc") or globals().get("summarize_h5ad_qc_from_adata") or globals().get("summarize_h5ad_qc_safe")
#         if qc_func is None:
#             raise ValueError("qc_func not provided and no default (return_qc/summarize_h5ad_qc_from_adata) found in globals.")

#     expected_cols = [
#         "n_obs",
#         "mean_total_counts",
#         "mean_n_genes_by_counts",
#         "mean_log1p_total_counts",
#         "mean_log1p_n_genes_by_counts",
#     ]

#     # make a working copy of metadata and ensure QC columns exist
#     meta = metadata.copy(deep=True)
#     for c in expected_cols:
#         if c not in meta.columns:
#             meta[c] = pd.NA

#     processed_runs: List[str] = []
#     per_slide: List[tuple] = []

#     for run_dir in run_dirs:
#         run_name = run_dir.name
#         processed_runs.append(run_name)

#         for slide_idx, slide_name in enumerate(["slide1", "slide2"], start=1):
#             slide_path = run_dir / slide_name
#             if not slide_path.exists():
#                 if verbose:
#                     print(f"[SKIP] {run_name}/{slide_name} not found.")
#                 continue

#             if verbose:
#                 print(f"\n[QC] Processing {run_name} / {slide_name} ...")

#             # call qc_func; try (path, save_csv) signature first
#             try:
#                 if save_csv_name is not None:
#                     df_qc = qc_func(str(slide_path), save_csv=save_csv_name)
#                 else:
#                     df_qc = qc_func(str(slide_path))
#             except TypeError:
#                 df_qc = qc_func(str(slide_path))
#             except Exception as e:
#                 if verbose:
#                     print(f"[ERROR] qc_func failed on {slide_path}: {e}")
#                 continue

#             if df_qc is None or df_qc.empty:
#                 if verbose:
#                     print("  No QC rows returned.")
#                 continue

#             # normalize df_qc: ensure sample_id
#             if "sample_id" not in df_qc.columns:
#                 df_qc = df_qc.reset_index().rename(columns={"index": "sample_id"})

#             # prefix sample_id like RunNameS{slide_idx}
#             prefix = f"{run_name}S{slide_idx}"
#             df_qc["sample_id"] = prefix + df_qc["sample_id"].astype(str)

#             # ensure expected columns exist in df_qc
#             for c in expected_cols:
#                 if c not in df_qc.columns:
#                     df_qc[c] = pd.NA

#             # trim to expected columns + sample_id
#             df_qc_trim = df_qc[["sample_id"] + expected_cols].copy()

#             per_slide.append((str(slide_path), df_qc_trim))

#             if verbose:
#                 # pretty display if in notebook, otherwise print head
#                 try:
#                     from IPython.display import display
#                     display(df_qc_trim)
#                 except Exception:
#                     print(df_qc_trim.head())

#             # merge into meta
#             for _, row in df_qc_trim.iterrows():
#                 sid = row["sample_id"]
#                 if sid in meta["sample_id"].values:
#                     if update_existing:
#                         for c in expected_cols:
#                             meta.loc[meta["sample_id"] == sid, c] = row[c]
#                     else:
#                         for c in expected_cols:
#                             mask = meta["sample_id"] == sid
#                             meta.loc[mask, c] = meta.loc[mask, c].fillna(row[c])
#                 else:
#                     # append new row preserving meta columns
#                     new_row = {col: pd.NA for col in meta.columns}
#                     new_row["sample_id"] = sid
#                     for c in expected_cols:
#                         new_row[c] = row[c]
#                     meta = pd.concat([meta, pd.DataFrame([new_row])], ignore_index=True)

#     # final tidy
#     if "sample_id" in meta.columns:
#         meta = meta.sort_values("sample_id").reset_index(drop=True)

#     if verbose:
#         print("\n✅ QC integration complete. Processed runs:", processed_runs)

#     return {"metadata": meta, "processed_runs": processed_runs, "per_slide": per_slide}





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
        "mean_total_counts",
        "mean_n_genes_by_counts",
        "mean_log1p_total_counts",
        "mean_log1p_n_genes_by_counts",
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
