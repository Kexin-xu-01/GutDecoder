from pathlib import Path
from typing import List, Union, Optional, Sequence, Dict, Any
import pandas as pd
import re


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