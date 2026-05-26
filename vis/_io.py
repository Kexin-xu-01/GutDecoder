"""
_io.py — data loading / IO utilities.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Tuple
import re

import numpy as np
import pandas as pd

from gutdecoder.config import (
    RESULTS_ROOT as DEFAULT_RUNS_ROOT,
    DATA_ROOT as DEFAULT_SPLITS_ROOT,
    CURATED_XLSX as DEFAULT_CURATED_XLSX,
    HEST_METADATA_CSV as DEFAULT_EXTRA_METADATA,
    BROAD_METADATA_CSV as DEFAULT_BROAD_METADATA,
)

from ._helpers import _safe_read_json, _find_col_ci


# --- Count training patches ----#
def _find_col_ci(df: pd.DataFrame, target: str):
    lowmap = {c.lower(): c for c in df.columns}
    return lowmap.get(target.lower())

def _find_any_col_ci(df: pd.DataFrame, targets):
    for t in targets:
        c = _find_col_ci(df, t)
        if c is not None:
            return c
    return None

def _dataset_prefixes(dataset: str):
    """
    XeniumPR10   -> ['XeniumPR10']
    XeniumPR4-5  -> ['XeniumPR4', 'XeniumPR5']
    VisiumR1-6   -> ['VisiumR1', ..., 'VisiumR6']
    """
    ds = str(dataset).strip()

    m = re.fullmatch(r'(XeniumPR|XeniumR|VisiumR)(\d+)(?:-(\d+))?(?:_.*)?', ds)
    if not m:
        return [ds]  # fallback for special names like pilot, broad, etc.

    family = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    if end < start:
        start, end = end, start

    return [f"{family}{i}" for i in range(start, end + 1)]

def _pick_patch_col(meta: pd.DataFrame, dataset: str):
    ds = str(dataset).lower()

    if "cell_centered" in ds:
        return _find_any_col_ci(meta, ["num_patches_cell_centered"])
    if "cell" in ds:
        return _find_any_col_ci(meta, ["num_patches_cell_100um"])
    if "_25um" in ds:
        return _find_any_col_ci(meta, ["num_patches_25um"])
    if "_50um" in ds:
        return _find_any_col_ci(meta, ["num_patches_50um"])
    if "unfiltered" in ds:
        return _find_any_col_ci(meta, ["num_patches_100um_unfiltered"])
    return _find_any_col_ci(meta, ["num_patches_100um", "num_patches_100um_unfiltered"])

def count_dataset_patches(meta: pd.DataFrame, dataset: str) -> float:
    sample_col = _find_col_ci(meta, "sample_id")
    if sample_col is None or meta.empty:
        return np.nan

    patch_col = _pick_patch_col(meta, dataset)
    if patch_col is None:
        return np.nan

    prefixes = _dataset_prefixes(dataset)
    sample_ids = meta[sample_col].astype(str)

    mask = pd.Series(False, index=meta.index)
    for p in prefixes:
        # important: this matches XeniumPR10 correctly, but not XeniumPR1
        mask |= sample_ids.str.match(rf"^{re.escape(p)}(?:$|[^0-9])", na=False)

    return pd.to_numeric(meta.loc[mask, patch_col], errors="coerce").sum(min_count=1)

def add_num_training_patches_mean(
    df_summary: pd.DataFrame,
    xenium_csv: str = "/project/gutdecoder/kxu/hest/metadata/hest_directory.csv",
    broad_csv: str = "/project/gutdecoder/kxu/hest/metadata/broad_directory.csv",
) -> pd.DataFrame:
    """
    Add a column 'num_training_patches_mean' to df_summary based on dataset rules,
    using CASE-INSENSITIVE column lookup for all 'num_patches*' columns and 'SampleID'.
    """
    out = df_summary.copy()

    # Load metadata (once)
    try:
        xen = pd.read_csv(xenium_csv)
    except Exception:
        xen = pd.DataFrame()

    try:
        brd = pd.read_csv(broad_csv)
    except Exception:
        brd = pd.DataFrame()

    # ---- case-insensitive column helpers ----
    def _find_col_ci(df: pd.DataFrame, target: str):
        if df.empty:
            return None
        lowmap = {c.lower(): c for c in df.columns}
        return lowmap.get(target.lower())

    def _find_any_col_ci(df: pd.DataFrame, targets: list[str]):
        for t in targets:
            col = _find_col_ci(df, t)
            if col is not None:
                return col
        return None

    def _safe_mean_ci(df: pd.DataFrame, col_candidates: list[str]) -> float:
        col = _find_any_col_ci(df, col_candidates)
        if col is None:
            return np.nan
        return pd.to_numeric(df[col], errors="coerce").dropna().mean()

    # --- resolve commonly used columns case-insensitively ---
    col_sample_id_xen = _find_col_ci(xen, "sample_id")

    # xenium means
    xen_mean_num_patches = _safe_mean_ci(xen, ["num_patches_100um_unfiltered"])
    xen_mean_num_patches_segger = _safe_mean_ci(xen, ["num_patches_100um"])
    xen_mean_num_patches_50um = _safe_mean_ci(xen, ["num_patches_50um"])
    xen_mean_num_patches_25um = _safe_mean_ci(xen, ["num_patches_25um"])
    num_patches_cell_100um = _safe_mean_ci(xen, ["num_patches_cell_100um"])

    # pilot subset (two sample IDs)
    pilot_ids = {"XeniumPR1S1ROI2", "XeniumPR1S1ROI3"}
    if col_sample_id_xen is not None:
        xen_pilot = xen[xen[col_sample_id_xen].astype(str).isin(pilot_ids)]
        xen_pilot_mean = _safe_mean_ci(xen_pilot, ["num_patches_100um_unfiltered"])
    else:
        xen_pilot_mean = np.nan

    # broad means
    brd_mean_num_patches = _safe_mean_ci(brd, ["num_patches_100um"])
    brd_mean_cell_centered = _safe_mean_ci(brd, ["num_patches_cell_centered"])

    # per-row mapping
    out["num_training_patches_mean"] = out["dataset"].apply(lambda ds: count_dataset_patches(xen, ds))
 
    return out


def summarize_runs(root_dir):
    """
    List runs in ST_pred_results and summarize config.json details along with
    highest Pearson mean/std from dataset_results.json and gene count.

    Args:
        root_dir (str): Root directory containing the run folders.

    Returns:
        pd.DataFrame: Summary dataframe for all runs.
    """
    summary = []

    for run in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run)
        if not os.path.isdir(run_path) or not run.startswith("run_"):
            continue

        config_data = {}
        gene_list = ""
        config_found = False
        best_model = None
        num_genes = None
        dataset_name = None

        # --- find config.json and parse basics ---
        config_dir = None
        for dirpath, _, filenames in os.walk(run_path):
            if "config.json" in filenames:
                config_path = os.path.join(dirpath, "config.json")
                config_dir = dirpath
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                gene_list = config_data.get("gene_list", "") or ""
                # datasets can be a list; take the first if present
                ds = config_data.get("datasets")
                if isinstance(ds, list) and ds:
                    dataset_name = ds[0]
                elif isinstance(ds, str):
                    dataset_name = ds
                config_found = True
                break

                   # --- try to locate and count genes in <gene_list>.json ---
        def _try_count_genes(path):
            try:
                with open(path, "r") as gf:
                    payload = json.load(gf)
                if isinstance(payload, dict) and isinstance(payload.get("genes"), list):
                    return len(payload["genes"])
                if isinstance(payload, list):
                    return len(payload)
            except Exception:
                pass
            return None

        if gene_list:
            candidates = []

            # 3) under eval/data/<dataset_name>/<gene_list>
            if dataset_name:
                data_root = DEFAULT_SPLITS_ROOT
                candidates.append(os.path.join(data_root, str(dataset_name), gene_list))

            for cand in candidates:
                if cand and os.path.isfile(cand):
                    num_genes = _try_count_genes(cand)
                    if num_genes is not None:
                        break  # stop at the first working location


        # Search for dataset_results.json
        dataset_results_path = os.path.join(run_path, "dataset_results.json")
        highest_mean = None
        highest_std = None
        if os.path.isfile(dataset_results_path):
            with open(dataset_results_path, 'r') as f:
                data = json.load(f)
                all_results = data.get("results", [])[0].get("results", [])
                if all_results:
                    best_entry = max(all_results, key=lambda x: x["pearson_mean"])
                    highest_mean = best_entry["pearson_mean"]
                    highest_std = best_entry["pearson_std"]
                    best_model = best_entry["encoder_name"]

        # -------------------
        # Discover encoders from enc_results.json files under run_path/<dataset>/
        # -------------------
        encoders_set = set()
        for dirpath, dirnames, filenames in os.walk(run_path):
            # only consider enc_results.json files (case-sensitive as provided)
            if "enc_results.json" in filenames:
                enc_results_path = os.path.join(dirpath, "enc_results.json")
                try:
                    with open(enc_results_path, "r") as ef:
                        payload = json.load(ef)
                    # payload expected to have top-level "results": [ { "encoder_name": ... }, ... ]
                    if isinstance(payload, dict):
                        results = payload.get("results", [])
                        if isinstance(results, list):
                            for entry in results:
                                if isinstance(entry, dict):
                                    enc_name = entry.get("encoder_name") or entry.get("name")
                                    if enc_name:
                                        encoders_set.add(str(enc_name))
                                # if the entry is a plain string, include it directly
                                elif isinstance(entry, str):
                                    encoders_set.add(entry)
                except Exception:
                    # ignore malformed enc_results.json and continue
                    pass

        encoders_list = sorted(encoders_set) if encoders_set else None
        encoders_str = ", ".join(encoders_list) if encoders_list else None

        from pathlib import Path

        slide_emb_root = config_data.get("slide_emb_root")
        encoder_root_name = Path(slide_emb_root).name if slide_emb_root else None
        encoder_short_name = encoder_root_name.replace("slide_features_", "") if encoder_root_name else None



        summary.append({
            "run": run,
            "gene_list": gene_list,
            "num_genes": num_genes,
            "alpha": config_data.get("alpha"),
            "batch_size": config_data.get("batch_size"),
            "dimreduce": config_data.get("dimreduce"),
            "encoders": encoders_str,
            "slide_emb_root": encoder_short_name,
            "fusion":config_data.get("fusion"),
            "normalize": config_data.get("normalize"),
            "library_size_normalize": config_data.get("library_size_normalize", False), # ensure library_size_normalize defaults to False if missing
            "latent_dim": config_data.get("latent_dim"),
            "method": config_data.get("method"),
            "dataset": config_data.get("datasets", [None])[0] if config_found else None,
            "best_model": best_model,
            "highest_pearson_mean": highest_mean,
            "highest_pearson_std": highest_std,
        })

    df = pd.DataFrame(summary)

    # >>> Add num_training_patches_mean here <<<
    df = add_num_training_patches_mean(
        df,
        xenium_csv=DEFAULT_EXTRA_METADATA,
        broad_csv=DEFAULT_BROAD_METADATA,
    )

    # --- Sort by dataset and highest mean ---
    if not df.empty:
        df = df.sort_values(by=["dataset", 'gene_list', "highest_pearson_mean"], ascending=[True, True, False]).reset_index(drop=True)

    return df


def load_run(run: str, runs_root: str = DEFAULT_RUNS_ROOT) -> Tuple[str, pd.DataFrame, dict]:
    """Load dataset_results.json from a run directory."""
    run_dir = Path(runs_root) / run
    ds = _safe_read_json(run_dir / "dataset_results.json")
    ds0 = (ds.get("results") or [None])[0] or {}
    dataset_name = ds0.get("dataset_name", "Unknown")
    entries = ds0.get("results") or []

    results_df = pd.DataFrame(entries)
    if results_df.empty:
        raise ValueError(f"No 'results' entries in {run_dir}/dataset_results.json")

    for col in ["encoder_name", "pearson_mean", "pearson_std"]:
        if col not in results_df:
            results_df[col] = np.nan

    idx = results_df["pearson_mean"].astype(float).idxmax()
    best_model_info: dict = results_df.loc[idx].to_dict()

    best_model_info.setdefault("gene_corrs", {})
    best_model_info.setdefault("per_sample_corrs", [])

    return dataset_name, results_df, best_model_info


def extract_best_model_gene_corrs(run: str,
                                  runs_root: str = DEFAULT_RUNS_ROOT,
                                  verbose: bool = True):
    """
    From a run folder:
      - choose best model by pearson_mean
      - read <run>/<dataset>/<encoder>/results_kfold.json
      - build df_genes with ['gene','mean_corr','std_corr','corr_per_split']
    """
    run_path = Path(runs_root) / run
    dataset_name, _, best_model_info = load_run(run_path)
    encoder_name = best_model_info.get("encoder_name", "Unknown")

    if verbose:
        print(f"[extract] dataset={dataset_name} encoder={encoder_name} "
              f"pearson_mean={best_model_info.get('pearson_mean')} "
              f"std={best_model_info.get('pearson_std')}")

    # results_kfold.json path
    kfold_path = run_path / dataset_name / encoder_name / "results_kfold.json"
    if not kfold_path.is_file():
        raise FileNotFoundError(f"results_kfold.json not found at {kfold_path}")

    kfold = _safe_read_json(kfold_path)
    pearson_corrs = kfold.get("pearson_corrs", [])

    records = []
    for g in pearson_corrs:
        if not g or "name" not in g:
            continue
        records.append({
            "gene": g.get("name"),
            "mean_corr": g.get("mean"),
            "std_corr": g.get("std"),
            "corr_per_split": g.get("pearson_corrs", [])
        })

    df_genes = pd.DataFrame(records).convert_dtypes().fillna(pd.NA)
    return best_model_info, dataset_name, df_genes


def annotate_genes_with_curated(df_genes: pd.DataFrame, path_meta = "/project/gutdecoder/kxu/hest/metadata/curated_gene_list.xlsx", case_insensitive: bool = True) -> pd.DataFrame:
    """
    Minimal annotation:
      - panel: '480' if gene in '480 panel full list', else <NA>
      - cell_type from 'Cell Type Specific' + 'Where?'
      - condition from 'Conditional' + 'Where?.1'
    Merges onto df_genes by 'gene' (case-insensitive by default).

    Args:
        df_genes: DataFrame with at least ['gene'].
        df_meta: curated Excel sheet as a DataFrame.
        case_insensitive: if True, match by uppercased gene symbols.

    Returns:
        df_genes with columns ['panel','cell_type','condition'] added when available.
    """

    if "gene" not in df_genes.columns:
        raise ValueError("df_genes must contain a 'gene' column")

    df_meta = pd.read_excel(path_meta, sheet_name=0)

    m = df_meta.copy()

    # 1) rename the two columns exactly as you showed
    m = m.rename(columns={
        "Where?": "cell_type",
        "Where?.1": "condition"
    })

    # 2) build the three mapping frames (one row per gene)
    # panel (480 list)
    if "480 panel full list" in m.columns:
        df_panel = (m[["480 panel full list"]].dropna().rename(columns={"480 panel full list": "gene"}))
        df_panel["panel"] = "480"
        df_panel = df_panel.drop_duplicates(subset=["gene"])
    else:
        df_panel = pd.DataFrame(columns=["gene", "panel"])

    # cell_type
    if "Cell Type Specific" in m.columns and "cell_type" in m.columns:
        df_celltype = (
            m[["Cell Type Specific", "cell_type"]]
            .dropna(subset=["Cell Type Specific"])
            .rename(columns={"Cell Type Specific": "gene"})
            .drop_duplicates(subset=["gene"])
        )
    else:
        df_celltype = pd.DataFrame(columns=["gene", "cell_type"])

    # condition
    if "Conditional" in m.columns and "condition" in m.columns:
        df_condition = (
            m[["Conditional", "condition"]]
            .dropna(subset=["Conditional"])
            .rename(columns={"Conditional": "gene"})
            .drop_duplicates(subset=["gene"])
        )
    else:
        df_condition = pd.DataFrame(columns=["gene", "condition"])

    # coeliac genes
    if "Coeliac" in m.columns:
        df_coeliac = (
            m[["Coeliac"]]
            .dropna()
            .rename(columns={"Coeliac": "gene"})
        )
        df_coeliac["coeliac"] = "coeliac"
        df_coeliac = df_coeliac.drop_duplicates(subset=["gene"])
    else:
        df_coeliac = pd.DataFrame(columns=["gene", "coeliac"])

    # tcr genes
    if "TCR" in m.columns:
        df_tcr = (
            m[["TCR"]]
            .dropna()
            .rename(columns={"TCR": "gene"})
        )
        df_tcr["tcr"] = "tcr"
        df_tcr = df_tcr.drop_duplicates(subset=["gene"])
    else:
        df_tcr = pd.DataFrame(columns=["gene", "tcr"])

    # 3) (optional) case-insensitive merge keys
    def _prep_key(df, col="gene"):
        out = df.copy()
        out[col] = out[col].astype(str).str.strip()
        if case_insensitive:
            out["_gk"] = out[col].str.upper()
        else:
            out["_gk"] = out[col]
        return out

    g   = _prep_key(df_genes, "gene")
    p   = _prep_key(df_panel, "gene")
    ct  = _prep_key(df_celltype, "gene")
    cond= _prep_key(df_condition, "gene")
    coeliac= _prep_key(df_coeliac, "gene")
    tcr= _prep_key(df_tcr, "gene")

    # 4) left-merge the three annotations
    out = g.merge(p[["_gk", "panel"]], on="_gk", how="left")
    out = out.merge(ct[["_gk", "cell_type"]], on="_gk", how="left")
    out = out.merge(cond[["_gk", "condition"]], on="_gk", how="left")
    out = out.merge(coeliac[["_gk", "coeliac"]], on="_gk", how="left")
    out = out.merge(tcr[["_gk", "tcr"]], on="_gk", how="left")

    # 5) clean up
    out = out.drop(columns=["_gk"]).convert_dtypes().fillna(pd.NA)
    return out


def get_test_splits(run: str,
                    runs_root: str = DEFAULT_RUNS_ROOT,
                    splits_root: str = DEFAULT_SPLITS_ROOT,
                    extra_metadata_csv: str = DEFAULT_EXTRA_METADATA) -> pd.DataFrame:
    """Load test splits and optionally merge HEST directory metadata."""
    run_dir = Path(runs_root) / run
    dataset_name, _, _ = load_run(run_dir)

    dataset_split_dir = Path(splits_root) / dataset_name / "splits"
    test_files = sorted(glob.glob(str(dataset_split_dir / "test_*.csv")))
    if not test_files:
        raise FileNotFoundError(f"No test_*.csv files found in {dataset_split_dir}")

    rows = []
    for tf in test_files:
        split_num = int(os.path.basename(tf).replace("test_", "").replace(".csv", ""))
        df_split = pd.read_csv(tf)
        for sample in df_split["sample_id"].tolist():
            rows.append({"split": split_num, "test_sample": str(sample)})

    df_test = pd.DataFrame(rows).convert_dtypes().fillna(pd.NA)

    ds_l = str(dataset_name).strip().lower()
    #if ds_l in {"xeniumpr1", "pilot",} and Path(extra_metadata_csv).exists():
    if "broad" not in ds_l and Path(extra_metadata_csv).exists():
        meta = pd.read_csv(extra_metadata_csv)
        df_test = df_test.merge(meta, left_on="test_sample", right_on="sample_id", how="left")
        df_test = df_test.drop(columns=["sample_id"], errors="ignore").convert_dtypes().fillna(pd.NA)

    return df_test


def merge_kfold_gene_corrs_with_test_metadata(df_genes: pd.DataFrame,
                                              df_test_splits: pd.DataFrame) -> pd.DataFrame:
    """Explode corr_per_split and merge with test splits."""
    if "gene" not in df_genes or "corr_per_split" not in df_genes:
        return pd.DataFrame()

    df_long = df_genes.explode("corr_per_split").rename(columns={"corr_per_split": "corr"})
    df_long["split"] = df_long.groupby("gene").cumcount()
    return df_long.merge(df_test_splits, on="split", how="left").convert_dtypes().fillna(pd.NA)


def best_results_by_gene_and_dataset(df):
    """
    Return the best result for each unique combination of gene_list and dataset.

    Args:
        df (pd.DataFrame): DataFrame returned by summarize_runs().

    Returns:
        pd.DataFrame: Best results per (gene_list, dataset).
    """
    if df.empty:
        return df

    # For each (gene_list, dataset), pick row with max highest_pearson_mean
    idx = (
        df.groupby(["gene_list", "dataset"])["highest_pearson_mean"]
        .idxmax()
        .dropna()
    )

    return df.loc[idx].reset_index(drop=True)
