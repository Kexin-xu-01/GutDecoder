"""
_grid.py — grid layout plots.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gutdecoder.config import (
    RESULTS_ROOT as DEFAULT_RUNS_ROOT,
    DATA_ROOT as DEFAULT_SPLITS_ROOT,
    SUMMARY_PLOT_DIR as DEFAULT_SUMMARY_PLOT_DIR,
)

from ._helpers import _find_col_ci, _assemble_color_map, _safe_filename
from ._io import (
    get_test_splits,
    extract_best_model_gene_corrs,
    merge_kfold_gene_corrs_with_test_metadata,
    load_run,
)


def plot_all_samples_grid_for_dataset(
    dataset_or_run: str,
    runs_root: str = DEFAULT_RUNS_ROOT,
    splits_root: str = DEFAULT_SPLITS_ROOT,
    outdir: str | Path | None = DEFAULT_SUMMARY_PLOT_DIR,
    candidate_runs: list | None = None,
    auto_discover_runs: bool = True,
    ncols: int = 4,
    max_samples: int | None = None,
    figsize_per_subplot: tuple[float, float] = (4, 2.5),
    dpi: int = 200,
    show: bool = False,
) -> tuple[Dict[str, Path], pd.DataFrame]:
    """
    For a given dataset name (e.g. "XeniumPR8") or a run folder (e.g. "run_..."),
    produce one grid image that contains one small barplot per test sample for that
    dataset/run. Each small barplot shows that sample's mean Pearson ± std across
    all candidate runs that include that sample.

    Args:
        dataset_or_run: dataset name (folder under splits_root) OR a run folder name.
        runs_root: root folder containing run_* directories.
        splits_root: root folder containing dataset splits, e.g. eval/data/<dataset>/splits.
        outdir: where to save the combined grid PNG.
        candidate_runs: optional list of run folder names to consider. If None and
                        auto_discover_runs=True, scans runs_root for run_* folders.
        auto_discover_runs: whether to auto-discover runs when candidate_runs is None.
        ncols: number of columns in grid.
        max_samples: optional cap on number of samples to plot (useful for very large datasets).
        figsize_per_subplot: (width, height) of each subplot in inches.
        dpi: image DPI.
        show: if True, display the final grid figure.

    Returns:
        (saved_paths, df_all)
        - saved_paths: dict {dataset_or_run: Path(saved_png)}
        - df_all: DataFrame with rows per sample-per-run with ['dataset_or_run','sample','run','dataset','mean_pearson','std_pearson','n_genes','gene_list']
    """
    outdir = Path(outdir) if outdir is not None else Path(DEFAULT_SUMMARY_PLOT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    runs_root = Path(runs_root)
    splits_root = Path(splits_root)

    # 1) Determine sample list for the provided dataset_or_run
    samples = []
    # If it's a run folder (starts with run_) prefer reading that run's splits
    if str(dataset_or_run).startswith("run_"):
        try:
            df_splits = get_test_splits(dataset_or_run, runs_root=str(runs_root), splits_root=str(splits_root))
            sample_col = _find_col_ci(df_splits, ["test_sample", "sample_id", "SampleID"])
            if sample_col is None:
                raise KeyError("No sample id column found in splits for run.")
            samples = sorted(df_splits[sample_col].dropna().astype(str).unique().tolist())
        except Exception as e:
            raise RuntimeError(f"Failed to read splits for run {dataset_or_run}: {e}")
        label_name = dataset_or_run
    else:
        # treat as dataset name and look under splits_root/<dataset>/splits/test_*.csv
        ds_dir = splits_root / str(dataset_or_run) / "splits"
        if not ds_dir.exists():
            raise FileNotFoundError(f"Splits folder not found for dataset '{dataset_or_run}' at {ds_dir}")
        test_files = sorted(ds_dir.glob("test_*.csv"))
        if not test_files:
            raise FileNotFoundError(f"No test_*.csv files found for dataset '{dataset_or_run}' in {ds_dir}")
        sample_set = set()
        for tf in test_files:
            try:
                df_t = pd.read_csv(tf)
                # find sample col
                sc = _find_col_ci(df_t, ["test_sample", "sample_id", "SampleID"])
                if sc is None:
                    continue
                sample_set.update(df_t[sc].dropna().astype(str).tolist())
            except Exception:
                continue
        samples = sorted(sample_set)
        label_name = dataset_or_run

    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        raise ValueError("No samples found to plot.")

    # 2) Determine candidate runs to check (if not provided)
    if candidate_runs is None:
        if not auto_discover_runs:
            raise ValueError("candidate_runs is None and auto_discover_runs is False.")
        candidate_runs = [p.name for p in sorted(runs_root.glob("run_*")) if p.is_dir()]

    # allow user to pass absolute run names if they included a path: normalize to names
    candidate_runs = [str(r).split("/")[-1] for r in candidate_runs]

    # Precompute datasets and gene_list for each candidate run (to reuse)
    run_meta = {}
    for run_name in candidate_runs:
        run_path = runs_root / run_name
        # read dataset_results.json safely to infer dataset_name quickly (best-effort)
        try:
            ds_name, _, _ = load_run(str(run_path))  # load_run expects run path or Path; it handles conversion
        except Exception:
            ds_name = "Unknown"
        # try to find gene_list from config.json (best-effort)
        gene_list_name = ""
        try:
            for dirpath, _, filenames in os.walk(run_path):
                if "config.json" in filenames:
                    with open(Path(dirpath) / "config.json", "r") as cf:
                        cfg = json.load(cf)
                    gene_list_name = cfg.get("gene_list") or cfg.get("genes") or ""
                    if isinstance(gene_list_name, str):
                        gene_list_name = Path(gene_list_name).name
                    break
        except Exception:
            gene_list_name = ""
        run_meta[run_name] = {"dataset": ds_name, "gene_list": gene_list_name}

    # 3) For each sample, collect per-run stats (mean/std/n_genes)
    rows = []
    sample_col_candidates = ["test_sample", "sample_id", "SampleID"]
    for sample in samples:
        for run_name in candidate_runs:
            try:
                # get splits for this run to check membership and to enable merge later
                try:
                    df_splits = get_test_splits(run_name, runs_root=str(runs_root), splits_root=str(splits_root))
                except Exception:
                    # no splits for this run -> skip
                    continue

                sample_col = _find_col_ci(df_splits, sample_col_candidates)
                if sample_col is None:
                    continue
                if not df_splits[sample_col].astype(str).isin([sample]).any():
                    # sample not present in this run's test splits
                    continue

                # extract gene-level correlations for best model of this run
                try:
                    best_info, ds_name, df_genes = extract_best_model_gene_corrs(run_name, runs_root=str(runs_root), verbose=False)
                except FileNotFoundError:
                    # missing kfold file etc -> skip
                    continue

                # merge to get per-sample per-split entries
                try:
                    df_long = merge_kfold_gene_corrs_with_test_metadata(df_genes, df_splits)
                except Exception:
                    continue

                sample_col_long = _find_col_ci(df_long, sample_col_candidates)
                if sample_col_long is None:
                    continue

                df_sample = df_long[df_long[sample_col_long].astype(str) == sample]
                if df_sample.empty:
                    continue

                mean_pearson = pd.to_numeric(df_sample["corr"], errors="coerce").dropna().mean()
                std_pearson = pd.to_numeric(df_sample["corr"], errors="coerce").dropna().std()
                n_genes = int(df_sample.shape[0])

                rows.append({
                    "dataset_or_run": label_name,
                    "sample": sample,
                    "run": run_name,
                    "dataset": run_meta.get(run_name, {}).get("dataset", ds_name if 'ds_name' in locals() else "Unknown"),
                    "gene_list": run_meta.get(run_name, {}).get("gene_list", ""),
                    "mean_pearson": float(mean_pearson) if not math.isnan(mean_pearson) else pd.NA,
                    "std_pearson": float(std_pearson) if not math.isnan(std_pearson) else pd.NA,
                    "n_genes": n_genes,
                })
            except Exception as e:
                # skip this run/sample pair on error but continue
                warnings.warn(f"[plot_all_samples_grid_for_dataset] skipping {sample} in {run_name}: {e}")
                continue

    if not rows:
        raise ValueError("No sample/run pairs found to plot. Check candidate runs and splits.")

    df_all = pd.DataFrame(rows).convert_dtypes()

    samples_order = sorted(df_all["sample"].unique().tolist())
    plots_per_page = 6
    ncols = 3
    nrows = 2

    datasets_all = sorted(df_all["dataset"].fillna("Unknown").unique().tolist())
    color_map = _assemble_color_map(datasets_all)

    pdf_path = outdir / f"{_safe_filename(str(label_name))}_samples_grid.pdf"

    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(samples_order), plots_per_page):
            page_samples = samples_order[page_start:page_start + plots_per_page]

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                    figsize=(ncols * 4, nrows * 3))
            axes = axes.flatten()

            for ax in axes:
                ax.axis("off")

            for i, sample in enumerate(page_samples):
                ax = axes[i]
                ax.axis("on")

                df_s = df_all[df_all["sample"] == sample].sort_values(
                    "mean_pearson", ascending=False
                )

                if df_s.empty:
                    ax.set_title(sample)
                    continue

                x = range(len(df_s))
                means = df_s["mean_pearson"].astype(float).to_numpy()
                errs = df_s["std_pearson"].astype(float).fillna(0.0).to_numpy()

                labels = [
                    f"{str(gl)} | {str(ds)}"
                    for gl, ds in zip(df_s["gene_list"], df_s["dataset"])
                ]

                colors = [color_map.get(ds, (0.5, 0.5, 0.5))
                        for ds in df_s["dataset"]]

                bars = ax.bar(x, means, yerr=errs,
                            color=colors, capsize=3,
                            edgecolor="black", linewidth=0.5)

                # annotate n_genes
                for bar, ng in zip(bars, df_s["n_genes"]):
                    ax.annotate(str(ng),
                                xy=(bar.get_x() + bar.get_width()/2,
                                    bar.get_height()),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center", va="bottom",
                                fontsize=7)

                ax.set_xticks(x)
                ax.set_xticklabels(labels,
                                rotation=45,
                                ha="right",
                                fontsize=7)

                ax.set_title(sample, fontsize=9)
                ax.set_ylabel("Mean Pearson", fontsize=8)
                ax.tick_params(axis="y", labelsize=7)

            fig.tight_layout(rect=[0, 0, 0.9, 1])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved multi-page grid to: {pdf_path}")

    return pdf_path, df_combined
