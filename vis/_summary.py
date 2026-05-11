"""
_summary.py — summary / aggregate plots.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import display
from adjustText import adjust_text
from matplotlib.backends.backend_pdf import PdfPages

from gutdecoder.config import (
    RESULTS_ROOT as DEFAULT_RUNS_ROOT,
    DATA_ROOT as DEFAULT_SPLITS_ROOT,
    SUMMARY_PLOT_DIR as DEFAULT_SUMMARY_PLOT_DIR,
    HEST_METADATA_CSV as DEFAULT_EXTRA_METADATA,
    CURATED_XLSX as DEFAULT_CURATED_XLSX,
    BROAD_METADATA_CSV as DEFAULT_BROAD_METADATA,
)

from ._helpers import (
    _safe_filename,
    _default_outdir,
    _assemble_color_map,
    _sample_cmap,
    _format_title,
    _find_col_ci,
)
from ._io import (
    load_run,
    extract_best_model_gene_corrs,
    get_test_splits,
    merge_kfold_gene_corrs_with_test_metadata,
    annotate_genes_with_curated,
)
from ._per_gene import (
    plot_gene_correlation_barplot,
    plot_gene_correlation_barplot_grouped,
    plot_gene_correlation_histogram,
)
from ._per_sample import (
    plot_corrs_by_sample,
    plot_pearson_vs_sample_metadata,
)


def plot_summary_bar(best_df: pd.DataFrame,
                     outdir: str | None = DEFAULT_SUMMARY_PLOT_DIR,
                     filename: str = "summary_barplot.png",
                     show: bool = False):
    """
    Barplot of highest_pearson_mean ± highest_pearson_std per run.
    - Grouped and colored by dataset
    - Within each dataset, runs sorted by mean desc
    - Each bar labeled with num_genes, offset to avoid overlapping error bar
    """
    if best_df.empty:
        print("[plot_summary_bar] Empty DataFrame.")
        return None

    df = best_df.copy()
    df["highest_pearson_mean"] = pd.to_numeric(df["highest_pearson_mean"], errors="coerce")
    df["highest_pearson_std"] = pd.to_numeric(df["highest_pearson_std"], errors="coerce").fillna(0.0)
    df["num_genes"] = pd.to_numeric(df["num_genes"], errors="coerce").fillna(0).astype(int)
    df["dataset"] = df["dataset"].astype(str).fillna("Unknown")

    datasets = sorted(df["dataset"].unique().tolist())
    n_datasets = len(datasets)

    # Combine qualitative matplotlib palettes (muted, publication-friendly)
    cmap_tab20  = list(plt.get_cmap("tab20").colors)
    cmap_tab20b = list(plt.get_cmap("tab20b").colors)
    cmap_tab20c = list(plt.get_cmap("tab20c").colors)

    combined_colors = cmap_tab20 + cmap_tab20b + cmap_tab20c

    if n_datasets > len(combined_colors):
        print(f"[plot_summary_bar] Warning: {n_datasets} datasets but only "
              f"{len(combined_colors)} distinct tab colors available. "
              f"Colors will repeat.")

    colors_list = combined_colors[:n_datasets]
    color_map = {ds: colors_list[i] for i, ds in enumerate(datasets)}


    # Order: dataset then mean desc
    ordered = []
    for ds in datasets:
        sub = df[df["dataset"] == ds].sort_values("highest_pearson_mean", ascending=False)
        ordered.append(sub)
    df = pd.concat(ordered)

    labels = (df["dataset"] + " | " + df["gene_list"]).tolist()
    x = np.arange(len(df))
    means = df["highest_pearson_mean"].to_numpy()
    errs = df["highest_pearson_std"].to_numpy()
    colors = [color_map[ds] for ds in df["dataset"]]
    nums = df["num_genes"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(12, len(df)*0.5), 8))
    bars = ax.bar(x, means, yerr=errs, color=colors, capsize=4, edgecolor="black")

    for bar, val, err, ng in zip(bars, means, errs, nums):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax.annotate(str(ng),
                    xy=(bar.get_x() + bar.get_width()/2, height + (err if height >= 0 else -err)),
                    xytext=(0, 5 if height >= 0 else -5), textcoords="offset points",
                    ha="center", va=va, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Pearson correlation (mean ± std)")
    ax.set_title("Run summary grouped by dataset (label = num_genes)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    handles = [plt.Rectangle((0,0),1,1, color=color_map[ds], edgecolor="black") for ds in datasets]
    ax.legend(handles, datasets, title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()

    outdir = _default_outdir(outdir)
    path = outdir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")

    if show:
        display(fig)

    return fig


def plot_summary_bar_by_dataset(
    best_df,
    outdir: str | Path | None = DEFAULT_SUMMARY_PLOT_DIR,
    prefix: str = "summary",
    show: bool = False,
    figsize_per_item: float = 0.5,
):
    """
    Same plotting logic as before (no color or layout changes) but:
      - saves PNGs into <outdir>/by_dataset/
      - creates <outdir>/by_dataset_combined.pdf containing all PNGs (if any)
    Returns:
      (results_dict, combined_pdf_path)
    """
    outdir = _default_outdir(outdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    folder = outdir / "by_dataset"
    folder.mkdir(parents=True, exist_ok=True)

    # Basic validation / normalization
    df = best_df.copy()
    df["highest_pearson_mean"] = pd.to_numeric(df["highest_pearson_mean"], errors="coerce")
    df["highest_pearson_std"] = pd.to_numeric(df["highest_pearson_std"], errors="coerce").fillna(0.0)
    df["num_genes"] = pd.to_numeric(df.get("num_genes", 0), errors="coerce").fillna(0).astype(int)
    df["dataset"] = df["dataset"].astype(str).fillna("Unknown")
    df["gene_list"] = df["gene_list"].astype(str).fillna("default")

    results = {}  # store matplotlib.Figure objects keyed by filenames

    datasets = sorted(df["dataset"].unique().tolist())
    dataset_color_map = _assemble_color_map(datasets)

    for ds in datasets:
        sub = df[df["dataset"] == ds].copy()
        if sub.empty:
            continue

        # order by mean desc
        sub = sub.sort_values("highest_pearson_mean", ascending=False).reset_index(drop=True)
        labels = (sub["gene_list"]).tolist()
        x = np.arange(len(sub))
        means = sub["highest_pearson_mean"].to_numpy()
        errs = sub["highest_pearson_std"].to_numpy()
        nums = sub["num_genes"].to_numpy()
        colors = [dataset_color_map[ds]] * len(sub)

        fig_w = max(8, len(sub) * figsize_per_item)
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        # keep color usage exactly as before
        bars = ax.bar(x, means, yerr=errs, color=colors, capsize=4, edgecolor="black")

        # annotate num_genes above bars
        for bar, val, err, ng in zip(bars, means, errs, nums):
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            offset = 6 if height >= 0 else -6
            ax.annotate(str(ng),
                        xy=(bar.get_x() + bar.get_width()/2, height + (err if height >= 0 else -err)),
                        xytext=(0, offset), textcoords="offset points",
                        ha="center", va=va, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_ylabel("Pearson correlation (mean ± std)")
        ax.set_title(f"{ds} — summary (label = num_genes)")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        # legend: single colored box for dataset (keeps your Rectangle usage)
        handle = plt.Rectangle((0, 0), 1, 1, color=dataset_color_map[ds], edgecolor="black")
        ax.legend([handle], [ds], title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout()

        fname = folder / f"{_safe_filename(prefix)}_{_safe_filename(ds)}_summary.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        results[str(fname)] = fig
        if show:
            display(fig)
        plt.close(fig)

    # Combine saved PNGs into a single PDF (if any)
    png_files = sorted(folder.glob("*.png"))
    combined_pdf_path = None
    if png_files:
        images = []
        for p in png_files:
            try:
                im = Image.open(p)
                # convert to RGB if needed (PIL can't save RGBA to PDF)
                if im.mode in ("RGBA", "LA") or (im.mode == "P"):
                    im = im.convert("RGB")
                images.append(im)
            except Exception as e:
                warnings.warn(f"Failed to open image {p}: {e}")

        if images:
            combined_pdf_path = outdir / "by_dataset_combined.pdf"
            try:
                images[0].save(combined_pdf_path, "PDF", resolution=200.0, save_all=True, append_images=images[1:])
            except Exception as e:
                warnings.warn(f"Failed to write combined PDF {combined_pdf_path}: {e}")
                combined_pdf_path = None

    return results, combined_pdf_path


def plot_gene_list_across_datasets(
    best_df,
    outdir: str | Path | None = DEFAULT_SUMMARY_PLOT_DIR,
    prefix: str = "gene_list_across_datasets",
    show: bool = False,
    figsize_per_item: float = 0.6,
    order_by: str = "mean",  # "mean" or "dataset" or "none"
):
    """
    For each gene_list present in `best_df`, create a plot showing that gene_list's runs
    across all datasets (one bar per run). Bars are colored by dataset.

    Saves PNGs into <outdir>/by_gene_list/ and also creates <outdir>/by_gene_list_combined.pdf

    Returns:
        (results, combined_pdf_path)
        - results: dict mapping saved PNG filepath (str) -> matplotlib.figure.Figure
        - combined_pdf_path: Path to combined PDF, or None if no images were produced
    """
    outdir = Path(outdir) if outdir is not None else Path(".")
    folder = outdir / "by_gene_list"
    folder.mkdir(parents=True, exist_ok=True)

    df = best_df.copy()
    df["highest_pearson_mean"] = pd.to_numeric(df["highest_pearson_mean"], errors="coerce")
    df["highest_pearson_std"] = pd.to_numeric(df.get("highest_pearson_std", 0), errors="coerce").fillna(0.0)
    df["num_genes"] = pd.to_numeric(df.get("num_genes", 0), errors="coerce").fillna(0).astype(int)
    df["dataset"] = df["dataset"].astype(str).fillna("Unknown")
    df["gene_list"] = df["gene_list"].astype(str).fillna("default")
    df["run"] = df.get("run", df.index.astype(str)).astype(str)

    results = {}
    gene_lists = sorted(df["gene_list"].unique().tolist())
    datasets = sorted(df["dataset"].unique().tolist())
    dataset_color_map = _assemble_color_map(datasets)

    for gl in gene_lists:
        sub = df[df["gene_list"] == gl].copy()
        if sub.empty:
            continue

        # Optionally order rows
        if order_by == "mean":
            sub = sub.sort_values("highest_pearson_mean", ascending=False).reset_index(drop=True)
        elif order_by == "dataset":
            sub = sub.sort_values(["dataset", "highest_pearson_mean"], ascending=[True, False]).reset_index(drop=True)
        # else keep original order

        labels = (sub["dataset"]).tolist()
        x = np.arange(len(sub))
        means = sub["highest_pearson_mean"].to_numpy()
        errs = sub["highest_pearson_std"].to_numpy()
        nums = sub["num_genes"].to_numpy()
        colors = [dataset_color_map[ds] for ds in sub["dataset"]]

        fig_w = max(6, len(sub) * figsize_per_item)
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        # keep color usage exactly as before
        bars = ax.bar(x, means, yerr=errs, color=colors, capsize=4, edgecolor="black")

        # annotate num_genes above bars (if present)
        for bar, val, err, ng in zip(bars, means, errs, nums):
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            offset = 6 if height >= 0 else -6
            if ng > 0:
                ax.annotate(str(ng),
                            xy=(bar.get_x() + bar.get_width()/2, height + (err if height >= 0 else -err)),
                            xytext=(0, offset), textcoords="offset points",
                            ha="center", va=va, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_ylabel("Pearson correlation (mean ± std)")
        ax.set_title(f"Gene list: {gl} — across datasets")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        # Legend uses the same colors as before
        present_datasets = sorted(sub["dataset"].unique().tolist())
        handles = [plt.Rectangle((0, 0), 1, 1, color=dataset_color_map[ds], edgecolor="black") for ds in present_datasets]
        ax.legend(handles, present_datasets, title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout()

        fname = folder / f"{_safe_filename(prefix)}_{_safe_filename(gl)}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        results[str(fname)] = fig
        if show:
            display(fig)
        plt.close(fig)

    # Combine saved PNGs into a single PDF (if any)
    png_files = sorted(folder.glob("*.png"))
    combined_pdf_path = None
    if png_files:
        images = []
        for p in png_files:
            try:
                im = Image.open(p)
                if im.mode in ("RGBA", "LA") or (im.mode == "P"):
                    im = im.convert("RGB")
                images.append(im)
            except Exception as e:
                warnings.warn(f"Failed to open image {p}: {e}")

        if images:
            combined_pdf_path = outdir / "by_gene_list_combined.pdf"
            try:
                images[0].save(combined_pdf_path, "PDF", resolution=200.0, save_all=True, append_images=images[1:])
            except Exception as e:
                warnings.warn(f"Failed to write combined PDF {combined_pdf_path}: {e}")
                combined_pdf_path = None

    return results, combined_pdf_path


def plot_summary_genes_vs_mean(best_df: pd.DataFrame,
                               outdir: str | None = DEFAULT_SUMMARY_PLOT_DIR,
                               filename: str = "summary_genes_vs_mean.png",
                               show: bool = False):
    """
    Scatter: x = highest_pearson_mean, y = num_genes, colored by dataset.
    """
    if best_df is None or best_df.empty:
        print("[plot_summary_genes_vs_mean] Empty DataFrame.")
        return None

    df = best_df.copy()
    need = {"dataset", "num_genes", "highest_pearson_mean"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[plot_summary_genes_vs_mean] Missing columns: {missing}")

    df["dataset"] = df["dataset"].astype("string").fillna("Unknown")
    df["num_genes"] = pd.to_numeric(df["num_genes"], errors="coerce")
    df["highest_pearson_mean"] = pd.to_numeric(df["highest_pearson_mean"], errors="coerce")
    df = df.dropna(subset=["num_genes", "highest_pearson_mean"])
    if df.empty:
        print("[plot_summary_genes_vs_mean] No valid rows.")
        return None

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    colors = _sample_cmap("tab20", len(datasets))
    color_map = {ds: colors[i] for i, ds in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        ax.scatter(
            sub["highest_pearson_mean"].to_numpy(dtype=float),
            sub["num_genes"].to_numpy(dtype=float),
            label=str(ds),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
            color=color_map[ds],
        )

    ax.set_xlabel("Mean Pearson correlation")
    ax.set_ylabel("Number of genes")
    ax.set_title("Gene count vs performance (colored by dataset)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()


    outdir = Path(outdir)
    path = outdir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")

    if show:
        display(fig)

    return fig


def plot_summary_patches_vs_mean(best_df: pd.DataFrame,
                                 outdir: str | None = DEFAULT_SUMMARY_PLOT_DIR,
                                 filename: str = "summary_patches_vs_mean.png",
                                 show: bool = False,
                                 figsize=(9, 7),
                                 size_scale: float = 0.05,
                                 min_size: float = 30,
                                 size_legend_values: list[int] = None):
    """
    Scatter:
      x = highest_pearson_mean
      y = num_training_patches_mean
      dot size = num_genes
      color = dataset
      label = gene_list (without .json), repelled with connectors
      size legend shows num_genes examples
    """
    if best_df is None or best_df.empty:
        print("[plot_summary_patches_vs_mean] Empty DataFrame.")
        return None

    need = {"dataset", "gene_list", "highest_pearson_mean", "num_training_patches_mean", "num_genes"}
    missing = need - set(best_df.columns)
    if missing:
        raise ValueError(f"[plot_summary_patches_vs_mean] Missing columns: {missing}")

    df = best_df.copy()
    df["dataset"] = df["dataset"].astype(str).fillna("Unknown")
    df["gene_list"] = df["gene_list"].astype(str).fillna("Unknown").str.replace(".json", "", regex=False)
    df["highest_pearson_mean"] = pd.to_numeric(df["highest_pearson_mean"], errors="coerce")
    df["num_training_patches_mean"] = pd.to_numeric(df["num_training_patches_mean"], errors="coerce")
    df["num_genes"] = pd.to_numeric(df["num_genes"], errors="coerce")

    df = df.dropna(subset=["highest_pearson_mean", "num_training_patches_mean", "num_genes"])
    if df.empty:
        print("[plot_summary_patches_vs_mean] No valid rows.")
        return None

    datasets = sorted(df["dataset"].unique().tolist())
    colors = _sample_cmap("tab20", len(datasets))
    color_map = {ds: colors[i] for i, ds in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=figsize)
    texts = []

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        ax.scatter(
            sub["num_training_patches_mean"],
            sub["highest_pearson_mean"],
            s=(sub["num_genes"] * size_scale).clip(lower=min_size),
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            color=color_map[ds],
            label=ds,
        )
        for _, row in sub.iterrows():
            texts.append(
                ax.text(
                    row["num_training_patches_mean"],
                    row["highest_pearson_mean"],
                    row["gene_list"],
                    fontsize=8,
                )
            )

    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )

    ax.set_xlabel("Mean number of training patches per split")
    ax.set_ylabel("Mean Pearson correlation")
    ax.set_title("Performance vs training patches vs  (dot size = num_genes)")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Dataset legend
    handles, labels = ax.get_legend_handles_labels()
    leg1 = ax.legend(handles, labels, title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Size legend
    if size_legend_values is None:
        size_legend_values = [int(df["num_genes"].min()),
                              int(df["num_genes"].median()),
                              int(df["num_genes"].max())]
        size_legend_values = sorted(set(size_legend_values))
    size_handles = [
        plt.scatter([], [],
                    s=max(min_size, val*size_scale),
                    color="gray", alpha=0.6, edgecolors="black")
        for val in size_legend_values
    ]
    leg2 = ax.legend(size_handles, [f"{v} genes" for v in size_legend_values],
                     title="Num genes (dot size)",
                     bbox_to_anchor=(1.02, 0.4), loc="upper left")
    ax.add_artist(leg1)

    fig.tight_layout()

    outdir = Path(outdir)
    path = outdir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")

    if show:
        display(fig)

    return fig


def plot_dataset_mean_across_runs(
    dataset_or_run: str,
    runs_root: str = DEFAULT_RUNS_ROOT,
    splits_root: str = DEFAULT_SPLITS_ROOT,
    outdir: str | Path | None = DEFAULT_SUMMARY_PLOT_DIR,
    candidate_runs: list | None = None,
    auto_discover_runs: bool = True,
    show: bool = False,
    dpi: int = 200,
) -> tuple[pd.DataFrame, "matplotlib.figure.Figure"]:
    """
    For a given dataset (or a run folder name), compute per-candidate-run the mean of
    per-sample mean Pearson correlations (i.e. average across samples belonging to the
    dataset_or_run). Plot one bar per candidate run showing:
        - bar height = mean across samples of that sample's mean Pearson (as before)
        - error bar = SD computed across ALL gene-level correlations used for those samples
        - bar annotation = number of UNIQUE genes used for those correlations

    Returns:
        (df_stats, fig)
        df_stats contains columns:
            ['run','dataset','gene_list','n_samples','mean_of_sample_means',
             'sd_on_genes','n_genes_used']
    """
    import matplotlib
    import matplotlib.pyplot as plt

    outdir = Path(outdir) if outdir is not None else Path(DEFAULT_SUMMARY_PLOT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    runs_root = Path(runs_root)
    splits_root = Path(splits_root)

    # 1) Determine the sample set for the provided dataset_or_run
    samples = []
    sample_col_candidates = ["test_sample", "sample_id", "SampleID"]

    if str(dataset_or_run).startswith("run_"):
        try:
            df_splits = get_test_splits(dataset_or_run, runs_root=str(runs_root), splits_root=str(splits_root))
            sample_col = _find_col_ci(df_splits, sample_col_candidates)
            if sample_col is None:
                raise KeyError(f"No sample column found in splits for run {dataset_or_run}")
            samples = sorted(df_splits[sample_col].dropna().astype(str).unique().tolist())
        except Exception as e:
            raise RuntimeError(f"Failed to read splits for run {dataset_or_run}: {e}")
        label_name = dataset_or_run
    else:
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
                sc = _find_col_ci(df_t, sample_col_candidates)
                if sc is None:
                    continue
                sample_set.update(df_t[sc].dropna().astype(str).tolist())
            except Exception:
                continue
        samples = sorted(sample_set)
        label_name = dataset_or_run

    if not samples:
        raise ValueError("No samples found for the provided dataset/run.")

    # 2) Candidate runs discovery / normalization
    if candidate_runs is None:
        if not auto_discover_runs:
            raise ValueError("candidate_runs is None and auto_discover_runs is False.")
        candidate_runs = [p.name for p in sorted(runs_root.glob("run_*")) if p.is_dir()]

    candidate_runs = [str(r).split("/")[-1] for r in candidate_runs]

    # 3) For each candidate run, compute per-sample mean Pearson, and compute SD across gene-level corrs
    results = []
    for run_name in candidate_runs:
        try:
            # load splits for this run
            try:
                df_splits_run = get_test_splits(run_name, runs_root=str(runs_root), splits_root=str(splits_root))
            except Exception:
                continue

            sample_col_run = _find_col_ci(df_splits_run, sample_col_candidates)
            if sample_col_run is None:
                continue

            samples_present = df_splits_run[sample_col_run].astype(str).dropna().unique().tolist()
            samples_in_common = sorted(set(samples) & set(samples_present))
            if not samples_in_common:
                continue

            # load gene-level correlations for best model of this run
            try:
                best_info, ds_name, df_genes = extract_best_model_gene_corrs(run_name, runs_root=str(runs_root), verbose=False)
            except Exception:
                continue

            # create long df with per-split entries attached to samples
            try:
                df_long = merge_kfold_gene_corrs_with_test_metadata(df_genes, df_splits_run)
            except Exception:
                continue

            sample_col_long = _find_col_ci(df_long, sample_col_candidates)
            if sample_col_long is None:
                continue

            # Filter df_long to rows for the samples_in_common
            df_filtered = df_long[df_long[sample_col_long].astype(str).isin(samples_in_common)].copy()

            if df_filtered.empty:
                continue

            # 3a) per-sample means (as before)
            per_sample_means = (
                df_filtered.groupby(sample_col_long)["corr"]
                .agg(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
                .dropna()
                .to_numpy()
            )
            if len(per_sample_means) == 0:
                continue
            mean_across_samples = float(pd.Series(per_sample_means).mean())

            # 3b) sd across ALL gene-level correlations used for these samples
            # convert corr column to numeric and dropna
            all_corrs = pd.to_numeric(df_filtered["corr"], errors="coerce").dropna().to_numpy()
            if all_corrs.size == 0:
                continue
            # use sample standard deviation (ddof=1). change to ddof=0 if you prefer population SD.
            sd_on_genes = float(all_corrs.std(ddof=1)) if all_corrs.size > 1 else 0.0

            # number of unique genes used (use 'gene' column if present)
            if "gene" in df_filtered.columns:
                n_genes_used = int(pd.Series(df_filtered["gene"].dropna().unique()).size)
            else:
                # fallback: count unique gene-sample pairs divided by number of samples (best-effort)
                n_genes_used = int(max(1, df_filtered.shape[0] // max(1, len(samples_in_common))))

            n_samples_used = len(set(df_filtered[sample_col_long].astype(str).unique().tolist()))

            # discover gene_list for label
            gene_list_name = ""
            run_path = runs_root / run_name
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

            results.append({
                "run": run_name,
                "dataset": ds_name,
                "gene_list": gene_list_name or run_name,
                "n_samples": n_samples_used,
                "mean_of_sample_means": mean_across_samples,
                "sd_on_genes": sd_on_genes,
                "n_genes_used": n_genes_used,
            })

        except Exception as e:
            warnings.warn(f"[plot_dataset_mean_across_runs] skipping run {run_name}: {e}")
            continue

    if not results:
        raise ValueError("No candidate runs produced statistics — check candidate_runs and splits.")

    df_stats = pd.DataFrame(results).convert_dtypes()
    df_stats = df_stats.sort_values("mean_of_sample_means", ascending=False).reset_index(drop=True)

    # 4) Plot with sd_on_genes as error bar and annotate n_genes_used on each bar
    fig, ax = plt.subplots(figsize=(max(8, len(df_stats) * 0.9), 6))
    x = np.arange(len(df_stats))
    means = df_stats["mean_of_sample_means"].astype(float).to_numpy()
    errs = df_stats["sd_on_genes"].astype(float).fillna(0.0).to_numpy()
    labels = [(str(gl) + " | " + str(ds)) for gl, ds in zip(df_stats["gene_list"], df_stats["dataset"])]

    datasets_seen = sorted(df_stats["dataset"].fillna("Unknown").unique().tolist())
    color_map = _assemble_color_map(datasets_seen)
    colors = [color_map.get(ds, (0.4, 0.4, 0.4)) for ds in df_stats["dataset"]]

    bars = ax.bar(x, means, yerr=errs, color=colors, capsize=5, edgecolor="black")
    for bar, n_genes in zip(bars, df_stats["n_genes_used"]):
        # annotate number of genes above the bar
        ax.annotate(n_genes,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Pearson (average across samples)")
    ax.set_title(f"Dataset/Run: {dataset_or_run} — mean across samples (per candidate run)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[ds],edgecolor="black") for ds in datasets_seen]
    if handles:
        ax.legend(handles, datasets_seen, title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()

    # save
    fname = outdir / f"{_safe_filename(str(label_name))}_mean_across_runs.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    if show:
        display(fig)
    plt.close(fig)

    return df_stats, fig


def compare_models(
    run: str,
    runs_root: str = DEFAULT_RUNS_ROOT,
    show: bool = False,
):
    """
    Visualize encoder-level dataset results for a given run as a point plot with error bars.
    Compatible with generate_all_plots: uses load_run() and returns a matplotlib Figure.
    If show=True, displays the figure inline (Jupyter-friendly).

    Args:
        run (str): Run folder name.
        runs_root (str): Root directory containing run folders.
        show (bool): If True, display the figure inline.

    Returns:
        matplotlib.figure.Figure | None
    """
    run_dir = Path(runs_root) / run

    # Use the shared loader to parse dataset_results.json and pick up dataset name
    try:
        dataset_name, results_df, _best = load_run(run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[visualize_dataset_results] {e}")
        return None

    if results_df.empty:
        print(f"[visualize_dataset_results] No encoder results found in {run_dir}")
        return None

    # Extract encoders + stats
    encoders = results_df.get("encoder_name", pd.Series([], dtype="string")).astype(str).tolist()
    means = pd.to_numeric(results_df.get("pearson_mean", pd.Series([], dtype="float")), errors="coerce").to_numpy()
    stds  = pd.to_numeric(results_df.get("pearson_std",  pd.Series([], dtype="float")), errors="coerce").to_numpy()

    # Find config.json anywhere in the run dir to show gene list info (optional)
    gene_list_name = ""
    gene_count = None
    for dirpath, _, filenames in os.walk(run_dir):
        if "config.json" in filenames:
            config_path = Path(dirpath) / "config.json"
            try:
                with open(config_path, "r") as cf:
                    config = json.load(cf)
                gene_list_name = config.get("gene_list", "") or config.get("genes", "") or ""
                # Try to count genes if file exists and contains a JSON list
                if gene_list_name:
                    gl_path = Path(gene_list_name)
                    if not gl_path.is_file():
                        gl_path = (Path(dirpath) / gene_list_name)
                    if gl_path.is_file():
                        try:
                            with open(gl_path, "r") as gf:
                                genes = json.load(gf)
                            if isinstance(genes, list):
                                gene_count = len(genes)
                        except Exception:
                            # Silently ignore if it's not JSON; you can extend to txt/tsv if needed
                            pass
            except Exception:
                pass
            break

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(encoders))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=5, markersize=8, linestyle="")

    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=45, ha="right")
    ax.set_ylabel("Pearson correlation")

    title = f"Dataset: {dataset_name}"
    if gene_list_name:
        title += f" | Gene List: {gene_list_name}"
    if gene_count is not None:
        title += f" ({gene_count} genes)"
    ax.set_title(title)

    if show:
        display(fig)

    return fig


def generate_all_plots(
    run: str,
    group_by: Union[str, List[str], None] = None,
    runs_root: str = DEFAULT_RUNS_ROOT,
    splits_root: str = DEFAULT_SPLITS_ROOT,
    curated_xlsx: Optional[str] = DEFAULT_CURATED_XLSX,
    extra_metadata_csv: Optional[str] = DEFAULT_EXTRA_METADATA,
    top_n: int = 30,
    show: bool = False,
) -> Dict[str, Path]:
    """
    New workflow:
      - Always produce base plots (gene_barplot, gene_hist, per_sample).
      - If `group_by` is provided (str or list[str]), produce only those grouped plots
        *when the needed metadata/columns exist*.
      - Save individual PNGs into <runs_root>/<run>/plots and also write a combined
        multi-page PDF named plots.pdf containing all PNGs found in that folder.

    Returns:
        arts: mapping of plot_name -> Path
    """
    # Paths
    run_dir = Path(runs_root) / run
    outdir = run_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Best model + gene correlations (k-fold aware)
    best, dataset_name, df_genes = extract_best_model_gene_corrs(run, runs_root=runs_root, verbose=False)
    encoder_name = best.get("encoder_name", "Unknown")

    # 2) Splits (+ optional HEST directory metadata inside get_test_splits)
    df_splits = get_test_splits(run, runs_root=runs_root, splits_root=splits_root, extra_metadata_csv=extra_metadata_csv)

    # 3) Long per-split df
    df_long = merge_kfold_gene_corrs_with_test_metadata(df_genes, df_splits)

    # 4) Curated annotations (optional; needed for gene-level grouping by panel/cell_type/condition)
    curated_ok = isinstance(curated_xlsx, str) and Path(curated_xlsx).exists()
    if curated_ok:
        df_genes_annot = annotate_genes_with_curated(df_genes, curated_xlsx)
        df_long_annot  = annotate_genes_with_curated(df_long,  curated_xlsx)
    else:
        df_genes_annot = df_genes
        df_long_annot  = df_long

    arts: Dict[str, Path] = {}

    # ---------------- Base plots (always) ----------------
    fig = compare_models(run, runs_root=runs_root, show=show)
    if fig is not None:
        p = (Path(runs_root) / run / "plots" / "model_comparison.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        if show:
            display(fig)
        plt.close(fig)
        arts["model_comparison"] = p

    # Gene barplot (ungrouped)
    fig = plot_gene_correlation_barplot(df_genes, dataset_name, encoder_name)
    p = outdir / "gene_barplot.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if show: display(fig)
    plt.close(fig)
    arts["gene_barplot"] = p

    # Histogram
    fig = plot_gene_correlation_histogram(df_long, dataset_name, encoder_name)
    p = outdir / "gene_hist.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if show: display(fig)
    plt.close(fig)
    arts["gene_hist"] = p

    # Per-sample (no grouping)
    fig = plot_corrs_by_sample(df_long, dataset_name, encoder_name, group_by=None)
    p = outdir / "per_sample.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if show: display(fig)
    plt.close(fig)
    arts["per_sample"] = p

    # ---------------- Gene-level grouped barplots (only what user requested) ----------------
    # normalize group_by to a list
    requested_groups: List[str] = []
    if group_by is not None:
        requested_groups = [group_by] if isinstance(group_by, str) else list(group_by)

    # Your grouped gene barplot supports: panel / cell_type / condition
    valid_gene_groups = {"panel", "cell_type", "condition"}

    for gb in requested_groups:
        if gb not in valid_gene_groups:
            # skip silently if user asked for something not supported by this function
            continue
        # only plot if curated annotations are available and column is present with non-NA values
        if curated_ok and (gb in df_genes_annot.columns) and df_genes_annot[gb].notna().any():
            fig = plot_gene_correlation_barplot_grouped(
                df_genes_annot,           # your function signature
                group_by=gb,              # pass the requested group
                show_mean=True
            )
            p = outdir / f"gene_barplot_by_{gb}.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            if show: display(fig)
            plt.close(fig)
            arts[f"gene_barplot_by_{gb}"] = p
        # if not curated / or column empty -> skip

    # ---------------- Per-sample grouped (optional, if user also wants these) ----------------
    # If you also want to produce per-sample grouped plots based on the same `group_by` items:
    for gb in requested_groups:
        # pick annotated long df if it has the column; else fall back to df_long
        if gb in df_long_annot.columns and df_long_annot[gb].notna().any():
            dplot = df_long_annot
        elif gb in df_long.columns and df_long[gb].notna().any():
            dplot = df_long
        else:
            continue

        print('plot sample-level group by', gb)

        fig = plot_corrs_by_sample(dplot, dataset_name, encoder_name, group_by=gb)
        p = outdir / f"per_sample_by_{gb}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        if show: display(fig)
        plt.close(fig)
        arts[f"per_sample_by_{gb}"] = p

    # ---- Per-sample Pearson vs sample metadata (n_obs / mean_total_counts / ... ) ----
    try:
        more_plots = plot_pearson_vs_sample_metadata(
            df_long=df_long,
            df_splits=df_splits,
            outdir=outdir,
            # you can pass metadata_cols_preference explicitly if you prefer
            show=show,
        )
        arts.update(more_plots)
    except Exception as e:
        print(f"[warning] failed to create pearson vs metadata plots: {e}")

    # ---------------- Combine all PNGs into a single PDF called plots.pdf ----------------
    try:
        from PIL import Image
        png_files = sorted(outdir.glob("*.png"))
        if len(png_files) > 0:
            # open images and convert to RGB as needed
            pil_images = []
            for idx, png in enumerate(png_files):
                img = Image.open(png)
                # convert RGBA -> RGB to avoid transparency issues in PDF
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img = img.convert("RGB")
                else:
                    img = img.convert("RGB")
                pil_images.append(img)

            pdf_path = outdir / "plots.pdf"
            # save first image and append the rest as additional pages
            first_img, rest = pil_images[0], pil_images[1:]
            first_img.save(pdf_path, "PDF", resolution=200.0, save_all=True, append_images=rest)
            arts["plots_pdf"] = pdf_path
    except Exception as e:
        # non-fatal: if PIL missing or something goes wrong, continue but record no PDF
        print(f"[warning] Failed to create combined PDF: {e}")

    return arts


def compare_runs_and_plot(run_a, label_a, run_b, label_b):
    # extract
    _, _, df_a = extract_best_model_gene_corrs(run=run_a)
    _, _, df_b = extract_best_model_gene_corrs(run=run_b)

    # get intersection
    genes_a = set(df_a["gene"])
    genes_b = set(df_b["gene"])
    common = sorted(genes_a & genes_b)

    # filter to common
    df_a_common = df_a[df_a["gene"].isin(common)].rename(
        columns={"mean_corr": f"mean_corr_{label_a}"}
    )
    df_b_common = df_b[df_b["gene"].isin(common)].rename(
        columns={"mean_corr": f"mean_corr_{label_b}"}
    )

    # merge
    wide = df_a_common.merge(df_b_common, on="gene")

    # reshape for plotting
    df_plot = wide.melt(
        id_vars="gene",
        value_vars=[f"mean_corr_{label_a}", f"mean_corr_{label_b}"],
        var_name="dataset",
        value_name="mean_corr"
    )
    df_plot["dataset"] = df_plot["dataset"].str.replace("mean_corr_", "")

    # order genes by highest correlation
    gene_order = (
        df_plot.groupby("gene")["mean_corr"]
        .max()
        .sort_values(ascending=False)
        .index
    )
    df_plot["gene"] = pd.Categorical(df_plot["gene"], categories=gene_order, ordered=True)

    # scale figure size
    n_genes = len(gene_order)
    fig_height = max(6, n_genes * 0.25)  # at least 6 inches tall
    fig_width = 12

    # plot horizontal grouped bars
    plt.figure(figsize=(fig_width, fig_height))
    for i, dataset in enumerate(df_plot["dataset"].unique()):
        subset = df_plot[df_plot["dataset"] == dataset]
        plt.barh(
            [y + i*0.4 for y in range(len(subset))],
            subset["mean_corr"],
            height=0.4,
            label=dataset,
        )

    plt.yticks([y + 0.2 for y in range(len(gene_order))], gene_order, fontsize=6)
    plt.xlabel("Mean correlation")
    plt.ylabel("Gene")
    plt.legend(title="Dataset")
    plt.title(f"Mean correlation per gene ({label_a} vs {label_b})")
    plt.gca().invert_yaxis()  # highest correlation at the top
    plt.tight_layout()
    plt.show()

    return common, wide


def grid_plot_for_datasets(
    datasets: list,
    runs_root: str = DEFAULT_RUNS_ROOT,
    splits_root: str = DEFAULT_SPLITS_ROOT,
    outdir: str | Path | None = DEFAULT_SUMMARY_PLOT_DIR,
    candidate_runs: list | None = None,
    auto_discover_runs: bool = True,
    per_page: int = 6,       # 6 datasets per page
    ncols: int = 3,
    nrows: int = 2,
    figsize_per_subplot: tuple = (5, 3),
    dpi: int = 200,
    show: bool = False,
):
    """
    For each dataset in `datasets`, call plot_dataset_mean_across_runs(dataset, ...) to
    compute df_stats (per-run summary). Then draw a small subplot per dataset and
    arrange them into pages of `per_page` (3x2 default). Save multi-page PDF.

    Returns:
        pdf_path: Path to multi-page PDF
        df_combined: concatenated DataFrame of all dataset-level stats
    """
    outdir = Path(outdir) if outdir is not None else Path(DEFAULT_SUMMARY_PLOT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    plotted_datasets = []  # keep order

    # 1) call the dataset-level routine for each requested dataset
    for ds in datasets:
        try:
            df_stats, _ = plot_dataset_mean_across_runs(
                dataset_or_run=ds,
                runs_root=runs_root,
                splits_root=splits_root,
                outdir=outdir,
                candidate_runs=candidate_runs,
                auto_discover_runs=auto_discover_runs,
                show=False,
                dpi=dpi,
            )
            if df_stats is None or df_stats.empty:
                print(f"[grid] no stats for dataset {ds} -> skipping")
                continue
            # keep dataset name recorded so we can plot it later
            df_stats["dataset_requested"] = ds
            all_stats.append(df_stats)
            plotted_datasets.append(ds)
        except Exception as e:
            print(f"[grid] failed for dataset {ds}: {e}")
            continue

    if not all_stats:
        raise RuntimeError("No dataset produced stats; nothing to plot.")

    df_combined = pd.concat(all_stats, ignore_index=True).convert_dtypes()

    # 2) Build multi-page PDF with per-dataset small subplots (per_page datasets per page)
    pdf_path = outdir / "datasets_mean_across_runs_grid.pdf"
    with PdfPages(pdf_path) as pdf:
        # iterate pages
        for page_start in range(0, len(plotted_datasets), per_page):
            page_ds = plotted_datasets[page_start:page_start + per_page]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     figsize=(ncols * figsize_per_subplot[0], nrows * figsize_per_subplot[1]))
            axes = axes.flatten()
            # blank all axes first
            for ax in axes:
                ax.axis("off")

            for i, ds in enumerate(page_ds):
                ax = axes[i]
                ax.axis("on")

                # select the rows corresponding to this dataset (the df_stats produced earlier)
                df_sub = df_combined[df_combined["dataset_requested"] == ds].sort_values("mean_of_sample_means", ascending=False)
                if df_sub.empty:
                    ax.set_title(ds)
                    continue

                x = range(len(df_sub))
                means = df_sub["mean_of_sample_means"].astype(float).to_numpy()
                errs = df_sub["sd_on_genes"].astype(float).fillna(0.0).to_numpy()
                labels = [(str(gl) + " | " + str(rn)) for gl, rn in zip(df_sub["gene_list"], df_sub["dataset"])]
                # color per dataset of the compared run (reuse _assemble_color_map)
                datasets_seen = sorted(df_sub["dataset"].fillna("Unknown").unique().tolist())
                color_map = _assemble_color_map(datasets_seen)
                colors = [color_map.get(dsname, (0.5, 0.5, 0.5)) for dsname in df_sub["dataset"]]

                bars = ax.bar(x, means, yerr=errs, color=colors, capsize=4, edgecolor="black")
                # annotate number of genes used on each bar
                if "n_genes_used" in df_sub.columns:
                    for bar, ng in zip(bars, df_sub["n_genes_used"]):
                        ax.annotate(f"{ng} genes",
                                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    xytext=(0, 4),
                                    textcoords="offset points",
                                    ha="center", va="bottom", fontsize=7)
                else:
                    # fallback: annotate n_samples if present
                    if "n_samples" in df_sub.columns:
                        for bar, ns in zip(bars, df_sub["n_samples"]):
                            ax.annotate(f"{ns} samples",
                                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                        xytext=(0, 4),
                                        textcoords="offset points",
                                        ha="center", va="bottom", fontsize=7)

                ax.set_xticks(x)
                # shorten overly long labels
                short_labels = []
                for L in labels:
                    Ls = str(L)
                    short_labels.append(Ls if len(Ls) <= 30 else (Ls[:27] + "…"))
                ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)

                ax.set_title(ds, fontsize=9)
                ax.set_ylabel("Mean Pearson", fontsize=8)
                ax.grid(True, axis="y", linestyle=":", alpha=0.4)
                ax.tick_params(axis='y', labelsize=7)

            fig.tight_layout(rect=[0, 0, 0.92, 1.0])  # leave room for legend if needed
            pdf.savefig(fig)
            if show:
                display(fig)
            plt.close(fig)

    return pdf_path, df_combined
