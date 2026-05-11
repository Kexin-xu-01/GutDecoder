"""
_per_sample.py — per-sample plots.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from adjustText import adjust_text

from gutdecoder.config import (
    RESULTS_ROOT as DEFAULT_RUNS_ROOT,
    DATA_ROOT as DEFAULT_SPLITS_ROOT,
    SUMMARY_PLOT_DIR as DEFAULT_SUMMARY_PLOT_DIR,
    HEST_METADATA_CSV as DEFAULT_EXTRA_METADATA,
)

from ._helpers import _find_col_ci, _sample_cmap, _assemble_color_map, _safe_filename, _format_title
from ._io import get_test_splits, extract_best_model_gene_corrs, merge_kfold_gene_corrs_with_test_metadata


def plot_corrs_by_sample(
    df,
    dataset_name,
    encoder_name,
    group_by: str | None = None,
    figsize=(18, 6)
):
    """
    Plot per-split gene correlations grouped by metadata, and return the figure object.

    Args:
        df (pd.DataFrame): must contain columns:
            [split, gene, corr, test_sample, Sample_type, Location, cell_type, condition]
        group_by (str or None): column to color/group samples by
            (e.g. 'Sample_type' or 'Location'). If None, no grouping (all samples same color).

    Returns:
        matplotlib.figure.Figure
    """

    df = df.copy()

    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"{group_by} must be a column in df")

    # ---------------- ordering of x-axis ----------------
    if group_by is not None:
        # Order groups by mean corr, then samples within each group
        group_means = df.groupby(group_by)["corr"].mean().sort_values(ascending=False)
        group_order = group_means.index.tolist()
        x_order = []
        for grp in group_order:
            samples = (
                df.loc[df[group_by] == grp, "test_sample"]
                  .dropna()
                  .drop_duplicates()
                  .tolist()
            )
            x_order.extend(samples)
        # deduplicate while preserving order
        x_order = list(dict.fromkeys(x_order))
    else:
        # Just order test_sample by mean corr
        sample_means = df.groupby("test_sample")["corr"].mean().sort_values(ascending=False)
        x_order = sample_means.index.tolist()

    df["test_sample"] = pd.Categorical(df["test_sample"], categories=x_order, ordered=True)

    # ---------------- plotting ----------------
    fig, ax = plt.subplots(figsize=figsize)

    if group_by is not None:
        sns.boxplot(
            data=df,
            x="test_sample",
            y="corr",
            hue=group_by,
            showfliers=False,
            palette="tab20",
            ax=ax,
        )
        ax.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xlabel(f"Test Sample (grouped by {group_by})")
    else:
        sns.boxplot(
            data=df,
            x="test_sample",
            y="corr",
            color="skyblue",
            showfliers=False,
            ax=ax,
        )
        # no legend
        ax.set_xlabel("Test Sample")

    # ---------------- styling ----------------
    plt.xticks(rotation=60, ha="right")
    ax.set_ylabel("Gene Correlation")
    ax.set_title(_format_title("Per-sample correlation", dataset_name, encoder_name))
    plt.tight_layout()

    return fig


def plot_pearson_vs_sample_metadata(
    df_long: pd.DataFrame,
    df_splits: pd.DataFrame,
    outdir: Path,
    sample_col_candidates: list[str] = ["test_sample", "sample_id", "SampleID"],
    metadata_cols_preference: list[str] | None = None,
    show: bool = False,
    dpi: int = 200,
) -> Dict[str, Path]:
    """
    Create scatter plots of per-sample mean Pearson correlation vs sample-level *in-tissue* metadata.
    Points are colored by Run (if available).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arts: Dict[str, Path] = {}

    if metadata_cols_preference is None:
        metadata_cols_preference = [
            "n_obs_in_tissue",
            "mean_total_counts_in_tissue",
            "mean_n_genes_by_counts_in_tissue",
            "mean_log1p_total_counts_in_tissue",
            "mean_log1p_n_genes_by_counts_in_tissue",
        ]

    # determine sample column
    sample_col = _find_col_ci(df_long, sample_col_candidates)
    if sample_col is None:
        sample_col = _find_col_ci(df_splits, sample_col_candidates)
    if sample_col is None:
        raise KeyError("Could not find sample id column.")

    # compute per-sample mean Pearson
    gb = df_long.groupby(sample_col)["corr"]
    df_sample_mean = (
        gb.agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "mean_pearson", "size": "n_genes_per_sample"})
    )

    # merge metadata
    merged = df_sample_mean.merge(
        df_splits.drop_duplicates(subset=[sample_col]),
        on=sample_col,
        how="left",
    )

    # detect run column
    run_col = _find_col_ci(merged, ["run", "Run", "RUN"])

    # create color mapping if run exists
    if run_col is not None:
        runs = sorted(merged[run_col].dropna().astype(str).unique())
        colors = _sample_cmap("tab20", len(runs))
        color_map = {r: colors[i] for i, r in enumerate(runs)}
        color_map = {r: colors[i] for i, r in enumerate(runs)}
    else:
        runs = []
        color_map = {}

    # loop over metadata columns
    for meta in metadata_cols_preference[:6]:
        meta_col = _find_col_ci(merged, [meta])
        if meta_col is None:
            print(f"[plot] metadata column '{meta}' not found - skipping.")
            continue

        df_plot = merged[[sample_col, "mean_pearson", meta_col] + ([run_col] if run_col else [])]
        df_plot = df_plot.dropna(subset=["mean_pearson", meta_col]).copy()

        df_plot[meta_col] = pd.to_numeric(df_plot[meta_col], errors="coerce")
        df_plot = df_plot.dropna(subset=[meta_col])
        if df_plot.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            df_plot[meta_col],
            df_plot["mean_pearson"],
            s=60,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )

        ax.set_xlabel(meta_col)
        ax.set_ylabel("Mean Pearson correlation (per-sample)")
        ax.set_title(f"Mean Pearson per sample vs {meta_col} (in-tissue)")
        ax.grid(True, linestyle=":", alpha=0.4)

        # Label samples
        texts = []
        for _, r in df_plot.iterrows():
            texts.append(ax.text(r[meta_col], r["mean_pearson"], str(r[sample_col]), fontsize=8))

        try:
            adjust_text(
                texts,
                ax=ax,
                expand_points=(1.2, 1.6),
                expand_text=(1.2, 1.6),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
        except Exception:
            pass

        fig.tight_layout()
        fname = f"pearson_vs_{meta_col}.png"
        path = outdir / fname
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if show:
            display(fig)
        plt.close(fig)

        arts[f"pearson_vs_{meta_col}"] = path

    return arts


def plot_sample_across_runs(
    sample_id: str,
    runs: list | None = None,
    runs_root: str = DEFAULT_RUNS_ROOT,
    splits_root: str = DEFAULT_SPLITS_ROOT,
    outdir: str | Path | None = DEFAULT_SUMMARY_PLOT_DIR,
    auto_discover: bool = True,
    show: bool = False,
    dpi: int = 200,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """
    For a given sample_id (e.g. "XeniumPR8S1ROI8"), find that sample in the specified runs
    (or auto-discover runs under `runs_root`), compute the per-run mean Pearson correlation
    for that sample (mean across genes) and std / number-of-genes, and plot a bar chart.

    X-axis labels are now "gene_list | dataset" (gene_list discovered from config.json where possible).

    Returns:
        (df_summary, fig)
    """
    outdir = Path(outdir) if outdir is not None else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    runs_root = Path(runs_root)
    candidate_runs = []
    if runs is None:
        if not auto_discover:
            raise ValueError("runs is None and auto_discover is False — nothing to do.")
        for ent in sorted(runs_root.iterdir()):
            if ent.is_dir() and ent.name.startswith("run_"):
                candidate_runs.append(ent.name)
    else:
        candidate_runs = list(runs)

    summary_rows = []
    sample_col_candidates = ["test_sample", "sample_id", "SampleID"]

    for run_name in candidate_runs:
        try:
            # get test splits for this run
            try:
                df_splits = get_test_splits(run_name, runs_root=str(runs_root), splits_root=splits_root)
            except Exception:
                continue

            # check if sample present in splits
            sample_col = _find_col_ci(df_splits, sample_col_candidates)
            present = False
            if sample_col is not None:
                present = df_splits[sample_col].astype(str).isin([sample_id]).any()

            if not present:
                continue

            # extract best model gene corrs
            best_info, dataset_name, df_genes = extract_best_model_gene_corrs(run_name, runs_root=str(runs_root), verbose=False)

            # attempt to discover gene_list from config.json under the run dir
            gene_list_name = ""
            run_path = Path(runs_root) / run_name
            for dirpath, _, filenames in os.walk(run_path):
                if "config.json" in filenames:
                    try:
                        with open(Path(dirpath) / "config.json", "r") as cf:
                            cfg = json.load(cf)
                        gene_list_name = cfg.get("gene_list") or cfg.get("genes") or ""
                        # if it's a path, keep only the filename portion
                        if isinstance(gene_list_name, str) and gene_list_name:
                            gene_list_name = Path(str(gene_list_name)).name
                    except Exception:
                        gene_list_name = ""
                    break
            if not gene_list_name:
                # fallback to using run name so label isn't empty
                gene_list_name = run_name

            # produce long df merged with test metadata (so it has 'test_sample')
            df_long = merge_kfold_gene_corrs_with_test_metadata(df_genes, df_splits)

            # locate sample column in the long DF
            sample_col_long = _find_col_ci(df_long, sample_col_candidates)
            if sample_col_long is None:
                continue

            # filter to this sample
            df_sample = df_long[df_long[sample_col_long].astype(str) == sample_id]
            if df_sample.empty:
                continue

            # compute mean and std across genes for that sample
            mean_pearson = pd.to_numeric(df_sample["corr"], errors="coerce").dropna().mean()
            std_pearson = pd.to_numeric(df_sample["corr"], errors="coerce").dropna().std()
            n_genes = df_sample.shape[0]

            summary_rows.append({
                "run": run_name,
                "dataset": dataset_name,
                "gene_list": gene_list_name,
                "sample_id": sample_id,
                "mean_pearson": float(mean_pearson) if not math.isnan(mean_pearson) else pd.NA,
                "std_pearson": float(std_pearson) if not math.isnan(std_pearson) else pd.NA,
                "n_genes": int(n_genes),
            })

        except Exception as e:
            print(f"[plot_sample_across_runs] Skipping run {run_name}: {e}")
            continue

    if not summary_rows:
        raise ValueError(f"Sample {sample_id!r} not found in any examined runs under {runs_root} (or provided runs).")

    df_summary = pd.DataFrame(summary_rows).convert_dtypes()
    df_summary = df_summary.sort_values("mean_pearson", ascending=False).reset_index(drop=True)

    # Prepare labels using gene_list | dataset
    labels = [(str(g) + " | " + str(d)) for g, d in zip(df_summary["gene_list"], df_summary["dataset"])]

    fig, ax = plt.subplots(figsize=(max(8, len(df_summary) * 0.8), 6))
    x = np.arange(len(df_summary))
    means = df_summary["mean_pearson"].astype(float).to_numpy()
    errs = df_summary["std_pearson"].astype(float).fillna(0.0).to_numpy()

    datasets = sorted(df_summary["dataset"].fillna("Unknown").unique().tolist())
    color_map = _assemble_color_map(datasets)
    colors = [color_map.get(ds, (0.4, 0.4, 0.4)) for ds in df_summary["dataset"]]

    bars = ax.bar(x, means, yerr=errs, color=colors, capsize=4, edgecolor="black")
    for bar, ng in zip(bars, df_summary["n_genes"]):
        height = bar.get_height()
        ax.annotate(str(ng), xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Pearson correlation (sample)")
    ax.set_title(f"Sample {sample_id} — mean Pearson across runs (n_genes annotated)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[ds],edgecolor="black") for ds in datasets]
    ax.legend(handles, datasets, title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()

    fname = outdir / f"sample_{_safe_filename(sample_id)}_across_runs.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    if show:
        display(fig)
    plt.close(fig)

    return df_summary, fig
