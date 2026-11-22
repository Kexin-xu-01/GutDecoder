"""
_per_gene.py — per-gene plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._helpers import _sample_cmap, _format_title


def plot_gene_correlation_barplot(df, dataset_name, encoder_name,
                                          value_col="mean_corr",
                                          top_n=30, figsize=(12, 7)):
    """
    Horizontal barplot of top-N genes, showing mean correlations with std dev as error bars.

    Args:
        df: DataFrame with at least ['gene', value_col]. Optional ['std_corr'] for error bars.
        dataset_name: str
        encoder_name: str
        group_col: str, metadata column for coloring (currently not used in this minimal version).
        value_col: str, column with mean correlations (default 'mean_corr').
        top_n: int, number of top genes to show.
        figsize: tuple
    """
    if df.empty or value_col not in df:
        fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
        ax.text(0.5, 0.5, "No gene correlations to plot", ha="center", va="center")
        return fig

    d = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    means = d[value_col].astype(float).to_numpy()
    y_pos = np.arange(len(d))

    # Use std_corr if available
    if "std_corr" in d.columns:
        errs = d["std_corr"].astype(float).to_numpy()
        ax.barh(y_pos, means, xerr=errs, align="center", alpha=0.7, ecolor="black", capsize=3)
    else:
        ax.barh(y_pos, means, align="center", alpha=0.7)

    ax.set_yticks(y_pos, d["gene"].astype(str))
    ax.set_xlabel("Pearson correlation (mean ± std)")
    ax.set_title(_format_title("Gene Correlation", dataset_name, encoder_name))
    fig.tight_layout()
    return fig


def plot_gene_correlation_barplot_grouped(df_genes, group_by='cell_type', show_mean=True):
    """
    Plot individual gene mean correlations grouped by a specified column (cell_type or condition).

    Args:
        df_genes (pd.DataFrame): Must contain 'gene', 'mean_corr', 'std_corr', and the grouping column.
        group_by (str): Column name to group by ('cell_type' or 'condition').
        show_mean (bool): Whether to show a horizontal line per group indicating mean correlation.

    Returns:
        matplotlib.figure.Figure: Figure object for further saving or manipulation.
    """
    if group_by not in ['cell_type', 'condition', 'panel','tcr','coeliac','run']:
        raise ValueError("group_by must be 'cell_type' or 'condition' or 'panel' or 'tcr' or'coeliac' or 'run'")

    # Filter out rows where grouping column is NA
    df_plot = df_genes[df_genes[group_by].notna()].copy()

    # Compute average mean_corr per group for ordering
    group_order = df_plot.groupby(group_by)['mean_corr'].mean().sort_values(ascending=False).index.tolist()

    # Sort genes by group and descending mean_corr
    df_plot[f'{group_by}_ordered'] = pd.Categorical(df_plot[group_by], categories=group_order, ordered=True)
    df_plot = df_plot.sort_values([f'{group_by}_ordered', 'mean_corr'], ascending=[True, False])

    # Map colors
    colors = _sample_cmap('tab20', len(group_order))
    color_map = {grp: colors[i] for i, grp in enumerate(group_order)}
    colors = [color_map[grp] for grp in df_plot[group_by]]

    # X positions
    x = np.arange(len(df_plot))

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot)*0.2), 6))

    # Plot bars with error bars
    ax.bar(x, df_plot['mean_corr'], yerr=df_plot['std_corr'], color=colors, edgecolor='black', capsize=4)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['gene'], rotation=90)
    ax.set_ylabel("Mean Pearson Correlation")
    ax.set_title(f"Gene Mean Correlations by {group_by.capitalize()}")

    # Draw horizontal line per group if requested
    if show_mean:
        start_idx = 0
        for grp in group_order:
            grp_genes = df_plot[df_plot[group_by] == grp]
            if len(grp_genes) == 0:
                continue
            mean_corr = grp_genes['mean_corr'].mean()
            end_idx = start_idx + len(grp_genes) - 1
            ax.hlines(y=mean_corr, xmin=start_idx-0.4, xmax=end_idx+0.4,
                      colors=color_map[grp], linestyles='dashed', linewidth=2, alpha=0.7)
            start_idx = end_idx + 1

    # Legend
    handles = [plt.Rectangle((0,0),1,1,color=color_map[grp]) for grp in group_order]
    ax.legend(handles, group_order, title=group_by.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig


def plot_gene_correlation_histogram(df, dataset_name, encoder_name,
                                    value_col="mean_corr", bins=30, figsize=(8, 5)):
    if df.empty or value_col not in df:
        fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
        return fig
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[value_col].dropna().astype(float), bins=bins)
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Number of genes")
    ax.set_title(_format_title("Gene Correlation Histogram", dataset_name, encoder_name))
    return fig


def plot_mean_std_ranked(
    df,
    gene_col="gene",
    mean_col="mean_corr",
    std_col="std_corr",
    top_n=None,
    figsize=(12, 4),
    dpi=150,
    xlabel=None,
    ylabel="Prediction correlation",
    title=None,
    max_labels=20,
    savepath=None,
):
    """
    Genes ranked by ``mean_col`` (descending) with ``std_col`` error bars.

    top_n: keep only the top-N genes (None = all).
    max_labels: with more than 40 genes, every k-th label is shown, k = len(genes) // max_labels.
    savepath: if given, saves <savepath>.pdf and <savepath>.png at 300 dpi.
    Returns (fig, ax).
    """
    missing = [c for c in (gene_col, mean_col, std_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    df_plot = (df[[gene_col, mean_col, std_col]]
               .dropna(subset=[mean_col])
               .sort_values(mean_col, ascending=False))
    if top_n is not None:
        df_plot = df_plot.iloc[:int(top_n)]

    genes = df_plot[gene_col].astype(str).tolist()
    x = np.arange(len(genes))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.errorbar(x, df_plot[mean_col].astype(float).values, yerr=df_plot[std_col].astype(float).values,
                fmt="o", ecolor="black", elinewidth=1, capsize=3, markersize=4)
    ax.set_xlim(-0.5, len(genes) - 0.5)
    ax.set_xticks(x)
    if len(genes) > 40:
        ticks = np.arange(0, len(genes), max(1, len(genes) // max_labels))
        ax.set_xticks(ticks)
        ax.set_xticklabels([genes[i] for i in ticks], rotation=90, fontsize=7)
    elif len(genes) > 8:
        ax.set_xticklabels(genes, rotation=45, fontsize=10, ha="right")
    else:
        ax.set_xticklabels(genes, rotation=0, fontsize=9)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if savepath:
        p = Path(savepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
        fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=300)
        print(f"Saved: {p.with_suffix('.pdf')} and {p.with_suffix('.png')}")

    plt.show()
    return fig, ax
