"""
Spatial prediction-vs-target figures with a shared colour scale.

Used by the presentation notebooks (JupyterFolder/04_presentation/*/01_plot.ipynb).
Every function reads per-gene values from two AnnData layers (``pred`` and ``target``
by default), puts both panels on one colour scale, and annotates the prediction panel
with its Pearson r against the target.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scanpy as sc
from loguru import logger
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

__all__ = [
    "layer_values",
    "shared_vmin_vmax",
    "show_pred_target",
    "save_one_sample_shared_cbar",
    "save_two_samples_shared_cbar",
]


# building blocks

def layer_values(adata, layer: str, gene: str) -> np.ndarray:
    """1-D array of ``adata.layers[layer]`` for ``gene``; KeyError if the gene is absent."""
    var_names = list(map(str, adata.var_names))
    if str(gene) not in var_names:
        raise KeyError(f"gene {gene!r} not in adata.var_names")
    return np.asarray(adata.layers[layer][:, var_names.index(str(gene))]).ravel()


def shared_vmin_vmax(arrays: Iterable[np.ndarray], vmin=None, vmax=None, label: str = "") -> tuple[float, float]:
    """
    Colour-scale limits shared by ``arrays``: explicit ``vmin``/``vmax`` win, otherwise the
    finite min/max. With no finite values the scale falls back to (0, 1) and a warning is logged.
    A degenerate range (vmax <= vmin) is widened slightly so the colour map stays defined.
    """
    finite = np.concatenate([np.asarray(a)[np.isfinite(a)] for a in arrays])
    if finite.size == 0 and (vmin is None or vmax is None):
        logger.warning(f"{label or 'values'}: no finite values; using colour scale 0-1")
    vmin_ = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    vmax_ = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else vmin_ + 1.0)
    if vmax_ <= vmin_:
        vmax_ = vmin_ + max(1e-6, abs(vmin_) * 1e-3)
    return vmin_, vmax_


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    return pearsonr(pred[mask], target[mask])[0] if mask.sum() >= 2 else np.nan


def _plot_pair(adata, pred, target, ax_target, ax_pred, *, img_key, cmap, size, vmin, vmax):
    """Draw target and pred spatial panels for one sample on a fixed colour scale."""
    adata.obs["_pred"] = pred
    adata.obs["_target"] = target
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for col, ax in (("_target", ax_target), ("_pred", ax_pred)):
                sc.pl.spatial(adata, img_key=img_key, color=col, ax=ax, show=False,
                              cmap=cmap, size=size, vmin=vmin, vmax=vmax, colorbar_loc=None)
    finally:
        adata.obs.drop(columns=["_pred", "_target"], inplace=True)


def _save_pdf_png(fig, out_pdf, dpi: int, **kwargs) -> None:
    out_pdf = Path(out_pdf)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", **kwargs)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=dpi, bbox_inches="tight", **kwargs)


# figures

def show_pred_target(
    adata,
    genes: Sequence[str],
    img_key: str = "downscaled_fullres",
    pred_layer: str = "pred",
    target_layer: str = "target",
    size: float = 1.5,
    fig_size=(8, 3),
    cmap=None,
    vmin=None,
    vmax=None,
) -> dict[str, tuple[float, float]]:
    """
    Show target | pred for each gene of ONE sample inline, with one shared colourbar per gene.
    Genes missing from ``adata.var_names`` are skipped with a warning.
    Returns {gene: (vmin, vmax)} for the genes plotted.
    """
    var_names = set(map(str, adata.var_names))
    scales = {}
    for gene in genes:
        if str(gene) not in var_names:
            logger.warning(f"show_pred_target: {gene} not in var_names, skipped")
            continue
        pred = layer_values(adata, pred_layer, gene)
        target = layer_values(adata, target_layer, gene)
        vmin_, vmax_ = shared_vmin_vmax([pred, target], vmin, vmax, label=str(gene))

        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        _plot_pair(adata, pred, target, axes[0], axes[1],
                   img_key=img_key, cmap=cmap, size=size, vmin=vmin_, vmax=vmax_)
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_, vmax=vmax_), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.1)

        axes[1].set_title(f"pred: {gene} (r={_pearson(pred, target):.2f})")
        axes[0].set_title(f"target: {gene}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        scales[str(gene)] = (vmin_, vmax_)
    return scales


def save_one_sample_shared_cbar(
    adata,
    gene: str,
    out_pdf,
    label: str | None = None,
    gene_label: str | None = None,
    img_key: str = "downscaled_fullres",
    pred_layer: str = "pred",
    target_layer: str = "target",
    cmap: str = "viridis",
    size: float = 1.5,
    vmin=None,
    vmax=None,
    fig_size=(6, 2.2),
    dpi: int = 300,
) -> tuple[float, float]:
    """
    ONE sample: target | pred with a shared colourbar (integer ticks). ``label`` is written
    vertically on the left, ``gene_label`` (default: gene) as the title.
    Saves ``out_pdf`` and a PNG with the same basename. Returns (vmin, vmax).
    """
    pred = layer_values(adata, pred_layer, gene)
    target = layer_values(adata, target_layer, gene)
    vmin_, vmax_ = shared_vmin_vmax([pred, target], vmin, vmax, label=str(gene))

    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    _plot_pair(adata, pred, target, axes[0], axes[1],
               img_key=img_key, cmap=cmap, size=size, vmin=vmin_, vmax=vmax_)
    axes[0].set_title("target", fontsize=9)
    axes[1].set_title(f"pred (r={_pearson(pred, target):.2f})", fontsize=9)
    for ax in axes:
        ax.axis("off")

    fig.subplots_adjust(left=0.10, right=0.88, top=0.85, bottom=0.15)
    cax = fig.add_axes([0.90, 0.2, 0.01, 0.60])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin_, vmax=vmax_), cmap=cmap)
    sm.set_array(np.linspace(vmin_, vmax_, 256))
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8)
    cb.locator = mticker.MaxNLocator(nbins=3, integer=True)
    cb.formatter = mticker.StrMethodFormatter("{x:.0f}")
    cb.update_ticks()

    if label is not None:
        fig.canvas.draw()
        bbox = axes[0].get_position()
        fig.text(0.04, bbox.y0 + bbox.height / 2, label, rotation=90,
                 va="center", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle(gene_label or gene, fontsize=10, fontweight="bold", y=0.97, fontstyle="italic")

    _save_pdf_png(fig, out_pdf, dpi, pad_inches=0.05)
    plt.close(fig)
    return vmin_, vmax_


def save_two_samples_shared_cbar(
    adata1,
    adata2,
    gene: str,
    out_pdf,
    layout: str = "rows",
    labels: Sequence[str] | None = None,
    img_key: str = "downscaled_fullres",
    pred_layer: str = "pred",
    target_layer: str = "target",
    cmap: str = "viridis",
    size: float = 1.5,
    vmin=None,
    vmax=None,
    fig_width: float = 8,
    per_row_height: float = 2.0,
    row_heights: Sequence[float] | None = None,
    col_widths: Sequence[float] | None = None,
    label_x: float = 0.03,
    cbar_floor: float | None = 0.0,
    dpi: int = 300,
) -> tuple[float, float]:
    """
    TWO samples with one shared colourbar on the right.

    layout="rows": one sample per row (target | pred), ``labels`` drawn vertically on the left,
        gene as the title. Row heights (inches) from ``row_heights`` or ``per_row_height``.
    layout="cols": one sample per column (target on top, pred below), gene as the colourbar
        label. Column widths (inches) from ``col_widths`` or half of ``fig_width`` each.
        ``labels`` are not drawn in this layout.

    The colourbar starts at max(cbar_floor, vmin); pass cbar_floor=None to show the full range.
    Saves ``out_pdf`` and a PNG with the same basename. Returns (vmin, vmax).
    """
    if layout not in ("rows", "cols"):
        raise ValueError(f"layout must be 'rows' or 'cols', got {layout!r}")
    if labels is not None and len(labels) != 2:
        raise ValueError("`labels` must have length 2 (one per sample).")
    if layout == "cols" and labels is not None:
        logger.warning("save_two_samples_shared_cbar: labels are not drawn with layout='cols'")

    adatas = [adata1, adata2]
    values = [(layer_values(a, pred_layer, gene), layer_values(a, target_layer, gene)) for a in adatas]
    vmin_, vmax_ = shared_vmin_vmax([v for pair in values for v in pair], vmin, vmax, label=str(gene))

    if layout == "rows":
        heights = list(row_heights) if row_heights is not None else [per_row_height] * 2
        widths = [fig_width / 2] * 2
    else:
        heights = [per_row_height] * 2
        widths = list(col_widths) if col_widths is not None else [fig_width / 2] * 2
    if len(heights) != 2 or len(widths) != 2:
        raise ValueError("`row_heights` and `col_widths` must have length 2 (inches).")

    fig = plt.figure(figsize=(sum(widths), sum(heights) + 0.4))
    gs = GridSpec(nrows=2, ncols=2, height_ratios=heights, width_ratios=widths, figure=fig)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)])

    for i, (a, (pred, target)) in enumerate(zip(adatas, values)):
        # rows: sample i on row i (target, pred); cols: sample i in column i (target over pred)
        ax_t, ax_p = (axes[i, 0], axes[i, 1]) if layout == "rows" else (axes[0, i], axes[1, i])
        _plot_pair(a, pred, target, ax_t, ax_p,
                   img_key=img_key, cmap=cmap, size=size, vmin=vmin_, vmax=vmax_)
        ax_t.set_title("target", fontsize=10)
        ax_p.set_title(f"pred (r={_pearson(pred, target):.2f})", fontsize=10)
        ax_t.axis("off")
        ax_p.axis("off")

    plt.subplots_adjust(left=0.07, right=0.90, top=0.97, bottom=0.03, hspace=0.05, wspace=0.02)
    if layout == "rows":
        fig.suptitle(gene, fontsize=10, fontweight="bold", y=1.05, fontstyle="italic")
    fig.canvas.draw()  # final axis positions for the labels below

    if layout == "rows" and labels is not None:
        for i, lab in enumerate(labels):
            bbox = axes[i, 0].get_position()
            fig.text(label_x, bbox.y0 + bbox.height / 2, lab, rotation=90,
                     va="center", ha="center", fontsize=10, fontweight="bold")
        plt.subplots_adjust(left=0.03, right=0.90, top=0.95, bottom=0.05, hspace=0.15, wspace=0.05)

    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cb_vmin = vmin_ if cbar_floor is None else max(cbar_floor, vmin_)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=cb_vmin, vmax=vmax_), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8)
    if layout == "cols":
        cb.set_label(gene, fontsize=9, fontstyle="italic", labelpad=6)

    plt.subplots_adjust(left=0.05, right=0.90, hspace=0.15, wspace=0.05)
    _save_pdf_png(fig, out_pdf, dpi)
    plt.close(fig)
    return vmin_, vmax_
