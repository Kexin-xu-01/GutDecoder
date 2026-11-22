"""
Opt-in matplotlib style for presentation figures. Nothing changes until apply_style() is called.

Sizes follow what the presentation notebooks set inline: 10-11 pt labels for oral slides,
12 pt everywhere for posters. Text stays editable in PDF/SVG (TrueType, not paths).
"""
from __future__ import annotations

import matplotlib as mpl

__all__ = ["PALETTES", "apply_style"]

PALETTES = {
    "dataset": "#8ecae6",          # violin fill used across presentation figures
    "literature_band": "brown",    # shaded literature ranges behind violins
    "sequential": "viridis",       # spatial pred/target maps
}

_BASE = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}

_SIZES = {
    "oral": {"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 11,
             "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9},
    "poster": {"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
               "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11},
}


def apply_style(kind: str = "oral") -> None:
    """Set matplotlib rcParams for 'oral' (slides) or 'poster' figures for this session."""
    if kind not in _SIZES:
        raise ValueError(f"kind must be one of {sorted(_SIZES)}, got {kind!r}")
    mpl.rcParams.update({**_BASE, **_SIZES[kind]})
