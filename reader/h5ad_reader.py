
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import hest
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from hest import XeniumReader 
import dask
dask.config.set({'dataframe.query-planning': False})
from pathlib import Path

#import cucim
#from cucim import CuImage

from hest.utils import read_xenium_alignment, align_xenium_df  


def coord_range(spatial: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) for an n×2 spatial array."""
    if spatial is None or spatial.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    x = spatial[:, 0]
    y = spatial[:, 1]
    return float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))


def get_xy_from_adata(adata: sc.AnnData) -> Tuple[np.ndarray, np.ndarray]:
    """Get x,y pixel coords preferring adata.obsm['spatial'] if present."""
    if "spatial" in adata.obsm_keys():
        xy = adata.obsm["spatial"]
        return np.asarray(xy[:, 0]), np.asarray(xy[:, 1])
    # fallback to common Scanpy column names
    return adata.obs["pxl_col_in_fullres"].to_numpy(), adata.obs["pxl_row_in_fullres"].to_numpy()


def subset_adata_to_wsi(adata: sc.AnnData, w: int, h: int) -> sc.AnnData:
    """Keep only cells/spots whose coordinates lie within [0,w)×[0,h)."""
    x, y = get_xy_from_adata(adata)
    mask = (x >= 0) & (y >= 0) & (x < w) & (y < h)
    return adata[mask].copy()


def remove_codeword_features(adata: sc.AnnData, pattern: str = "Codeword") -> sc.AnnData:
    """Drop features whose var_names contain the given pattern (default 'Codeword')."""
    keep = ~adata.var_names.str.contains(pattern)
    return adata[:, keep].copy()


def load_xenium_dataset(
    exp_stem: Path,
    img_path: Path,
    alignment_csv: Path,
    use_dask: bool = True,
    feature_matrix_name: str = "cell_feature_matrix.h5",
    transcripts_name: str = "transcripts.parquet",
    cells_name: str = "cells.parquet",
    nucleus_bound_name: str = "nucleus_boundaries.parquet",
    cell_bound_name: str = "cell_boundaries.parquet",
    dapi_relpath: str = "morphology_focus/morphology_focus_0000.ome.tif",
    spot_size_um=100
):
    """Load Xenium data via XeniumReader with the given experiment stem and asset names.

    Returns
    -------
    st : object
        The object returned by XeniumReader().read(...). Expected to expose:
        - .adata (spots)
        - .wsi.width / .wsi.height
        - .meta (dict with 'pixel_size' at least)
    """
    if XeniumReader is None:
        raise ImportError("XeniumReader not found; ensure 'hest' is installed and import path is correct.")

    exp_stem = Path(exp_stem)
    st = XeniumReader().read(
        img_path=str(img_path),
        experiment_path=exp_stem / "experiment.xenium",
        alignment_file_path=str(alignment_csv),
        feature_matrix_path=exp_stem / feature_matrix_name,
        transcripts_path=exp_stem / transcripts_name,
        cells_path=exp_stem / cells_name,
        nucleus_bound_path=exp_stem / nucleus_bound_name,
        cell_bound_path=exp_stem / cell_bound_name,
        dapi_path=exp_stem / dapi_relpath,
        use_dask=use_dask,
        spot_size_um=spot_size_um
    )
    return st


def align_labelled_to_he(
    adata_labelled: sc.AnnData,
    alignment_csv: Path,
    pixel_size_um: float,
    x_key: str = "cell_centroid_x",
    y_key: str = "cell_centroid_y",
    out_x: str = "he_x",
    out_y: str = "he_y",
    to_dapi: bool = False,
) -> sc.AnnData:
    """Apply the 3×3 affine transform to labelled cells and store H&E pixel coords.

    Adds columns out_x/out_y to .obs and writes an n×2 array to .obsm['spatial'].
    """
    A = read_xenium_alignment(str(alignment_csv))
    df = adata_labelled.obs[[x_key, y_key]].copy()
    df = align_xenium_df(
        df,
        alignment_matrix=A,
        pixel_size_morph=pixel_size_um,
        x_key=x_key,
        y_key=y_key,
        to_dapi=to_dapi,
        x_key_dist=out_x,
        y_key_dist=out_y,
    )
    adata_labelled.obs[[out_x, out_y]] = df[[out_x, out_y]]
    adata_labelled.obsm["spatial"] = df[[out_x, out_y]].to_numpy()
    return adata_labelled


def standardize_obs_columns(
    adata_labelled: sc.AnnData,
    rename_map: Optional[Dict[str, str]] = None,
    segmentation_method: Optional[str] = "segger",
) -> sc.AnnData:
    """Optionally rename common columns and set a segmentation method field."""
    default_map = {
        "cell_centroid_x": "x_centroid",
        "cell_centroid_y": "y_centroid",
        "index_safe": "cell_id",
        "predicted.id": "cell_type",
        "transcripts": "transcript_counts",
        "predicted.id.score": "cell_type_prob",
    }
    if rename_map:
        default_map.update(rename_map)
    existing = {k: v for k, v in default_map.items() if k in adata_labelled.obs.columns}
    if existing:
        adata_labelled.obs.rename(columns=existing, inplace=True)
    if segmentation_method is not None:
        adata_labelled.obs["segmentation_method"] = segmentation_method
    return adata_labelled


def update_st_with_filtered_and_labelled(
    st,
    adata_labelled: Optional[sc.AnnData] = None,
    drop_codeword: bool = True,
) -> None:
    """Filter st.adata to its WSI extent, optionally drop 'Codeword' features, 
    then attach labelled cells if provided."""
    W = st.wsi.width
    H = st.wsi.height

    adata_piece = subset_adata_to_wsi(st.adata, W, H)
    if drop_codeword:
        adata_piece = remove_codeword_features(adata_piece)

    st.adata = adata_piece

    if adata_labelled is not None:
        st.cell_adata = adata_labelled


def refresh_meta_counts(st) -> Dict[str, float]:
    """Update st.meta summary stats from st.cell_adata and return the new values."""
    meta = st.meta
    old_vals = {
        "num_cells": meta.get("num_cells"),
        "cells_under_tissue": meta.get("cells_under_tissue"),
        "transcripts_per_cell": meta.get("transcripts_per_cell"),
        "total_cell_area": meta.get("total_cell_area"),
    }
    new_vals = {}
    new_vals["num_cells"] = st.cell_adata.n_obs
    new_vals["cells_under_tissue"] = st.cell_adata.n_obs

    # Robust means with fallbacks
    if "transcript_counts" in st.cell_adata.obs.columns:
        new_vals["transcripts_per_cell"] = float(st.cell_adata.obs["transcript_counts"].mean())
    else:
        new_vals["transcripts_per_cell"] = float("nan")

    if "cell_area" in st.cell_adata.obs.columns:
        new_vals["total_cell_area"] = float(st.cell_adata.obs["cell_area"].sum())
    else:
        new_vals["total_cell_area"] = float("nan")

    # Commit
    meta.update(new_vals)
    st.meta = meta
    return {"old": old_vals, "new": new_vals}

def fast_spatial(
    adata: sc.AnnData,
    color: str | None = None,
    save: Path | str | None = None,
    figsize: tuple[int, int] = (6, 6),
    point_size: float = 4.0,
    alpha: float = 0.6,
    cmap: str = "viridis",
    invert_y: bool = True,
):
    """
    Quick scatterplot of spatial coordinates from .obsm["spatial"].

    Parameters
    ----------
    adata : AnnData
        Must contain .obsm["spatial"] (n×2 array).
    color : str or None
        Column in adata.obs to color by. If None, plot plain points.
    save : path-like or None
        If provided, save the figure to this path.
    figsize : tuple
        Figure size in inches.
    point_size : float
        Marker size.
    alpha : float
        Marker transparency.
    cmap : str
        Colormap name if coloring by a continuous variable.
    invert_y : bool
        Invert y-axis to match microscopy conventions.

    Returns
    -------
    Path or None : path of saved file if `save` was provided.
    """
    import matplotlib.pyplot as plt

    coords = adata.obsm.get("spatial", None)
    if coords is None:
        raise ValueError("AnnData object has no .obsm['spatial'].")

    x, y = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    if color is None:
        sca = ax.scatter(x, y, s=point_size, alpha=alpha, c="k")
    else:
        vals = adata.obs[color]
        if vals.dtype.kind in "iufc":  # numeric
            sca = ax.scatter(x, y, s=point_size, alpha=alpha, c=vals, cmap=cmap)
            cbar = plt.colorbar(sca, ax=ax)
            cbar.set_label(color)
        else:
            cats = vals.astype("category")
            colors = plt.cm.tab20(np.linspace(0, 1, len(cats.cat.categories)))
            for col, ct in zip(colors, cats.cat.categories):
                m = cats == ct
                ax.scatter(x[m], y[m], s=point_size, alpha=alpha, c=[col], label=str(ct))
            ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")

    if invert_y:
        ax.invert_yaxis()
    ax.set_title(f"fast_spatial: {color if color else 'points'}")

    if save:
        save_path = Path(save)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None

def plot_cell(
    st_adata: sc.AnnData,
    adata_labelled: sc.AnnData,
    save_dir: Path | str,
    fname: str = "cell_plot.png",
    spot_color_key: str = "log1p_total_counts",
    cell_type_key: str = "cell_type",
):
    """Overlay spatial spots (colored by `spot_color_key`) with labelled cells (colored by `cell_type_key`)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    outpath = save_dir / fname

    fig, ax = plt.subplots(figsize=(8, 8))

    # spots
    if spot_color_key not in st_adata.obs.columns:
        raise KeyError(f"'{spot_color_key}' not found in st_adata.obs")
    sca = ax.scatter(
        st_adata.obsm["spatial"][:, 0],
        st_adata.obsm["spatial"][:, 1],
        c=st_adata.obs[spot_color_key],
        s=8, alpha=0.6, cmap="viridis", label="spots",
    )

    # cells colored by cell type
    if cell_type_key not in adata_labelled.obs.columns:
        raise KeyError(f"'{cell_type_key}' not found in adata_labelled.obs")
    cell_types = adata_labelled.obs[cell_type_key].astype("category")
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types.cat.categories)))
    for color, ct in zip(colors, cell_types.cat.categories):
        m = cell_types == ct
        ax.scatter(
            adata_labelled.obsm["spatial"][m, 0],
            adata_labelled.obsm["spatial"][m, 1],
            c=[color], s=2, alpha=0.6, label=str(ct)
        )

    ax.invert_yaxis()
    ax.set_title("Overlay: spots (counts) and cells (labels)")

    # legend + colorbar
    ax.legend(markerscale=4, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.35])
    cbar = fig.colorbar(sca, cax=cbar_ax)
    cbar.set_label(spot_color_key)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.show()
    return outpath

def save_all(
    st,
    save_dir: Path | str,
    pyramidal: bool = True,
    plot_fname: str = "cell_plot.png",
    spatial_name: str = "",
    spatial_key: str = "total_counts",
    **pl_kwargs,
) -> Path:
    """
    Save spatial plot, pyramid, and overlay figure to `save_dir` and return the overlay path.
    
    Args:
        st: object with .adata and .save_spatial_plot method
        save_dir (Path | str): Directory to save output
        pyramidal (bool): Whether to save pyramidal data
        plot_fname (str): Filename for overlay plot
        spatial_name (str): Prefix for spatial plot filename
        spatial_key (str): Feature to plot in spatial plot
        **pl_kwargs: Additional keyword arguments for sc.pl.spatial
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Delegate to st's own writers
    st.save_spatial_plot(save_dir, name=spatial_name, key=spatial_key, pl_kwargs=pl_kwargs)
    st.save(save_dir, pyramidal=pyramidal)

    # Our overlay
    out = plot_cell(st.adata, st.cell_adata, save_dir=save_dir, fname=plot_fname)
    return Path(out)


# ---------------- rule-based exclusions ----------------
def _points_in_polygon(px, py, poly_x, poly_y):
    """
    Vectorized point-in-polygon test (ray-casting).
    px, py : arrays of point coords
    poly_x, poly_y : polygon vertex coords (1D arrays, closed polygon assumed or will be closed)
    Returns boolean array of same shape as px indicating points inside polygon.
    """
    # Ensure polygon is closed
    if poly_x[0] != poly_x[-1] or poly_y[0] != poly_y[-1]:
        poly_x = np.concatenate([poly_x, poly_x[:1]])
        poly_y = np.concatenate([poly_y, poly_y[:1]])

    n_vert = len(poly_x) - 1  # last is duplicate of first
    inside = np.zeros_like(px, dtype=bool)

    # Ray casting algorithm
    for i in range(n_vert):
        x_i, y_i = poly_x[i], poly_y[i]
        x_j, y_j = poly_x[i+1], poly_y[i+1]

        # Check if edge crosses the horizontal ray to the right of the point
        cond1 = ((y_i > py) != (y_j > py))
        # compute x coordinate of intersection of edge with horizontal line at py
        # avoid division by zero
        denom = (y_j - y_i)
        # safe compute intersection x
        inter_x = x_i + (py - y_i) * (x_j - x_i) / (denom + 1e-20)

        cond2 = px < inter_x
        inside ^= (cond1 & cond2)

    return inside


def apply_spot_exclusions(sx, sy, bbox, rules):
    """
    Exclude spots by region rules. Returns a boolean mask of spots to KEEP.

    Parameters
    ----------
    sx, sy : 1D arrays of spot coordinates (pixels)
    bbox   : (xmin, xmax, ymin, ymax) in pixels
    rules  : list of dicts. Supported types:
      - {'type':'corner', 'corner':'top-left|top-right|bottom-left|bottom-right',
         'width': <float>, 'height': <float>, 'units':'px'|'frac'}
      - {'type':'strip', 'side':'top|bottom|left|right', 'size': <float>, 'units':'px'|'frac'}
      - {'type':'rect', 'xmin': <float>, 'xmax': <float>, 'ymin': <float>, 'ymax': <float>, 'units':'px'|'frac'}
        * For 'rect' in 'frac' units, values are in [0,1] relative to the bbox (0=left/top, 1=right/bottom).
      - {'type':'trapezoid', 'orientation':'top|bottom|left|right',
         'top_width': <float>, 'bottom_width': <float>, 'height': <float>,
         'units':'px'|'frac', 'center_offset': <float (px or frac)>}
        * 'orientation' describes which bbox edge the trapezoid base sits on:
           - 'top' : base along top edge y = ymin and trapezoid points downward
           - 'bottom': base along bottom edge y = ymax and trapezoid points upward
           - 'left' : base along left edge x = xmin and trapezoid points rightward
           - 'right': base along right edge x = xmax and trapezoid points leftward
        * widths/heights are in px or fraction of bbox width/height depending on 'units'.
        * 'center_offset' (optional, default 0) shifts the trapezoid along the edge axis (positive moves right for top/bottom, down for left/right).
    """
    xmin, xmax, ymin, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin

    exclude = np.zeros_like(sx, dtype=bool)

    for r in rules:
        rtype = r.get('type')

        if rtype == 'corner':
            units = r.get('units', 'px')
            corner = r['corner']
            ww = r['width']
            hh = r['height']
            if units == 'frac':
                ww = ww * w
                hh = hh * h

            if corner == 'top-left':
                cond = (sx < xmin + ww) & (sy < ymin + hh)
            elif corner == 'top-right':
                cond = (sx > xmax - ww) & (sy < ymin + hh)
            elif corner == 'bottom-left':
                cond = (sx < xmin + ww) & (sy > ymax - hh)
            elif corner == 'bottom-right':
                cond = (sx > xmax - ww) & (sy > ymax - hh)
            else:
                raise ValueError("corner must be one of: top-left, top-right, bottom-left, bottom-right")
            exclude |= cond

        elif rtype == 'strip':
            units = r.get('units', 'px')
            side = r['side']
            ss = r['size']
            if units == 'frac':
                ss = ss * (h if side in ('top','bottom') else w)

            if side == 'top':
                cond = sy < ymin + ss
            elif side == 'bottom':
                cond = sy > ymax - ss
            elif side == 'left':
                cond = sx < xmin + ss
            elif side == 'right':
                cond = sx > xmax - ss
            else:
                raise ValueError("side must be one of: top, bottom, left, right")
            exclude |= cond

        elif rtype == 'rect':
            units = r.get('units', 'px')
            if units == 'frac':
                rxmin = xmin + r['xmin'] * w
                rxmax = xmin + r['xmax'] * w
                rymin = ymin + r['ymin'] * h
                rymax = ymin + r['ymax'] * h
            else:
                rxmin, rxmax = r['xmin'], r['xmax']
                rymin, rymax = r['ymin'], r['ymax']
            cond = (sx >= rxmin) & (sx <= rxmax) & (sy >= rymin) & (sy <= rymax)
            exclude |= cond

        elif rtype == 'trapezoid':
            # Parameters & defaults
            units = r.get('units', 'px')
            ori = r.get('orientation', 'top')
            top_w = r.get('top_width')
            bot_w = r.get('bottom_width')
            height = r.get('height')
            offset = r.get('center_offset', 0.0)

            if top_w is None or bot_w is None or height is None:
                raise ValueError("trapezoid requires 'top_width', 'bottom_width', and 'height'")

            # Convert frac to px if needed
            if units == 'frac':
                if ori in ('top', 'bottom'):
                    top_w = top_w * w
                    bot_w = bot_w * w
                    height = height * h
                    offset = offset * w
                else:
                    top_w = top_w * h  # for left/right, interpret widths along vertical axis
                    bot_w = bot_w * h
                    height = height * w
                    offset = offset * h

            # Build polygon vertices depending on orientation.
            # We'll define vertices in (x,y) order (clockwise or counterclockwise).
            if ori == 'top':
                # Base along top edge at y = ymin, trapezoid points downward
                cx = (xmin + xmax) / 2.0 + offset  # center x of base, with offset
                top_left_x = cx - top_w / 2.0
                top_right_x = cx + top_w / 2.0
                bottom_half = height  # distance from top edge downward
                # bottom center aligned with cx
                bot_left_x = cx - bot_w / 2.0
                bot_right_x = cx + bot_w / 2.0

                poly_x = np.array([top_left_x, top_right_x, bot_right_x, bot_left_x])
                poly_y = np.array([ymin, ymin, ymin + bottom_half, ymin + bottom_half])

            elif ori == 'bottom':
                cx = (xmin + xmax) / 2.0 + offset
                top_left_x = cx - bot_w / 2.0
                top_right_x = cx + bot_w / 2.0
                # trapezoid points upward; base along bottom at y = ymax
                bot_left_x = cx - top_w / 2.0
                bot_right_x = cx + top_w / 2.0

                poly_x = np.array([top_left_x, top_right_x, bot_right_x, bot_left_x])
                poly_y = np.array([ymax - height, ymax - height, ymax, ymax])

            elif ori == 'left':
                cy = (ymin + ymax) / 2.0 + offset
                top_left_y = cy - top_w / 2.0
                bottom_left_y = cy + top_w / 2.0
                # trapezoid points rightward; base along left edge x = xmin
                top_right_y = cy - bot_w / 2.0
                bottom_right_y = cy + bot_w / 2.0

                poly_x = np.array([xmin, xmin, xmin + height, xmin + height])
                poly_y = np.array([top_left_y, bottom_left_y, bottom_right_y, top_right_y])

            elif ori == 'right':
                cy = (ymin + ymax) / 2.0 + offset
                top_left_y = cy - bot_w / 2.0
                bottom_left_y = cy + bot_w / 2.0
                # trapezoid points leftward; base along right edge x = xmax
                top_right_y = cy - top_w / 2.0
                bottom_right_y = cy + top_w / 2.0

                poly_x = np.array([xmax - height, xmax - height, xmax, xmax])
                poly_y = np.array([top_left_y, bottom_left_y, bottom_right_y, top_right_y])

            else:
                raise ValueError("orientation must be one of: top, bottom, left, right")

            # Now check which points are inside polygon
            inside_trap = _points_in_polygon(sx, sy, poly_x, poly_y)
            exclude |= inside_trap

        else:
            raise ValueError(f"Unknown rule type: {rtype}")

    return ~exclude  # keep mask



__all__ = [
    "coord_range",
    "get_xy_from_adata",
    "subset_adata_to_wsi",
    "remove_codeword_features",
    "load_xenium_dataset",
    "align_labelled_to_he",
    "standardize_obs_columns",
    "update_st_with_filtered_and_labelled",
    "refresh_meta_counts",
    "plot_cell",
    "save_all",
    "apply_spot_exclusions",
    'fast_spatial'
]
