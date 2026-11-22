from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import hest
import pandas as pd
from loguru import logger
import scanpy as sc
import matplotlib.pyplot as plt
from hest import XeniumReader 
import dask
dask.config.set({'dataframe.query-planning': False})

#import cucim
#from cucim import CuImage

from hest.utils import read_xenium_alignment, align_xenium_df  

# keep original function
_orig_calc_qc = sc.pp.calculate_qc_metrics

try:
    # replace with a no-op
    sc.pp.calculate_qc_metrics = lambda *args, **kwargs: None

finally:
    # restore original function
    sc.pp.calculate_qc_metrics = _orig_calc_qc


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
    *,
    x: str | None = None,
    y: str | None = None,
    max_cells: int | None = None,
    seed: int = 0,
):
    """
    Quick scatterplot of spatial coordinates from .obsm["spatial"]
    (or from obs columns ``x``/``y`` when both are given).

    Parameters
    ----------
    adata : AnnData
        Must contain .obsm["spatial"] (n×2 array).
    color : str or None
        Column in adata.obs, or a gene in adata.var_names, to color by. If None, plot plain points.
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
    x, y : str or None
        obs columns holding coordinates, used instead of .obsm["spatial"] when both are given.
    max_cells : int or None
        Plot a random subsample of this many cells (reproducible via ``seed``).

    Returns
    -------
    Path or None : path of saved file if `save` was provided.
    """
    import matplotlib.pyplot as plt

    if (x is None) != (y is None):
        raise ValueError("give both x and y obs columns, or neither")
    if max_cells is not None and adata.n_obs > max_cells:
        idx = np.random.default_rng(seed).choice(adata.n_obs, max_cells, replace=False)
        adata = adata[np.sort(idx)]

    if x is not None:
        xs, ys = adata.obs[x].to_numpy(), adata.obs[y].to_numpy()
    else:
        coords = adata.obsm.get("spatial", None)
        if coords is None:
            raise ValueError("AnnData object has no .obsm['spatial'].")
        xs, ys = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    if color is None:
        sca = ax.scatter(xs, ys, s=point_size, alpha=alpha, c="k")
    else:
        if color in adata.obs.columns:
            vals = adata.obs[color]
        elif color in adata.var_names:
            col = adata[:, color].X
            vals = pd.Series(np.asarray(col.todense() if hasattr(col, "todense") else col).ravel())
        else:
            raise ValueError(f"{color!r} not found in adata.obs or adata.var_names")
        if vals.dtype.kind in "iufc":  # numeric
            sca = ax.scatter(xs, ys, s=point_size, alpha=alpha, c=vals, cmap=cmap)
            cbar = plt.colorbar(sca, ax=ax)
            cbar.set_label(color)
        else:
            cats = vals.astype("category")
            colors = plt.cm.tab20(np.linspace(0, 1, len(cats.cat.categories)))
            for col, ct in zip(colors, cats.cat.categories):
                m = cats == ct
                ax.scatter(xs[np.asarray(m)], ys[np.asarray(m)], s=point_size, alpha=alpha, c=[col], label=str(ct))
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
    label_key: str = "cell_type",
    title: Optional[str] = None,
):
    """
    Overlay spatial spots (colored by `spot_color_key`) with labelled cells
    (colored by `label_key`, e.g. 'cell_type' or 'niche').
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    outpath = save_dir / fname

    if "spatial" not in st_adata.obsm_keys():
        raise KeyError("'spatial' not found in st_adata.obsm")
    if "spatial" not in adata_labelled.obsm_keys():
        raise KeyError("'spatial' not found in adata_labelled.obsm")
    if spot_color_key not in st_adata.obs.columns:
        raise KeyError(f"'{spot_color_key}' not found in st_adata.obs")
    if label_key not in adata_labelled.obs.columns:
        raise KeyError(f"'{label_key}' not found in adata_labelled.obs")

    fig, ax = plt.subplots(figsize=(8, 8))

    # spots
    sca = ax.scatter(
        st_adata.obsm["spatial"][:, 0],
        st_adata.obsm["spatial"][:, 1],
        c=st_adata.obs[spot_color_key],
        s=8,
        alpha=0.6,
        cmap="viridis",
        label="spots",
    )

    # cells colored by requested label key
    labels = adata_labelled.obs[label_key].astype("category")
    categories = labels.cat.categories
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    for color, cat in zip(colors, categories):
        m = labels == cat
        ax.scatter(
            adata_labelled.obsm["spatial"][m, 0],
            adata_labelled.obsm["spatial"][m, 1],
            c=[color],
            s=2,
            alpha=0.6,
            label=str(cat),
        )

    ax.invert_yaxis()
    ax.set_title(title or f"Overlay: spots ({spot_color_key}) and cells ({label_key})")

    # legend + colorbar
    ax.legend(markerscale=4, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.35])
    cbar = fig.colorbar(sca, cax=cbar_ax)
    cbar.set_label(spot_color_key)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return outpath


def save_all(
    st,
    save_dir: Path | str,
    pyramidal: bool = True,
    plot_fname: str = "cell_plot.png",
    spatial_name: str = "",
    spatial_key: str = "total_counts",
    cell_label_key: str = "cell_type",
    plot_overlay: bool = True,
    **pl_kwargs,
) -> Path | None:
    """
    Save spatial plot, pyramid, and optional overlay figure to `save_dir`.

    Args:
        st: object with .adata and .save_spatial_plot method
        save_dir: output directory
        pyramidal: whether to save pyramidal data
        plot_fname: filename for overlay plot
        spatial_name: prefix for spatial plot filename
        spatial_key: feature to plot in spatial plot
        cell_label_key: which column in st.cell_adata.obs to color cells by
                        (e.g. 'cell_type' or 'niche')
        plot_overlay: whether to make the overlay plot
        **pl_kwargs: additional keyword arguments for sc.pl.spatial
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Delegate to st's own writers
    st.save_spatial_plot(save_dir, name=spatial_name, key=spatial_key, pl_kwargs=pl_kwargs)
    st.save(save_dir, pyramidal=pyramidal)

    # Optional overlay, only if cell_adata exists
    if plot_overlay and getattr(st, "cell_adata", None) is not None:
        out = plot_cell(
            st.adata,
            st.cell_adata,
            save_dir=save_dir,
            fname=plot_fname,
            label_key=cell_label_key,
        )
        return Path(out)

    return None


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
        OR to define a middle strip:
        - {'type':'strip', 'axis':'x'|'y', 'bounds': (start, end), 'units':'px'|'frac'}
          * For axis='y' bounds are (ymin_rel, ymax_rel) (0..1 in 'frac') or pixel coords in 'px'
          * For axis='x' bounds are (xmin_rel, xmax_rel) similarly.
          * If `bounds` is provided it takes precedence over side/size behavior.
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
            # New behaviour: if 'bounds' provided, it defines an arbitrary strip region
            # bounds = (start, end) in 'px' or 'frac' relative to bbox
            bounds = r.get('bounds', None)
            units = r.get('units', 'px')

            if bounds is not None:
                if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                    raise ValueError("strip 'bounds' must be a 2-tuple/list (start, end)")

                start, end = bounds
                # convert frac to pixels relative to bbox
                if units == 'frac':
                    # If both values between 0 and 1 we treat them as fractions of bbox
                    # For clarity: start=0 -> left/top; end=1 -> right/bottom
                    if r.get('axis') == 'x' or r.get('side') in ('left', 'right'):
                        sx_min = xmin + start * w
                        sx_max = xmin + end * w
                        cond = (sx >= min(sx_min, sx_max)) & (sx <= max(sx_min, sx_max))
                    else:
                        # default axis is 'y' if unspecified or if side is top/bottom
                        sy_min = ymin + start * h
                        sy_max = ymin + end * h
                        cond = (sy >= min(sy_min, sy_max)) & (sy <= max(sy_min, sy_max))
                else:
                    # units == 'px'
                    if r.get('axis') == 'x' or r.get('side') in ('left', 'right'):
                        sx_min = start
                        sx_max = end
                        cond = (sx >= min(sx_min, sx_max)) & (sx <= max(sx_min, sx_max))
                    else:
                        sy_min = start
                        sy_max = end
                        cond = (sy >= min(sy_min, sy_max)) & (sy <= max(sy_min, sy_max))

                exclude |= cond

            else:
                # legacy edge-strip behavior using 'side' + 'size'
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
            if ori == 'top':
                cx = (xmin + xmax) / 2.0 + offset
                top_left_x = cx - top_w / 2.0
                top_right_x = cx + top_w / 2.0
                bot_left_x = cx - bot_w / 2.0
                bot_right_x = cx + bot_w / 2.0

                poly_x = np.array([top_left_x, top_right_x, bot_right_x, bot_left_x])
                poly_y = np.array([ymin, ymin, ymin + height, ymin + height])

            elif ori == 'bottom':
                cx = (xmin + xmax) / 2.0 + offset
                top_left_x = cx - bot_w / 2.0
                top_right_x = cx + bot_w / 2.0
                bot_left_x = cx - top_w / 2.0
                bot_right_x = cx + top_w / 2.0

                poly_x = np.array([top_left_x, top_right_x, bot_right_x, bot_left_x])
                poly_y = np.array([ymax - height, ymax - height, ymax, ymax])

            elif ori == 'left':
                cy = (ymin + ymax) / 2.0 + offset
                top_left_y = cy - top_w / 2.0
                bottom_left_y = cy + top_w / 2.0
                top_right_y = cy - bot_w / 2.0
                bottom_right_y = cy + bot_w / 2.0

                poly_x = np.array([xmin, xmin, xmin + height, xmin + height])
                poly_y = np.array([top_left_y, bottom_left_y, bottom_right_y, top_right_y])

            elif ori == 'right':
                cy = (ymin + ymax) / 2.0 + offset
                top_left_y = cy - bot_w / 2.0
                bottom_left_y = cy + bot_w / 2.0
                top_right_y = cy - top_w / 2.0
                bottom_right_y = cy + top_w / 2.0

                poly_x = np.array([xmax - height, xmax - height, xmax, xmax])
                poly_y = np.array([top_left_y, bottom_left_y, bottom_right_y, top_right_y])

            else:
                raise ValueError("orientation must be one of: top, bottom, left, right")

            # Now check which points are inside polygon (requires your _points_in_polygon helper)
            inside_trap = _points_in_polygon(sx, sy, poly_x, poly_y)
            exclude |= inside_trap

        else:
            raise ValueError(f"Unknown rule type: {rtype}")

    return ~exclude  # keep mask

def rule_based_spot_exclusion(
    st,
    rules,
    img_key="downscaled_fullres",
    plot=True,
    figsize=(8, 6),
    show=False,
    save_path=None,
    verbose=True,
):
    """
    Apply rule-based exclusions to spots across the full WSI (no cell bbox restriction).

    Parameters
    ----------
    st : object
        Object containing spatial AnnData at `st.adata` and (optionally) a `st.wsi` with .width/.height.
        Must have st.adata.obsm["spatial"] present.
    rules : list of dict
        List of rule dictionaries. Each dict must contain:
            - 'type': currently only 'strip' supported
            - 'side': 'top' | 'bottom' | 'left' | 'right'
            - 'size': numeric (if units == 'px', size in pixels; if units == 'frac', fractional size 0-1)
            - 'units': 'px' | 'frac'  (default 'frac' if omitted)
        Example: [{'type':'strip','side':'top','size':0.09,'units':'frac'}]
        Rules are applied sequentially; a spot is kept only if it passes ALL rules (logical AND).
    img_key : str
        Image key used for plotting with sc.pl.spatial (default "downscaled_fullres").
    plot : bool
        If True, build and return a matplotlib Figure showing spots kept (uses sc.pl.spatial(..., return_fig=True)).
    figsize : tuple
        Figure size in inches.
    show : bool
        Passed to sc.pl.spatial show argument (if False the function will return the fig and not show it).
    save_path : str or Path or None
        If provided, the plotted figure will be saved to this path (requires plot=True).
    verbose : bool
        Print summaries.

    Returns
    -------
    final_keep : np.ndarray (bool)
        Boolean mask (len = number of spots) with True for spots kept after applying all rules.
    fig : matplotlib.figure.Figure or None
        If plot=True, returns the figure; otherwise returns None.

    Notes
    -----
    - Coordinate system assumptions: `st.adata.obsm["spatial"]` is Nx2 array of (x, y).
      The bounding box is assumed to be [xmin, xmax] x [ymin, ymax]. If `st.wsi` exists it uses
      width/height from there. Otherwise it uses min/max of the coordinates as a fallback.
    - Currently only 'strip' rules implemented. A 'strip' removes a band along one side:
        side = 'top'    -> band near ymin
        side = 'bottom' -> band near ymax
        side = 'left'   -> band near xmin
        side = 'right'  -> band near xmax
      For 'frac' units the size is fraction of the full height (for top/bottom) or width (for left/right).
    """
    # get spot coordinates
    xy = st.adata.obsm["spatial"]
    sx = xy[:, 0].astype(float)
    sy = xy[:, 1].astype(float)

    # determine bounding box: the WSI if loaded, else the spot coordinates
    if st.wsi is not None:
        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(st.wsi.width), float(st.wsi.height)
    else:
        xmin, xmax = float(sx.min()), float(sx.max())
        ymin, ymax = float(sy.min()), float(sy.max())
        # add a tiny padding so strips that reference fractions behave sensibly
        pad_x = (xmax - xmin) * 0.001 if (xmax > xmin) else 1.0
        pad_y = (ymax - ymin) * 0.001 if (ymax > ymin) else 1.0
        xmin -= pad_x; xmax += pad_x
        ymin -= pad_y; ymax += pad_y

    width = xmax - xmin
    height = ymax - ymin

    # start with everything kept
    final_keep = apply_spot_exclusions(sx, sy, (xmin, xmax, ymin, ymax), rules)

    if verbose:
        kept = int(np.count_nonzero(final_keep))
        total = len(final_keep)
        print(f"Spots kept after rules: {kept} / {total}")

    fig = None
    if plot:
        fig = sc.pl.spatial(
            st.adata[final_keep],
            img_key="downscaled_fullres",
            color="total_counts",
            title="In-tissue spots (after rule exclusions)",
            return_fig=True
        )
        if save_path is not None:
            fig.savefig(str(save_path), bbox_inches="tight")

    return final_keep, fig


# ----------------------------------------------------------------------------- feature padding

def pick_spatial_xy_columns(obs: pd.DataFrame) -> Tuple[str, str]:
    """Return the (x, y) centroid column names of a cell table; KeyError if no known pair exists."""
    for x_col, y_col in [("cell_centroid_x", "cell_centroid_y"), ("centroid_x", "centroid_y"),
                         ("cell_x", "cell_y"), ("x_centroid", "y_centroid"), ("x", "y"), ("X", "Y")]:
        if x_col in obs.columns and y_col in obs.columns:
            return x_col, y_col
    centroidish = [c for c in obs.columns if re.search(r"centroid|center", str(c), flags=re.IGNORECASE)]
    raise KeyError(f"Could not find centroid x/y columns in obs. Centroid-like columns: {centroidish[:50]}")


def collect_feature_union(rows: pd.DataFrame, h5ad_col: str, rename_map: Optional[Dict[str, str]] = None) -> list:
    """
    Sorted union of cell_type labels across the labelled h5ad files listed in ``rows[h5ad_col]``.
    Files are opened backed (no count matrices loaded). Missing files are skipped with a warning.
    """
    features, seen = set(), set()
    for value in rows[h5ad_col]:
        if pd.isna(value) or not str(value).strip():
            continue
        path = Path(str(value).strip()).resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            logger.warning(f"Feature union: labelled h5ad not found, its labels are excluded: {path}")
            continue
        labelled = sc.read_h5ad(path, backed="r")
        try:
            standardize_obs_columns(labelled, rename_map=rename_map)
            if "cell_type" not in labelled.obs:
                raise ValueError(f"Expected cell_type after annotation renaming: {path}")
            labels = labelled.obs["cell_type"]
            if not isinstance(labels, pd.Series):
                raise ValueError(f"Multiple annotation columns renamed to cell_type: {path}")
            features.update(labels.dropna().astype(str))
        finally:
            labelled.file.close()
    if not features:
        raise ValueError("No cell type/niche labels found for the selected samples")
    return sorted(features)


def pad_adata_to_var_union(adata: sc.AnnData, features: Iterable[str]) -> sc.AnnData:
    """
    Zero-fill and reorder the feature axis to exactly ``features``, preserving obs, obsm, obsp,
    uns, layers, varm, varp and raw. Raises if ``adata`` has features outside ``features``.
    A var column named ``_index`` (which h5ad writing rejects) is renamed to ``index_value``.
    """
    import anndata as ad
    import scipy.sparse as sp
    from copy import deepcopy

    target = pd.Index(list(map(str, features)))
    if not target.is_unique or not adata.var_names.is_unique:
        raise ValueError("Feature names must be unique for zero padding")
    unexpected = adata.var_names.difference(target)
    if len(unexpected):
        raise ValueError(f"Features absent from the shared feature list: {list(unexpected)}")

    var = adata.var.rename(columns={"_index": "index_value"}) if "_index" in adata.var.columns else adata.var
    if adata.var_names.equals(target):
        if var is adata.var:
            return adata
        out = adata.copy()
        out.var = var.copy()
        return out

    missing = target.difference(adata.var_names, sort=False)
    order = adata.var_names.append(missing).get_indexer(target)

    def pad_columns(matrix):
        if matrix is None:
            return None
        if sp.issparse(matrix):
            zeros = sp.csr_matrix((matrix.shape[0], len(missing)), dtype=matrix.dtype)
            return sp.hstack([matrix, zeros], format="csr")[:, order]
        zeros = np.zeros((matrix.shape[0], len(missing)), dtype=matrix.dtype)
        return np.concatenate([matrix, zeros], axis=1)[:, order]

    varm = {}
    for key, values in adata.varm.items():
        if isinstance(values, pd.DataFrame):
            varm[key] = values.reindex(target, fill_value=0)
        elif sp.issparse(values):
            varm[key] = pad_columns(values.T).T.tocsr()
        else:
            zeros = np.zeros((len(missing),) + values.shape[1:], dtype=values.dtype)
            varm[key] = np.concatenate([values, zeros], axis=0)[order]
    padded = ad.AnnData(
        X=pad_columns(adata.X), obs=adata.obs.copy(), var=var.reindex(target).copy(),
        obsm=deepcopy(dict(adata.obsm)), obsp=deepcopy(dict(adata.obsp)), uns=deepcopy(adata.uns), varm=varm,
        # recent AnnData versions expose X as layers[None]; it is supplied above
        layers={key: pad_columns(value) for key, value in adata.layers.items() if key is not None},
        varp={key: pad_columns(pad_columns(value).T).T for key, value in adata.varp.items()},
    )
    if adata.raw is not None:
        padded.raw = adata.raw.to_adata()
    return padded


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
    'fast_spatial',
    "rule_based_spot_exclusion",
    "pick_spatial_xy_columns",
    "collect_feature_union",
    "pad_adata_to_var_union",
]
