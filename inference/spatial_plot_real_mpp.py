import os
import argparse
from typing import Optional, List, Tuple
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import scanpy as sc

from hest.utils import load_wsi, register_downscale_img
from loguru import logger


def _safe_min_value(x):
    """Return a scalar min for dense/sparse/lazy arrays when possible."""
    try:
        v = x.min()
        if hasattr(v, "compute"):
            v = v.compute()
        if hasattr(v, "item"):
            v = v.item()
        return v
    except Exception as e:
        logger.warning(f"Could not compute min value for adata.X: {e}")
        return None



def clip_negative_values_to_zero(adata, sample_name: str = ""):
    """
    Convert negative values in adata.X to 0 in-place before QC metrics.

    This is useful when adata.X contains model predictions / transformed values
    with small negative values, because scanpy QC uses log1p internally and
    negative values can trigger invalid-value warnings.
    """
    try:
        from scipy import sparse
    except Exception:
        sparse = None

    x = adata.X

    if sparse is not None and sparse.issparse(x):
        neg_mask = x.data < 0
        n_negative = int(np.sum(neg_mask))
        if n_negative > 0:
            min_before = float(np.min(x.data))
            x = x.copy()
            x.data[neg_mask] = 0
            x.eliminate_zeros()
            adata.X = x
            logger.warning(
                f"Converted {n_negative} negative sparse values to 0 in adata.X "
                f"for {sample_name}; min_before={min_before}"
            )
        else:
            logger.info(f"No negative sparse values found in adata.X for {sample_name}")
        return

    try:
        arr = np.asarray(x)
        neg_mask = arr < 0
        n_negative = int(np.sum(neg_mask))
        if n_negative > 0:
            min_before = float(np.nanmin(arr))
            # Preserve ndarray assignment when possible.
            if isinstance(x, np.ndarray):
                x[neg_mask] = 0
                adata.X = x
            else:
                arr = arr.copy()
                arr[neg_mask] = 0
                adata.X = arr
            logger.warning(
                f"Converted {n_negative} negative dense values to 0 in adata.X "
                f"for {sample_name}; min_before={min_before}"
            )
        else:
            logger.info(f"No negative dense values found in adata.X for {sample_name}")
    except Exception as e:
        logger.warning(f"Could not clip negative values in adata.X for {sample_name}: {e}")



def _as_float(value):
    """Best-effort conversion of metadata values such as '0.221' or '0.221 um' to float."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    try:
        import re
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
        return float(m.group(0)) if m else None
    except Exception:
        return None


def _read_mpp_from_openslide(wsi_path: str):
    """Read MPP from slide metadata using OpenSlide, when available."""
    try:
        import openslide
    except Exception as e:
        logger.debug(f"OpenSlide is not available for MPP read: {e}")
        return None

    try:
        slide = openslide.OpenSlide(wsi_path)
        props = dict(slide.properties)
        slide.close()
    except Exception as e:
        logger.debug(f"Could not open slide with OpenSlide for MPP read: {wsi_path}; {e}")
        return None

    candidates_x = [
        "openslide.mpp-x",
        "aperio.MPP",
        "hamamatsu.PhysicalWidth",
        "tiff.XResolution",
    ]
    candidates_y = [
        "openslide.mpp-y",
        "aperio.MPP",
        "hamamatsu.PhysicalHeight",
        "tiff.YResolution",
    ]

    mpp_x = None
    mpp_y = None
    for k in candidates_x:
        if k in props:
            mpp_x = _as_float(props.get(k))
            if mpp_x is not None:
                break
    for k in candidates_y:
        if k in props:
            mpp_y = _as_float(props.get(k))
            if mpp_y is not None:
                break

    # OpenSlide's mpp-x/mpp-y are already microns per pixel. Some TIFF resolution
    # fields are pixels per unit, so only trust the explicit OpenSlide/Aperio fields.
    if mpp_x is not None and mpp_y is not None:
        return float((mpp_x + mpp_y) / 2.0), mpp_x, mpp_y, "openslide properties"
    if mpp_x is not None:
        return float(mpp_x), mpp_x, None, "openslide properties"
    if mpp_y is not None:
        return float(mpp_y), None, mpp_y, "openslide properties"
    return None


def _read_mpp_from_tiffslide(wsi_path: str):
    """Read MPP from slide metadata using tiffslide, when available."""
    try:
        import tiffslide
    except Exception as e:
        logger.debug(f"tiffslide is not available for MPP read: {e}")
        return None

    try:
        slide = tiffslide.TiffSlide(wsi_path)
        props = dict(slide.properties)
        slide.close()
    except Exception as e:
        logger.debug(f"Could not open slide with tiffslide for MPP read: {wsi_path}; {e}")
        return None

    mpp_x = _as_float(props.get("tiffslide.mpp-x") or props.get("openslide.mpp-x"))
    mpp_y = _as_float(props.get("tiffslide.mpp-y") or props.get("openslide.mpp-y"))
    if mpp_x is not None and mpp_y is not None:
        return float((mpp_x + mpp_y) / 2.0), mpp_x, mpp_y, "tiffslide properties"
    if mpp_x is not None:
        return float(mpp_x), mpp_x, None, "tiffslide properties"
    if mpp_y is not None:
        return float(mpp_y), None, mpp_y, "tiffslide properties"
    return None


def get_real_mpp_from_wsi(wsi_path: str, fallback_pixel_size: float = 0.221):
    """
    Return the real MPP stored in the WSI metadata when possible.
    Falls back to fallback_pixel_size only if metadata cannot be read.
    """
    for reader in (_read_mpp_from_openslide, _read_mpp_from_tiffslide):
        result = reader(wsi_path)
        if result is None:
            continue
        mpp, mpp_x, mpp_y, source = result
        if mpp is not None and np.isfinite(mpp) and mpp > 0:
            logger.info(
                f"Using real slide MPP from {source}: mpp={mpp:.6g} "
                f"(mpp_x={mpp_x}, mpp_y={mpp_y}) for {wsi_path}"
            )
            return float(mpp)

    logger.warning(
        f"Could not read real MPP from slide metadata for {wsi_path}; "
        f"falling back to --pixel_size={fallback_pixel_size}"
    )
    return float(fallback_pixel_size)

def get_wsi_size(wsi):
    """
    Best-effort extraction of WSI width/height for hestcore.wsi.NumpyWSI and other WSI wrappers.
    Returns (width, height) or (None, None).
    """
    # Common OpenSlide/PIL-style attributes
    for attr in ["dimensions", "level_dimensions"]:
        if hasattr(wsi, attr):
            try:
                val = getattr(wsi, attr)
                if attr == "level_dimensions" and len(val) > 0:
                    return tuple(val[0])
                if attr == "dimensions" and len(val) == 2:
                    return tuple(val)
            except Exception:
                pass

    # Common image-array attributes on wrappers
    for attr in ["img", "image", "array", "np_img", "numpy_img", "slide", "data"]:
        if hasattr(wsi, attr):
            try:
                arr = getattr(wsi, attr)
                if hasattr(arr, "shape") and len(arr.shape) >= 2:
                    h, w = arr.shape[:2]
                    return int(w), int(h)
                if hasattr(arr, "size") and len(arr.size) == 2:
                    return tuple(arr.size)
            except Exception:
                pass

    # Methods that may return an image or array
    for method in ["get_img", "get_image", "get_thumbnail", "get_numpy", "numpy"]:
        if hasattr(wsi, method):
            try:
                fn = getattr(wsi, method)
                obj = fn() if method not in ["get_thumbnail"] else fn()
                if hasattr(obj, "shape") and len(obj.shape) >= 2:
                    h, w = obj.shape[:2]
                    return int(w), int(h)
                if hasattr(obj, "size") and len(obj.size) == 2:
                    return tuple(obj.size)
            except Exception:
                pass

    return None, None


def debug_wsi_object(wsi):
    """Log useful public attributes/methods of an unknown WSI wrapper."""
    try:
        public_attrs = [a for a in dir(wsi) if not a.startswith("_")]
        logger.info(f"WSI public attrs/methods: {public_attrs[:80]}")
    except Exception as e:
        logger.warning(f"Could not inspect WSI object: {e}")


def debug_spatial_alignment(adata, wsi, sample_name: str = ""):
    logger.info(f"--- Debug spatial alignment: {sample_name} ---")
    logger.info(f"adata shape: {adata.shape}")
    logger.info(f"WSI type: {type(wsi)}")

    wsi_width, wsi_height = get_wsi_size(wsi)
    if wsi_width is not None and wsi_height is not None:
        logger.info(f"WSI size inferred: width={wsi_width}, height={wsi_height}")
    else:
        logger.warning("Could not infer WSI size from this WSI object")
        debug_wsi_object(wsi)

    coords = None
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        logger.info(
            f"obsm['spatial'] x range: {np.nanmin(coords[:, 0])} - {np.nanmax(coords[:, 0])}"
        )
        logger.info(
            f"obsm['spatial'] y range: {np.nanmin(coords[:, 1])} - {np.nanmax(coords[:, 1])}"
        )
        logger.info(f"obsm['spatial'] first 5 rows:\n{coords[:5]}")

        if wsi_width is not None and wsi_height is not None:
            x_min, x_max = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
            y_min, y_max = np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1])
            if x_min < 0 or y_min < 0 or x_max >= wsi_width or y_max >= wsi_height:
                logger.error(
                    "Spatial coordinates are outside WSI bounds: "
                    f"x=[{x_min}, {x_max}] vs width={wsi_width}; "
                    f"y=[{y_min}, {y_max}] vs height={wsi_height}"
                )
            else:
                logger.info("Spatial coordinates are within inferred WSI bounds")
    else:
        logger.warning("No adata.obsm['spatial'] found")

    logger.info(f"obs columns: {list(adata.obs.columns)}")

    for col in [
        "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres",
        "x", "y", "imagecol", "imagerow"
    ]:
        if col in adata.obs.columns:
            vals = pd.to_numeric(adata.obs[col], errors="coerce")
            logger.info(f"obs[{col}] range: {vals.min()} - {vals.max()}")

    if "pxl_col_in_fullres" in adata.obs.columns and "pxl_row_in_fullres" in adata.obs.columns:
        x = pd.to_numeric(adata.obs["pxl_col_in_fullres"], errors="coerce")
        y = pd.to_numeric(adata.obs["pxl_row_in_fullres"], errors="coerce")
        if wsi_width is not None and wsi_height is not None:
            bad = (x < 0) | (y < 0) | (x >= wsi_width) | (y >= wsi_height)
            logger.info(f"spots outside inferred WSI bounds using obs pxl_* columns: {int(bad.sum())} / {adata.n_obs}")

    logger.info(f"uns keys: {list(adata.uns.keys())}")



def _get_wsi_thumbnail_array(wsi, width: int, height: int, max_side: int = 2000):
    """Return a downscaled RGB numpy image from a WSI-like object."""
    scale = max(width, height) / float(max_side)
    if scale < 1:
        scale = 1.0
    thumb_w = max(1, int(round(width / scale)))
    thumb_h = max(1, int(round(height / scale)))

    # Prefer WSI thumbnail APIs when present.
    for method_name in ["get_thumbnail", "thumbnail"]:
        method = getattr(wsi, method_name, None)
        if callable(method):
            try:
                thumb = method((thumb_w, thumb_h))
                return np.asarray(thumb.convert("RGB") if hasattr(thumb, "convert") else thumb), scale
            except Exception as e:
                logger.warning(f"Could not use wsi.{method_name}() for thumbnail: {e}")

    # Fallback for NumpyWSI-like objects that expose the image array.
    for attr in ["img", "image", "arr", "array", "data"]:
        if hasattr(wsi, attr):
            try:
                arr = np.asarray(getattr(wsi, attr))
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.shape[-1] > 3:
                    arr = arr[..., :3]
                im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                im = im.resize((thumb_w, thumb_h), Image.LANCZOS)
                return np.asarray(im), scale
            except Exception as e:
                logger.warning(f"Could not build thumbnail from wsi.{attr}: {e}")

    raise RuntimeError("Could not create thumbnail from WSI object")


def _estimate_spot_diameter_fullres(adata, default: float = 224.0) -> float:
    """Estimate spot/bin diameter in full-resolution pixels from spatial coordinate spacing."""
    if "spatial" not in adata.obsm:
        return default
    coords = np.asarray(adata.obsm["spatial"])
    diffs = []
    for axis in [0, 1]:
        vals = np.unique(coords[:, axis])
        vals = vals[np.isfinite(vals)]
        if len(vals) > 1:
            d = np.diff(np.sort(vals))
            d = d[d > 0]
            if len(d) > 0:
                diffs.append(float(np.median(d)))
    if diffs:
        return float(np.median(diffs))
    return default


def manual_register_downscale_img(adata, wsi, sample_name: str = "manual", max_side: int = 2000):
    """
    Minimal replacement for hest.utils.register_downscale_img used only for plotting.
    This does not change coordinates. It only attaches a downscaled full-resolution image
    in the format expected by scanpy.pl.spatial.
    """
    width, height = get_wsi_size(wsi)
    if width is None or height is None:
        raise RuntimeError("Cannot manually register image because WSI size could not be inferred")
    thumb, downscale_factor = _get_wsi_thumbnail_array(wsi, width, height, max_side=max_side)
    hires_scalef = 1.0 / float(downscale_factor)
    spot_diameter_fullres = _estimate_spot_diameter_fullres(adata)

    library_id = str(sample_name) if sample_name else "manual"
    adata.uns["spatial"] = {
        library_id: {
            "images": {"downscaled_fullres": thumb},
            "scalefactors": {
                "tissue_hires_scalef": hires_scalef,
                "tissue_lowres_scalef": hires_scalef,
                "spot_diameter_fullres": spot_diameter_fullres,
            },
            "metadata": {"source": "manual_register_downscale_img_fallback"},
        }
    }
    logger.warning(
        f"Used manual image fallback for {library_id}: "
        f"thumbnail={thumb.shape}, downscale_factor={downscale_factor:.4f}, "
        f"spot_diameter_fullres={spot_diameter_fullres:.2f}"
    )
    return thumb, downscale_factor

def save_spatial_plot(adata: sc.AnnData, save_path: str, name: str='', key='total_counts', pl_kwargs=None):
    """Save a spatial plot from an AnnData object."""
    import scanpy as sc
    if pl_kwargs is None:
        pl_kwargs = {}

    if key not in adata.obs.columns and key not in adata.var_names:
        logger.warning(f"Cannot plot {key}: key not found in adata.obs or adata.var_names")
        return None

    # Use the only library_id if spatial metadata exists. This avoids Scanpy guessing wrong.
    if "spatial" in adata.uns and isinstance(adata.uns["spatial"], dict) and len(adata.uns["spatial"]) == 1:
        pl_kwargs = dict(pl_kwargs)
        pl_kwargs.setdefault("library_id", next(iter(adata.uns["spatial"].keys())))

    fig = sc.pl.spatial(
        adata,
        show=False,
        img_key="downscaled_fullres",
        color=[key],
        return_fig=True,
        **pl_kwargs,
    )

    filename = f"{name}_spatial_plots.png"
    out = os.path.join(save_path, filename)
    fig.savefig(out, dpi=400)
    logger.info(f"Saved spatial plot: {out}")
    return out

def add_image(adata_path: str, wsi_path: str, pixel_size: float = 0.221):
    """
    Load adata, attach/register a downscaled WSI, and add QC metrics when safe.
    If HEST register_downscale_img fails despite coordinates being inside image bounds,
    fall back to a minimal Scanpy-compatible image registration for plotting.
    """
    sample_name_local = os.path.splitext(os.path.basename(adata_path))[0]

    logger.info(f"Loading WSI for {sample_name_local}: {wsi_path}")
    wsi, _ = load_wsi(wsi_path)

    logger.info(f"Loading adata for {sample_name_local}: {adata_path}")
    adata = sc.read_h5ad(adata_path)

    effective_pixel_size = get_real_mpp_from_wsi(
        wsi_path,
        fallback_pixel_size=pixel_size,
    )
    logger.info(
        f"Pixel size / MPP used for {sample_name_local}: {effective_pixel_size} "
        f"(--pixel_size fallback was {pixel_size})"
    )

    debug_spatial_alignment(adata, wsi, sample_name=sample_name_local)

    try:
        downscaled_fullres, downscale_factor = register_downscale_img(adata, wsi, effective_pixel_size)
        logger.info(f"HEST register_downscale_img succeeded for {sample_name_local}")
    except Exception as e:
        logger.exception(
            f"HEST register_downscale_img failed for {sample_name_local}: {e}. "
            "Trying manual plotting-only image fallback."
        )
        downscaled_fullres, downscale_factor = manual_register_downscale_img(
            adata,
            wsi,
            sample_name=sample_name_local,
            max_side=2000,
        )

    # total_counts belongs in adata.obs, not adata.var_names.
    # Clip negative values to zero before QC so scanpy log1p does not produce invalid values.
    if 'total_counts' not in adata.obs.columns and adata.n_obs > 0:
        x_min = _safe_min_value(adata.X)
        if x_min is None:
            logger.warning(f"Could not determine min(adata.X) for {sample_name_local}; trying QC metrics anyway")
        elif x_min < 0:
            clip_negative_values_to_zero(adata, sample_name=sample_name_local)

        try:
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            logger.info(f"Calculated QC metrics for {sample_name_local}")
        except Exception as e:
            logger.warning(f"Could not calculate QC metrics for {sample_name_local}: {e}")

    logger.info(
        f"Added image / QC for {os.path.basename(adata_path)} "
        f"(factor={downscale_factor})"
    )

    return adata

def save_adata(adata, out_dir: str, name_data: Optional[str] = None):
    """Save an AnnData object to out_dir/adata_processed/<name>.h5ad."""
    if name_data is None:
        name_data = None
        if isinstance(adata.uns, dict) and 'sample_name' in adata.uns:
            name_data = str(adata.uns['sample_name'])
        elif hasattr(adata, 'obs_names') and len(adata.obs_names) > 0:
            name_data = str(adata.obs_names[0]).split('_')[0]
        else:
            raise ValueError("Could not infer sample name for saving. Provide name_data.")

    out_filename = f"{name_data}.h5ad"
    out_path = os.path.join(out_dir, 'adata_processed')
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, out_filename)
    adata.write(out_file)

    logger.info(f"Adata saved: {out_file}")
    return out_file


def add_image_to_adata_and_save(adata_path: str, wsi_path: str, out_dir: str, pixel_size: float = 0.221):
    """Convenience wrapper: run add_image then save adata and spatial plots for a single sample."""
    name_data = os.path.splitext(os.path.basename(adata_path))[0]

    adata = add_image(adata_path, wsi_path, pixel_size=pixel_size)
    saved_adata_path = save_adata(adata, out_dir, name_data=name_data)

    spatial_plot_dir = os.path.join(out_dir, 'spatial_plots')
    os.makedirs(spatial_plot_dir, exist_ok=True)
    save_spatial_plot(adata, save_path=spatial_plot_dir, name=name_data)

    conf_plot_dir = os.path.join(out_dir, 'confidence_plots')
    os.makedirs(conf_plot_dir, exist_ok=True)
    save_spatial_plot(adata, save_path=conf_plot_dir, key='mean_pred_sd_per_spot', name=name_data)

    return saved_adata_path, spatial_plot_dir, conf_plot_dir


def _find_matching_wsi(sample_name: str, wsi_root: str, extensions: Optional[List[str]] = None) -> Optional[str]:
    """
    Look for a WSI file matching sample_name in wsi_root.

    Strict behavior: if globbing finds multiple possible WSIs, return None instead of
    silently choosing the wrong image, because a wrong/cropped WSI can trigger
    "Positions outside range of features" during register_downscale_img.
    """
    if extensions is None:
        extensions = ['.ndpi', '.svs', '.tif', '.tiff', '.mrxs']

    exact_matches = []
    for ext in extensions:
        candidate = os.path.join(wsi_root, f"{sample_name}{ext}")
        if os.path.exists(candidate):
            exact_matches.append(candidate)

    if len(exact_matches) == 1:
        logger.info(f"Exact WSI match for {sample_name}: {exact_matches[0]}")
        return exact_matches[0]

    if len(exact_matches) > 1:
        logger.error(f"Multiple exact WSI matches for {sample_name}; not choosing automatically:")
        for m in exact_matches:
            logger.error(f"  {m}")
        return None

    pattern = os.path.join(wsi_root, f"{sample_name}*")
    matches = sorted([
        m for m in glob.glob(pattern)
        if any(m.lower().endswith(e) for e in extensions)
    ])

    if len(matches) == 1:
        logger.info(f"Glob WSI match for {sample_name}: {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        logger.error(f"Multiple possible WSI matches for {sample_name}; not choosing automatically:")
        for m in matches:
            logger.error(f"  {m}")
        return None

    return None


def combine_run_plots_to_pdf(
    run_dir: str,
    out_pdf: Optional[str] = None,
    rows_per_page: int = 4,
    page_size: tuple = (2480, 3508),  # A4 portrait ~300dpi
) -> str:
    if out_pdf is None:
        out_pdf = os.path.join(run_dir, "combined_plots.pdf")

    spatial_dir = os.path.join(run_dir, "spatial_plots")
    conf_dir = os.path.join(run_dir, "confidence_plots")

    suffix = "_spatial_plots.png"

    samples = sorted([
        os.path.basename(p).replace(suffix, "")
        for p in glob.glob(os.path.join(spatial_dir, f"*{suffix}"))
    ])

    if not samples:
        raise ValueError("No spatial plots found.")

    page_w, page_h = page_size
    row_h = page_h // rows_per_page
    half_w = page_w // 2
    label_h = max(40, row_h // 10)
    img_area = (half_w, row_h - label_h)

    def fit(im, target):
        tw, th = target
        iw, ih = im.size
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        imr = im.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", target, "white")
        canvas.paste(imr, ((tw - nw) // 2, (th - nh) // 2))
        return canvas

    pages = []
    page = Image.new("RGB", page_size, "white")
    row_idx = 0

    for sample in samples:
        sp_path = os.path.join(spatial_dir, f"{sample}{suffix}")
        cf_path = os.path.join(conf_dir, f"{sample}{suffix}")

        sp = Image.open(sp_path).convert("RGB") if os.path.exists(sp_path) else Image.new("RGB", img_area, "white")
        cf = Image.open(cf_path).convert("RGB") if os.path.exists(cf_path) else Image.new("RGB", img_area, "white")

        row = Image.new("RGB", (page_w, row_h), "white")
        row.paste(fit(sp, img_area), (0, label_h))
        row.paste(fit(cf, img_area), (half_w, label_h))

        draw = ImageDraw.Draw(row)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_h // 2)
        except Exception:
            font = ImageFont.load_default()

        bbox = font.getbbox(sample)
        text_height = bbox[3] - bbox[1]

        draw.text(
            (10, (label_h - text_height) // 2),
            sample,
            fill="black",
            font=font,
        )

        page.paste(row, (0, row_idx * row_h))
        row_idx += 1

        if row_idx == rows_per_page:
            pages.append(page)
            page = Image.new("RGB", page_size, "white")
            row_idx = 0

    if row_idx > 0:
        pages.append(page)

    pages[0].save(out_pdf, "PDF", resolution=300.0, save_all=True, append_images=pages[1:])

    logger.info(f"Saved PDF: {out_pdf}")
    return out_pdf


def process_run(
    run_dir: str,
    wsi_root: str,
    pixel_size: float = 0.221,
    adata_subdir_name: str = 'adata',
) -> Tuple[List[str], List[str]]:
    """Process all .h5ad files found under run_dir/<adata_subdir_name>/."""
    adata_dir = os.path.join(run_dir, adata_subdir_name)
    if not os.path.isdir(adata_dir):
        raise FileNotFoundError(f"Adata directory not found: {adata_dir}")

    h5ad_paths = sorted(glob.glob(os.path.join(adata_dir, "*.h5ad")))
    if len(h5ad_paths) == 0:
        logger.warning(f"No .h5ad files found in {adata_dir}")
        return [], []

    saved_adata_paths = []
    spatial_plot_dirs = []

    processed_adata_dir = os.path.join(run_dir, 'adata_processed')
    processed_spatial_dir_root = os.path.join(run_dir, 'spatial_plots')

    for h5ad in h5ad_paths:
        sample_name = os.path.splitext(os.path.basename(h5ad))[0]
        logger.info(f"Processing sample: {sample_name}")

        processed_adata_path = os.path.join(processed_adata_dir, f"{sample_name}.h5ad")
        processed_spatial_dir = os.path.join(processed_spatial_dir_root, sample_name)

        if os.path.exists(processed_adata_path):
            logger.info(f"Skipping {sample_name}: already processed")
            saved_adata_paths.append(processed_adata_path)

            spatial_png_path = os.path.join(
                run_dir,
                "spatial_plots",
                f"{sample_name}_spatial_plots.png",
            )

            if os.path.exists(spatial_png_path):
                spatial_plot_dirs.append(spatial_png_path)
                logger.debug(f"Existing spatial plot: {spatial_png_path}")
            else:
                spatial_plot_dirs.append(None)
                logger.warning(f"Spatial plot PNG not found for {sample_name} (expected {spatial_png_path})")

            continue

        wsi_path = _find_matching_wsi(sample_name, wsi_root)
        if wsi_path is None:
            logger.warning(f"No unambiguous matching WSI for {sample_name} in {wsi_root}. Skipping.")
            saved_adata_paths.append(None)
            spatial_plot_dirs.append(None)
            continue

        try:
            saved_adata, plot_dir, conf_plot_dir = add_image_to_adata_and_save(
                h5ad,
                wsi_path,
                run_dir,
                pixel_size=pixel_size,
            )
            saved_adata_paths.append(saved_adata)
            spatial_plot_dirs.append(plot_dir)
            logger.info(f"Done: adata={saved_adata}, plots={plot_dir}")
        except Exception as e:
            logger.exception(f"Error processing {sample_name if 'sample_name' in locals() else h5ad}: {e}")
            saved_adata_paths.append(None)
            spatial_plot_dirs.append(None)

    out_pdf = os.path.join(run_dir, "combined_plots.pdf")
    try:
        combine_run_plots_to_pdf(run_dir, out_pdf=out_pdf, rows_per_page=4)
        logger.info(f"Combined PDF saved to {out_pdf}")
    except Exception as e:
        logger.error(f"Error creating combined PDF: {e}")

    return saved_adata_paths, spatial_plot_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Attach WSIs to all AnnData files in a run directory and save AnnData & spatial plots."
    )

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--wsi_root", required=True)
    parser.add_argument("--pixel_size", type=float, default=0.221, help="Fallback MPP only. The script first tries to read the real MPP from the WSI/NDPI metadata.")
    parser.add_argument("--adata_subdir_name", default="adata")

    args = parser.parse_args()

    process_run(
        run_dir=args.run_dir,
        wsi_root=args.wsi_root,
        pixel_size=args.pixel_size,
        adata_subdir_name=args.adata_subdir_name,
    )


if __name__ == "__main__":
    main()
