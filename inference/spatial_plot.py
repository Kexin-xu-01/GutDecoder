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


def save_spatial_plot(adata: sc.AnnData, save_path: str, name: str='', key='total_counts', pl_kwargs={}):
    """Save the spatial plot from that sc.AnnData

    Args:
        save_path (str): path to a directory where the spatial plot will be saved
        name (str): save plot as {name}_spatial_plots.png
        key (str): feature to plot. Default: 'total_counts'
        pl_kwargs(Dict): arguments for sc.pl.spatial
    """
    import scanpy as sc
    
    fig = sc.pl.spatial(adata, show=False, img_key="downscaled_fullres", color=[key], return_fig=True, **pl_kwargs)
    
    filename = f"{name}_spatial_plots.png"
    
    # Save the figure
    fig.savefig(os.path.join(save_path, filename), dpi=400)


def add_image(adata_path: str, wsi_path: str, pixel_size: float = 0.221):
    """
    Load adata, attach QC metrics if missing, compute/register a downscaled WSI
    using register_downscale_img. Returns the mutated AnnData.
    """
    wsi, _ = load_wsi(wsi_path)
    adata = sc.read_h5ad(adata_path)
    # register_downscale_img should handle reading coordinates from adata.obs etc.
    downscaled_fullres, downscale_factor = register_downscale_img(adata, wsi, pixel_size)

    # calculate QC metrics if missing (guarded)
    # Note: behavior depends on your scanpy version; this mirrors your original code.
    if 'total_counts' not in adata.var_names and len(adata) > 0:
        sc.pp.calculate_qc_metrics(adata, inplace=True)

    print(f"[add_image] Added QC metrics / downscale for {os.path.basename(adata_path)} "
          f"(downscale factor={downscale_factor})")

    return adata


def save_adata(adata, out_dir: str, name_data: Optional[str] = None):
    """
    Save an AnnData object to out_dir/adata_processed/<name>.h5ad.

    If name_data is not provided, try to infer from:
      - provided name_data argument
      - adata.uns['sample_name'] (if present)
      - adata.obs_names[0] or raise if not possible
    """
    # Infer name_data if not given
    if name_data is None:
        name_data = None
        if isinstance(adata.uns, dict) and 'sample_name' in adata.uns:
            name_data = str(adata.uns['sample_name'])
        elif hasattr(adata, 'obs_names') and len(adata.obs_names) > 0:
            # best-effort fallback: use first obs name (not ideal but safe)
            name_data = str(adata.obs_names[0]).split('_')[0]
        else:
            raise ValueError("Could not infer sample name for saving. Provide name_data.")

    out_filename = f"{name_data}.h5ad"
    out_path = os.path.join(out_dir, 'adata_processed')
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, out_filename)
    adata.write(out_file)

    print(f"\n-- {name_data} PREDICTION ADATA SAVED: {out_file}\n")
    return out_file


def add_image_to_adata_and_save(adata_path: str, wsi_path: str, out_dir: str, pixel_size: float = 0.221):
    """
    Convenience wrapper: run add_image then save adata and spatial plots for a single sample.
    """
    # derive name_data from adata filename
    name_data = os.path.splitext(os.path.basename(adata_path))[0]

    adata = add_image(adata_path, wsi_path, pixel_size=pixel_size)
    saved_adata_path = save_adata(adata, out_dir, name_data=name_data)

    spatial_plot_dir = os.path.join(out_dir, 'spatial_plots')
    os.makedirs(spatial_plot_dir, exist_ok=True)
    save_spatial_plot(adata, save_path=spatial_plot_dir, name = name_data)

    conf_plot_dir = os.path.join(out_dir, 'confidence_plots')
    os.makedirs(conf_plot_dir, exist_ok=True)
    save_spatial_plot(adata, save_path=conf_plot_dir, key = 'mean_pred_sd_per_spot',name = name_data)

    return saved_adata_path, spatial_plot_dir, conf_plot_dir


def _find_matching_wsi(sample_name: str, wsi_root: str, extensions: Optional[List[str]] = None) -> Optional[str]:
    """
    Look for a WSI file matching sample_name in wsi_root. Try multiple extensions (.ndpi, .svs, .tif, .tiff).
    Returns the first match path or None.
    """
    if extensions is None:
        extensions = ['.ndpi', '.svs', '.tif', '.tiff', '.mrxs']
    for ext in extensions:
        candidate = os.path.join(wsi_root, f"{sample_name}{ext}")
        if os.path.exists(candidate):
            return candidate

    # try glob search (in case files are nested or named slightly differently)
    pattern = os.path.join(wsi_root, f"{sample_name}*")
    matches = glob.glob(pattern)
    for m in matches:
        lower = m.lower()
        if any(lower.endswith(e) for e in extensions):
            return m

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

    # collect sample names
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

        bbox = font.getbbox(sample)   # (x0, y0, x1, y1)
        text_height = bbox[3] - bbox[1]

        draw.text(
            (10, (label_h - text_height) // 2),
            sample,
            fill="black",
            font=font
        )

        page.paste(row, (0, row_idx * row_h))
        row_idx += 1

        if row_idx == rows_per_page:
            pages.append(page)
            page = Image.new("RGB", page_size, "white")
            row_idx = 0

    if row_idx > 0:
        pages.append(page)

    pages[0].save(out_pdf, "PDF", resolution=300.0,
                  save_all=True, append_images=pages[1:])

    print(f"Saved PDF: {out_pdf}")
    return out_pdf


def process_run(run_dir: str,
                wsi_root: str,
                pixel_size: float = 0.221,
                adata_subdir_name: str = 'adata') -> Tuple[List[str], List[str]]:
    """
    Wrapper that processes all .h5ad files found under run_dir/<adata_subdir_name>/.

    If a processed adata already exists at:
        run_dir/processed_outputs/adata_processed/<sample_name>.h5ad
    then processing for that sample is skipped.

    Returns:
      (saved_adata_paths, spatial_plot_dirs)
    """
    adata_dir = os.path.join(run_dir, adata_subdir_name)
    if not os.path.isdir(adata_dir):
        raise FileNotFoundError(f"Adata directory not found: {adata_dir}")

    h5ad_paths = sorted(glob.glob(os.path.join(adata_dir, "*.h5ad")))
    if len(h5ad_paths) == 0:
        print(f"[process_run] No .h5ad files found in {adata_dir}")
        return [], []

    saved_adata_paths = []
    spatial_plot_dirs = []

    # where processed outputs are written when using out_dir=run_dir
    processed_adata_dir = os.path.join(run_dir, 'adata_processed')
    processed_spatial_dir_root = os.path.join(run_dir, 'spatial_plots')

    for h5ad in h5ad_paths:
        sample_name = os.path.splitext(os.path.basename(h5ad))[0]
        print(f"[process_run] Processing sample: {sample_name}")

        # path where processed adata would be saved
        processed_adata_path = os.path.join(processed_adata_dir, f"{sample_name}.h5ad")
        processed_spatial_dir = os.path.join(processed_spatial_dir_root, sample_name)

        # If processed adata already exists, skip add_image step
        if os.path.exists(processed_adata_path):
            print(f"  - SKIP: processed adata already exists at {processed_adata_path}")
            saved_adata_paths.append(processed_adata_path)

            # Look for spatial plot PNG 
            spatial_png_path = os.path.join(
                run_dir,
                "spatial_plots",
                f"{sample_name}_spatial_plots.png"
            )

            if os.path.exists(spatial_png_path):
                spatial_plot_dirs.append(spatial_png_path)
                print(f"  - Found existing spatial plot: {spatial_png_path}")
            else:
                spatial_plot_dirs.append(None)
                print(f"  - WARNING: spatial plot PNG not found for {sample_name} "
                    f"(expected {spatial_png_path})")

            continue

        # find matching WSI
        wsi_path = _find_matching_wsi(sample_name, wsi_root)
        if wsi_path is None:
            print(f"  - WARNING: no matching WSI found for {sample_name} in {wsi_root}. Skipping.")
            continue

        try:
            # keep previous behavior: pass run_dir as out_dir so outputs go to run_dir/processed_outputs/...
            saved_adata, plot_dir, conf_plot_dir = add_image_to_adata_and_save(h5ad, wsi_path, run_dir, pixel_size=pixel_size)
            saved_adata_paths.append(saved_adata)
            spatial_plot_dirs.append(plot_dir)
            print(f"  - Done: saved adata -> {saved_adata}, plots -> {plot_dir}")
        except Exception as e:
            print(f"  - ERROR processing {sample_name}: {e}")
            # record failure with None so caller can inspect lengths
            saved_adata_paths.append(None)
            spatial_plot_dirs.append(None)

    out_pdf = os.path.join(run_dir, "combined_plots.pdf")
    try:
        # call the combine function; ensure it's available in this module or imported
        combine_run_plots_to_pdf(run_dir, out_pdf=out_pdf, rows_per_page=4)
        print(f"[process_run] Combined PDF saved to {out_pdf}")
    except Exception as e:
        print(f"[process_run] ERROR while creating combined PDF: {e}")

    return saved_adata_paths, spatial_plot_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Attach WSIs to all AnnData files in a run directory and save AnnData & spatial plots."
    )

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--wsi_root", required=True)
    parser.add_argument("--pixel_size", type=float, default=0.221)

    args = parser.parse_args()

    process_run(
        run_dir=args.run_dir,
        wsi_root=args.wsi_root,
        pixel_size=args.pixel_size,
    )


if __name__ == "__main__":
    main()