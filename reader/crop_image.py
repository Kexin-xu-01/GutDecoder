from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib.path import Path as MplPath

import pyvips
from PIL import Image, ImageDraw

from gutdecoder.reader.rules import Box, exclusion_regions, parse_crop_rules


def build_exclusion_mask_pil(width, height, rules, mpp=None):
    """PIL 'L' mask of the image: 0 = keep, 255 = exclude."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for region in exclusion_regions(rules, (0, width, 0, height), mpp=mpp):
        if isinstance(region, Box):
            x0, x1 = (int(round(min(max(v, 0), width))) for v in (region.x0, region.x1))
            y0, y1 = (int(round(min(max(v, 0), height))) for v in (region.y0, region.y1))
            if x1 > x0 and y1 > y0:
                draw.rectangle((x0, y0, x1, y1), fill=255)
        else:
            draw.polygon(region, fill=255)
    return mask


def mask_tif_openslide_readable(tif_path, out_path, crop_rules, mpp=None, compression="deflate"):
    """
    Read TIFF with pyvips, apply exclusion mask, and save as tiled pyramidal TIFF
    that OpenSlide can read as generic TIFF.
    """
    tif_path = Path(tif_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the slide lazily
    img = pyvips.Image.new_from_file(str(tif_path), access="sequential")
    width, height = img.width, img.height

    # Build a binary exclusion mask using PIL, then save it as a temp PNG
    mask = build_exclusion_mask_pil(width, height, crop_rules, mpp=mpp)

    tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_mask_path = tmp_mask.name
    tmp_mask.close()
    try:
        mask.save(tmp_mask_path)
        cond = pyvips.Image.new_from_file(tmp_mask_path, access="sequential")

        # White fill for excluded regions
        fill_value = [255] * img.bands if img.bands > 1 else 255
        white = img.new_from_image(fill_value)

        # Non-zero in cond => white, zero => original
        masked = cond.ifthenelse(white, img)

        masked.tiffsave(
            str(out_path),
            tile=True,
            pyramid=True,
            bigtiff=True,
            compression=compression,
            tile_width=256,
            tile_height=256,
            properties=True,
        )
    finally:
        if os.path.exists(tmp_mask_path):
            os.unlink(tmp_mask_path)


def mask_tif(
    tif_dir,
    excel_path,
    out_dir_name="tif_masked_os",
    sample_id_col="Sample_ID",
    crop_col="crop_100_um",
    mpp=None,
):
    """
    Match TIFF filename stem to sample_id, read crop rules from Excel,
    and save OpenSlide-readable masked copies.
    """
    tif_dir = Path(tif_dir)
    out_dir = tif_dir.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path)
    df = df[[sample_id_col, crop_col]].dropna(subset=[sample_id_col])

    crop_map = {str(row[sample_id_col]): row[crop_col] for _, row in df.iterrows()}

    for tif_path in sorted(tif_dir.glob("*.tif")):
        sample_id = tif_path.stem
        out_path = out_dir / tif_path.name

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[SKIP] already exists: {out_path.name}")
            continue

        crop_str = crop_map.get(sample_id, None)

        if crop_str is None or pd.isna(crop_str) or str(crop_str).strip() == "":
            print(f"[COPY] no crop rule: {tif_path.name}")
            import shutil
            shutil.copy2(tif_path, out_path)
            continue

        rules = parse_crop_rules(crop_str)
        print(f"[WRITE] {tif_path.name} -> {out_path.name}")
        mask_tif_openslide_readable(tif_path, out_path, rules, mpp=mpp)

    print(f"Done. Output folder: {out_dir}")