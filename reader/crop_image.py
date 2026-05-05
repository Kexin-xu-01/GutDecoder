import ast
import json
import re
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib.path import Path as MplPath

import pyvips
from PIL import Image, ImageDraw


def parse_crop_rules(crop_str):
    """
    Parse a crop_100_um cell into a list of rule dicts.
    Handles:
      - single dict
      - multiple dicts separated by commas/newlines
      - single/double quotes
      - wrapped/unwrapped text
    """
    if pd.isna(crop_str) or not str(crop_str).strip():
        return []

    s = str(crop_str).strip()

    # Try literal eval directly
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Try wrapped in list
    for candidate in (s, f"[{s}]"):
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return [obj]
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return [obj]
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    # Extract dict-like blocks
    blocks = []
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(s[start : i + 1])

    rules = []
    for block in blocks:
        txt = block.replace('""', '"')
        try:
            obj = ast.literal_eval(txt)
        except Exception:
            try:
                obj = json.loads(txt.replace("'", '"'))
            except Exception:
                continue
        if isinstance(obj, dict):
            rules.append(obj)

    return rules


def _to_pixels(value, units, dim, mpp=None):
    if units == "frac":
        return int(round(float(value) * dim))
    if units in {"px", "pixel", "pixels"}:
        return int(round(float(value)))
    if units in {"um", "µm", "micron", "microns"}:
        if mpp is None:
            raise ValueError("mpp is required when units are um")
        return int(round(float(value) / float(mpp)))
    raise ValueError(f"Unknown units: {units}")


def build_exclusion_mask_pil(width, height, rules, mpp=None):
    """
    Build a PIL 'L' mask image:
      0 = keep
      255 = exclude
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for r in rules:
        rtype = r.get("type")

        if rtype == "corner":
            units = r.get("units", "px")
            corner = r["corner"]
            ww = _to_pixels(r["width"], units, width, mpp=mpp)
            hh = _to_pixels(r["height"], units, height, mpp=mpp)

            ww = max(0, min(width, ww))
            hh = max(0, min(height, hh))

            if corner == "top-left":
                box = (0, 0, ww, hh)
            elif corner == "top-right":
                box = (width - ww, 0, width, hh)
            elif corner == "bottom-left":
                box = (0, height - hh, ww, height)
            elif corner == "bottom-right":
                box = (width - ww, height - hh, width, height)
            else:
                raise ValueError("corner must be one of: top-left, top-right, bottom-left, bottom-right")

            draw.rectangle(box, fill=255)

        elif rtype == "strip":
            units = r.get("units", "px")
            bounds = r.get("bounds", None)

            if bounds is not None:
                if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                    raise ValueError("strip 'bounds' must be a 2-tuple/list (start, end)")

                start, end = bounds
                axis = r.get("axis")
                side = r.get("side")

                if axis == "x" or side in ("left", "right"):
                    if units == "frac":
                        x0 = int(round(min(start, end) * width))
                        x1 = int(round(max(start, end) * width))
                    else:
                        x0 = int(round(min(start, end)))
                        x1 = int(round(max(start, end)))
                    x0 = max(0, min(width, x0))
                    x1 = max(0, min(width, x1))
                    if x1 > x0:
                        draw.rectangle((x0, 0, x1, height), fill=255)
                else:
                    if units == "frac":
                        y0 = int(round(min(start, end) * height))
                        y1 = int(round(max(start, end) * height))
                    else:
                        y0 = int(round(min(start, end)))
                        y1 = int(round(max(start, end)))
                    y0 = max(0, min(height, y0))
                    y1 = max(0, min(height, y1))
                    if y1 > y0:
                        draw.rectangle((0, y0, width, y1), fill=255)

            else:
                side = r["side"]
                ss = _to_pixels(r["size"], units, height if side in ("top", "bottom") else width, mpp=mpp)
                ss = max(0, ss)

                if side == "top":
                    draw.rectangle((0, 0, width, min(height, ss)), fill=255)
                elif side == "bottom":
                    draw.rectangle((0, max(0, height - ss), width, height), fill=255)
                elif side == "left":
                    draw.rectangle((0, 0, min(width, ss), height), fill=255)
                elif side == "right":
                    draw.rectangle((max(0, width - ss), 0, width, height), fill=255)
                else:
                    raise ValueError("side must be one of: top, bottom, left, right")

        elif rtype == "rect":
            units = r.get("units", "px")
            if units == "frac":
                rxmin = int(round(float(r["xmin"]) * width))
                rxmax = int(round(float(r["xmax"]) * width))
                rymin = int(round(float(r["ymin"]) * height))
                rymax = int(round(float(r["ymax"]) * height))
            else:
                rxmin = _to_pixels(r["xmin"], units, width, mpp=mpp)
                rxmax = _to_pixels(r["xmax"], units, width, mpp=mpp)
                rymin = _to_pixels(r["ymin"], units, height, mpp=mpp)
                rymax = _to_pixels(r["ymax"], units, height, mpp=mpp)

            x0, x1 = sorted((rxmin, rxmax))
            y0, y1 = sorted((rymin, rymax))
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            if x1 > x0 and y1 > y0:
                draw.rectangle((x0, y0, x1, y1), fill=255)

        elif rtype == "trapezoid":
            units = r.get("units", "px")
            ori = r.get("orientation", "top")
            top_w = r.get("top_width")
            bot_w = r.get("bottom_width")
            height_w = r.get("height")
            offset = r.get("center_offset", 0.0)

            if top_w is None or bot_w is None or height_w is None:
                raise ValueError("trapezoid requires 'top_width', 'bottom_width', and 'height'")

            if units == "frac":
                if ori in ("top", "bottom"):
                    top_w = float(top_w) * width
                    bot_w = float(bot_w) * width
                    height_w = float(height_w) * height
                    offset = float(offset) * width
                else:
                    top_w = float(top_w) * height
                    bot_w = float(bot_w) * height
                    height_w = float(height_w) * width
                    offset = float(offset) * height
            else:
                top_w = _to_pixels(top_w, units, width if ori in ("top", "bottom") else height, mpp=mpp)
                bot_w = _to_pixels(bot_w, units, width if ori in ("top", "bottom") else height, mpp=mpp)
                height_w = _to_pixels(height_w, units, height if ori in ("top", "bottom") else width, mpp=mpp)
                offset = _to_pixels(offset, units, width if ori in ("top", "bottom") else height, mpp=mpp)

            if ori == "top":
                cx = width / 2.0 + offset
                poly = [
                    (cx - top_w / 2.0, 0),
                    (cx + top_w / 2.0, 0),
                    (cx + bot_w / 2.0, height_w),
                    (cx - bot_w / 2.0, height_w),
                ]
            elif ori == "bottom":
                cx = width / 2.0 + offset
                poly = [
                    (cx - bot_w / 2.0, height - height_w),
                    (cx + bot_w / 2.0, height - height_w),
                    (cx + top_w / 2.0, height),
                    (cx - top_w / 2.0, height),
                ]
            elif ori == "left":
                cy = height / 2.0 + offset
                poly = [
                    (0, cy - top_w / 2.0),
                    (0, cy + top_w / 2.0),
                    (height_w, cy + bot_w / 2.0),
                    (height_w, cy - bot_w / 2.0),
                ]
            elif ori == "right":
                cy = height / 2.0 + offset
                poly = [
                    (width - height_w, cy - bot_w / 2.0),
                    (width - height_w, cy + bot_w / 2.0),
                    (width, cy + top_w / 2.0),
                    (width, cy - top_w / 2.0),
                ]
            else:
                raise ValueError("orientation must be one of: top, bottom, left, right")

            draw.polygon(poly, fill=255)

        else:
            raise ValueError(f"Unknown rule type: {rtype}")

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
        if not rules:
            print(f"[COPY] parse failed: {tif_path.name}")
            import shutil
            shutil.copy2(tif_path, out_path)
            continue

        print(f"[WRITE] {tif_path.name} -> {out_path.name}")
        mask_tif_openslide_readable(tif_path, out_path, rules, mpp=mpp)

    print(f"Done. Output folder: {out_dir}")