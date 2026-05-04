import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib.path import Path as MplPath


def parse_crop_rules(crop_str):
    """
    Parse a crop_100_um cell into a list of rule dicts.

    Handles examples like:
      "{""type"": ""strip"", ...}, {""type"":""corner"", ...}"
      "{'type': 'strip', ...}"
      "{\"type\":\"corner\", ...}"
      single dicts
      multiple dicts without outer brackets
    """
    if pd.isna(crop_str) or not str(crop_str).strip():
        return []

    s = str(crop_str).strip()

    # 1) direct literal eval
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
        if isinstance(obj, str):
            s = obj.strip()
    except Exception:
        pass

    # 2) wrapped list
    for candidate in (s, f"[{s}]"):
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return [obj]
            if isinstance(obj, list):
                return obj
            if isinstance(obj, str):
                s = obj.strip()
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

    # 3) extract dict-like blocks
    blocks = re.findall(r"\{[^{}]*\}", s, flags=re.S)
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
    """
    Convert a rule value to pixels.
    """
    if units == "frac":
        return int(round(float(value) * dim))
    if units in {"px", "pixel", "pixels"}:
        return int(round(float(value)))
    if units in {"um", "µm", "micron", "microns"}:
        if mpp is None:
            raise ValueError("mpp is required when units are um")
        return int(round(float(value) / float(mpp)))
    raise ValueError(f"Unknown units: {units}")


def _clip_int(v, lo, hi):
    return int(max(lo, min(hi, int(v))))


def _polygon_exclude_mask(H, W, poly_x, poly_y):
    """
    Rasterize a polygon into a boolean mask of shape (H, W),
    where True means exclude.
    """
    mask = np.zeros((H, W), dtype=bool)

    xmin = max(int(np.floor(np.min(poly_x))), 0)
    xmax = min(int(np.ceil(np.max(poly_x))), W)
    ymin = max(int(np.floor(np.min(poly_y))), 0)
    ymax = min(int(np.ceil(np.max(poly_y))), H)

    if xmin >= xmax or ymin >= ymax:
        return mask

    xs = np.arange(xmin, xmax) + 0.5
    ys = np.arange(ymin, ymax) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    path = MplPath(np.column_stack([poly_x, poly_y]))
    inside = path.contains_points(pts)

    mask[ymin:ymax, xmin:xmax] = inside.reshape((ymax - ymin, xmax - xmin))
    return mask


def build_keep_mask_from_rules(shape, rules, mpp=None):
    """
    Build a boolean mask where True = keep, False = exclude.

    Supports:
      - corner
      - strip (side+size)
      - strip (axis+bounds)
      - rect
      - trapezoid
    """
    H, W = shape[:2]
    exclude = np.zeros((H, W), dtype=bool)

    for r in rules:
        rtype = r.get("type")

        if rtype == "corner":
            units = r.get("units", "px")
            corner = r["corner"]

            ww = r["width"]
            hh = r["height"]

            if units == "frac":
                ww = int(round(float(ww) * W))
                hh = int(round(float(hh) * H))
            else:
                ww = _to_pixels(ww, units, W, mpp=mpp)
                hh = _to_pixels(hh, units, H, mpp=mpp)

            ww = max(0, min(W, ww))
            hh = max(0, min(H, hh))

            if corner == "top-left":
                exclude[:hh, :ww] = True
            elif corner == "top-right":
                exclude[:hh, W - ww:] = True
            elif corner == "bottom-left":
                exclude[H - hh:, :ww] = True
            elif corner == "bottom-right":
                exclude[H - hh:, W - ww:] = True
            else:
                raise ValueError("corner must be one of: top-left, top-right, bottom-left, bottom-right")

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
                        x0 = int(round(min(start, end) * W))
                        x1 = int(round(max(start, end) * W))
                    else:
                        x0 = int(round(min(start, end)))
                        x1 = int(round(max(start, end)))

                    x0 = _clip_int(x0, 0, W)
                    x1 = _clip_int(x1, 0, W)
                    if x1 > x0:
                        exclude[:, x0:x1] = True

                else:
                    if units == "frac":
                        y0 = int(round(min(start, end) * H))
                        y1 = int(round(max(start, end) * H))
                    else:
                        y0 = int(round(min(start, end)))
                        y1 = int(round(max(start, end)))

                    y0 = _clip_int(y0, 0, H)
                    y1 = _clip_int(y1, 0, H)
                    if y1 > y0:
                        exclude[y0:y1, :] = True

            else:
                side = r["side"]
                ss = r["size"]

                if units == "frac":
                    ss = int(round(float(ss) * (H if side in ("top", "bottom") else W)))
                else:
                    ss = _to_pixels(ss, units, H if side in ("top", "bottom") else W, mpp=mpp)

                ss = max(0, ss)

                if side == "top":
                    ss = min(H, ss)
                    exclude[:ss, :] = True
                elif side == "bottom":
                    ss = min(H, ss)
                    exclude[H - ss:, :] = True
                elif side == "left":
                    ss = min(W, ss)
                    exclude[:, :ss] = True
                elif side == "right":
                    ss = min(W, ss)
                    exclude[:, W - ss:] = True
                else:
                    raise ValueError("side must be one of: top, bottom, left, right")

        elif rtype == "rect":
            units = r.get("units", "px")

            if units == "frac":
                rxmin = int(round(float(r["xmin"]) * W))
                rxmax = int(round(float(r["xmax"]) * W))
                rymin = int(round(float(r["ymin"]) * H))
                rymax = int(round(float(r["ymax"]) * H))
            else:
                rxmin = _to_pixels(r["xmin"], units, W, mpp=mpp)
                rxmax = _to_pixels(r["xmax"], units, W, mpp=mpp)
                rymin = _to_pixels(r["ymin"], units, H, mpp=mpp)
                rymax = _to_pixels(r["ymax"], units, H, mpp=mpp)

            x0, x1 = sorted((rxmin, rxmax))
            y0, y1 = sorted((rymin, rymax))

            x0 = _clip_int(x0, 0, W)
            x1 = _clip_int(x1, 0, W)
            y0 = _clip_int(y0, 0, H)
            y1 = _clip_int(y1, 0, H)

            if x1 > x0 and y1 > y0:
                exclude[y0:y1, x0:x1] = True

        elif rtype == "trapezoid":
            units = r.get("units", "px")
            ori = r.get("orientation", "top")
            top_w = r.get("top_width")
            bot_w = r.get("bottom_width")
            height = r.get("height")
            offset = r.get("center_offset", 0.0)

            if top_w is None or bot_w is None or height is None:
                raise ValueError("trapezoid requires 'top_width', 'bottom_width', and 'height'")

            if units == "frac":
                if ori in ("top", "bottom"):
                    top_w = float(top_w) * W
                    bot_w = float(bot_w) * W
                    height = float(height) * H
                    offset = float(offset) * W
                else:
                    top_w = float(top_w) * H
                    bot_w = float(bot_w) * H
                    height = float(height) * W
                    offset = float(offset) * H
            else:
                if ori in ("top", "bottom"):
                    top_w = _to_pixels(top_w, units, W, mpp=mpp)
                    bot_w = _to_pixels(bot_w, units, W, mpp=mpp)
                    height = _to_pixels(height, units, H, mpp=mpp)
                    offset = _to_pixels(offset, units, W, mpp=mpp)
                else:
                    top_w = _to_pixels(top_w, units, H, mpp=mpp)
                    bot_w = _to_pixels(bot_w, units, H, mpp=mpp)
                    height = _to_pixels(height, units, W, mpp=mpp)
                    offset = _to_pixels(offset, units, H, mpp=mpp)

            if ori == "top":
                cx = W / 2.0 + offset
                poly_x = np.array([cx - top_w / 2.0, cx + top_w / 2.0, cx + bot_w / 2.0, cx - bot_w / 2.0])
                poly_y = np.array([0.0, 0.0, height, height])

            elif ori == "bottom":
                cx = W / 2.0 + offset
                poly_x = np.array([cx - bot_w / 2.0, cx + bot_w / 2.0, cx + top_w / 2.0, cx - top_w / 2.0])
                poly_y = np.array([H - height, H - height, H, H])

            elif ori == "left":
                cy = H / 2.0 + offset
                poly_x = np.array([0.0, 0.0, height, height])
                poly_y = np.array([cy - top_w / 2.0, cy + top_w / 2.0, cy + bot_w / 2.0, cy - bot_w / 2.0])

            elif ori == "right":
                cy = H / 2.0 + offset
                poly_x = np.array([W - height, W - height, W, W])
                poly_y = np.array([cy - bot_w / 2.0, cy + bot_w / 2.0, cy + top_w / 2.0, cy - top_w / 2.0])

            else:
                raise ValueError("orientation must be one of: top, bottom, left, right")

            exclude |= _polygon_exclude_mask(H, W, poly_x, poly_y)

        else:
            raise ValueError(f"Unknown rule type: {rtype}")

    return ~exclude

def mask_tif_with_rules(tif_path, out_path, rules, mpp=None, white_fill=True):
    """
    Save a TIFF copy with excluded areas painted white (or black).
    Keeps the same canvas size.
    """
    tif_path = Path(tif_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(tif_path) as tf:
        img = tf.asarray()
        page0 = tf.pages[0]
        desc = getattr(page0, "description", None)
        is_bigtiff = tf.is_bigtiff

        tags = page0.tags
        xres = tags["XResolution"].value if "XResolution" in tags else None
        yres = tags["YResolution"].value if "YResolution" in tags else None
        unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None

    keep = build_keep_mask_from_rules(img.shape, rules, mpp=mpp)

    if np.issubdtype(img.dtype, np.integer):
        fill_value = np.iinfo(img.dtype).max if white_fill else 0
    elif np.issubdtype(img.dtype, np.floating):
        fill_value = 1.0 if white_fill else 0.0
    else:
        fill_value = 255 if white_fill else 0

    out = img.copy()
    if out.ndim == 2:
        out[~keep] = fill_value
    else:
        out[~keep, :] = fill_value

    imwrite_kwargs = {}
    if desc is not None:
        imwrite_kwargs["description"] = desc
    if xres is not None and yres is not None and unit is not None:
        imwrite_kwargs["resolution"] = (xres[0] / xres[1], yres[0] / yres[1])
        if unit == 2:
            imwrite_kwargs["resolutionunit"] = "INCH"
        elif unit == 3:
            imwrite_kwargs["resolutionunit"] = "CENTIMETER"

    tiff.imwrite(out_path, out, bigtiff=is_bigtiff, **imwrite_kwargs)
    return out_path


def mask_tif(
    tif_dir,
    excel_path,
    out_dir_name="tif_masked",
    sample_id_col="Sample_ID",
    crop_col="crop_100_um",
    mpp=None,
    review_csv="mask_review.csv",
):
    """
    Match TIFF filename stem to sample_id, read crop rules from Excel,
    and save masked copies.
    """
    tif_dir = Path(tif_dir)
    out_dir = tif_dir.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)


    df = pd.read_excel(excel_path)
    df = df[[sample_id_col, crop_col]].dropna(subset=[sample_id_col])

    crop_map = {str(row[sample_id_col]): row[crop_col] for _, row in df.iterrows()}

    review_rows = []

    for tif_path in sorted(tif_dir.glob("*.tif")):
        sample_id = tif_path.stem
        crop_str = crop_map.get(sample_id, None)
        out_path = out_dir / tif_path.name

        if crop_str is None or pd.isna(crop_str) or str(crop_str).strip() == "":
            print(f"[COPY] no crop rule: {tif_path.name}")

            import shutil
            shutil.copy2(tif_path, out_path)

            review_rows.append({
                "sample_id": sample_id,
                "file": str(tif_path),
                "status": "copied_no_rule",
            })
            continue

        rules = parse_crop_rules(crop_str)
        if not rules:
            print(f"[COPY] parse failed: {tif_path.name}")

            import shutil
            shutil.copy2(tif_path, out_path)

            review_rows.append({
                "sample_id": sample_id,
                "file": str(tif_path),
                "status": "copied_parse_failed",
                "crop_value": str(crop_str),
            })
            continue


        if out_path.exists():
            try:
                import tifffile as tiff
                with tiff.TiffFile(out_path):
                    print(f"[SKIP] valid existing file: {out_path.name}")
                    continue
            except Exception:
                print(f"[RETRY] corrupt file, regenerating: {out_path.name}")

        print(f"[WRITE] {tif_path.name} -> {out_path.name}")
        mask_tif_with_rules(tif_path, out_path, rules, mpp=mpp, white_fill=True)

    if review_rows:
        review_path = out_dir / review_csv
        pd.DataFrame(review_rows).to_csv(review_path, index=False)
        print(f"Review log written to: {review_path}")

    print(f"Done. Output folder: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_dir", required=True)
    parser.add_argument("--excel_path", required=True)
    parser.add_argument("--out_dir_name", default="tif_masked")
    parser.add_argument("--sample_id_col", default="Sample_ID")
    parser.add_argument("--crop_col", default="crop_100_um")
    parser.add_argument("--mpp", type=float, default=None)

    args = parser.parse_args()

    mask_tif(
        tif_dir=args.tif_dir,
        excel_path=args.excel_path,
        out_dir_name=args.out_dir_name,
        sample_id_col=args.sample_id_col,
        crop_col=args.crop_col,
        mpp=args.mpp,
    )