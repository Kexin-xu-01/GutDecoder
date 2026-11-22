"""Crop/exclusion rules from the sample spreadsheets: parsing and region geometry."""
import ast
import json
import math
from dataclasses import dataclass

import pandas as pd


def _literal(text):
    """Parse a Python or JSON literal; None if the text is neither."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_crop_rules(crop_str):
    """
    Parse a crop_100_um spreadsheet cell into a list of rule dicts.

    Accepts a dict, a list of dicts, or several dicts separated by commas/newlines,
    with single or double quotes. Raises ValueError if any part cannot be parsed,
    so a typo never results in a silently unmasked slide.
    """
    if pd.isna(crop_str) or not str(crop_str).strip():
        return []

    s = str(crop_str).strip()

    for candidate in (s, f"[{s}]"):
        obj = _literal(candidate)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj

    # Extract dict-like blocks, e.g. from text pasted with doubled quotes
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
    if not blocks:
        raise ValueError(f"No crop rule found in {crop_str!r}")

    rules = []
    for block in blocks:
        txt = block.replace('""', '"')
        obj = _literal(txt)
        if obj is None:
            obj = _literal(txt.replace("'", '"'))
        if not isinstance(obj, dict):
            raise ValueError(f"Cannot parse crop rule {block!r} in {crop_str!r}")
        rules.append(obj)
    return rules


@dataclass
class Box:
    """Axis-aligned region; `closed` decides whether points on its edges count as inside."""
    x0: float
    x1: float
    y0: float
    y1: float
    closed: bool


_CORNERS = {"top-left", "top-right", "bottom-left", "bottom-right"}
_SIDES = {"top", "bottom", "left", "right"}


def _length(value, units, extent, mpp):
    """A length in pixels: 'frac' is a fraction of `extent`, 'um' needs `mpp`."""
    value = float(value)
    if units == "frac":
        return value * extent
    if units in {"px", "pixel", "pixels"}:
        return value
    if units in {"um", "µm", "micron", "microns"}:
        if mpp is None:
            raise ValueError("mpp is required when rule units are um")
        return value / float(mpp)
    raise ValueError(f"Unknown units: {units}")


def _position(value, units, origin, extent, mpp):
    """A coordinate in pixels: 'frac' is relative to the bbox, 'px'/'um' are absolute."""
    if units == "frac":
        return origin + float(value) * extent
    return _length(value, units, extent, mpp)


def exclusion_regions(rules, bbox, mpp=None):
    """
    Regions to exclude, as Box or polygon vertex lists, in pixel coordinates.

    bbox is (xmin, xmax, ymin, ymax). Supported rules (units 'px' | 'frac' | 'um', default 'px'):
      - {'type': 'corner', 'corner': 'top-left|top-right|bottom-left|bottom-right', 'width', 'height'}
      - {'type': 'strip', 'side': 'top|bottom|left|right', 'size'}
      - {'type': 'strip', 'axis': 'x'|'y', 'bounds': (start, end)}   # 'bounds' takes precedence
      - {'type': 'rect', 'xmin', 'xmax', 'ymin', 'ymax'}
      - {'type': 'trapezoid', 'orientation': 'top|bottom|left|right', 'top_width', 'bottom_width',
         'height', 'center_offset' (optional)}; the base sits on the named bbox edge.
    """
    xmin, xmax, ymin, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    inf = math.inf
    regions = []

    for r in rules:
        rtype = r.get("type")
        units = r.get("units", "px")

        if rtype == "corner":
            if r["corner"] not in _CORNERS:
                raise ValueError(f"corner must be one of: {sorted(_CORNERS)}")
            ww = _length(r["width"], units, w, mpp)
            hh = _length(r["height"], units, h, mpp)
            vertical, horizontal = r["corner"].split("-")
            x0, x1 = (-inf, xmin + ww) if horizontal == "left" else (xmax - ww, inf)
            y0, y1 = (-inf, ymin + hh) if vertical == "top" else (ymax - hh, inf)
            regions.append(Box(x0, x1, y0, y1, closed=False))

        elif rtype == "strip" and r.get("bounds") is not None:
            bounds = r["bounds"]
            if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                raise ValueError("strip 'bounds' must be a 2-tuple/list (start, end)")
            if r.get("axis") == "x" or r.get("side") in ("left", "right"):
                a, b = sorted(_position(v, units, xmin, w, mpp) for v in bounds)
                regions.append(Box(a, b, -inf, inf, closed=True))
            else:
                a, b = sorted(_position(v, units, ymin, h, mpp) for v in bounds)
                regions.append(Box(-inf, inf, a, b, closed=True))

        elif rtype == "strip":
            side = r["side"]
            if side not in _SIDES:
                raise ValueError(f"side must be one of: {sorted(_SIDES)}")
            ss = _length(r["size"], units, h if side in ("top", "bottom") else w, mpp)
            x0, x1, y0, y1 = {
                "top": (-inf, inf, -inf, ymin + ss),
                "bottom": (-inf, inf, ymax - ss, inf),
                "left": (-inf, xmin + ss, -inf, inf),
                "right": (xmax - ss, inf, -inf, inf),
            }[side]
            regions.append(Box(x0, x1, y0, y1, closed=False))

        elif rtype == "rect":
            x0, x1 = sorted(_position(r[k], units, xmin, w, mpp) for k in ("xmin", "xmax"))
            y0, y1 = sorted(_position(r[k], units, ymin, h, mpp) for k in ("ymin", "ymax"))
            regions.append(Box(x0, x1, y0, y1, closed=True))

        elif rtype == "trapezoid":
            ori = r.get("orientation", "top")
            if ori not in _SIDES:
                raise ValueError(f"orientation must be one of: {sorted(_SIDES)}")
            if r.get("top_width") is None or r.get("bottom_width") is None or r.get("height") is None:
                raise ValueError("trapezoid requires 'top_width', 'bottom_width', and 'height'")
            along, across = (w, h) if ori in ("top", "bottom") else (h, w)
            top_w = _length(r["top_width"], units, along, mpp)
            bot_w = _length(r["bottom_width"], units, along, mpp)
            depth = _length(r["height"], units, across, mpp)
            offset = _length(r.get("center_offset", 0.0), units, along, mpp)
            cx = (xmin + xmax) / 2.0 + offset
            cy = (ymin + ymax) / 2.0 + offset
            regions.append({
                "top": [(cx - top_w / 2, ymin), (cx + top_w / 2, ymin),
                        (cx + bot_w / 2, ymin + depth), (cx - bot_w / 2, ymin + depth)],
                "bottom": [(cx - bot_w / 2, ymax - depth), (cx + bot_w / 2, ymax - depth),
                           (cx + top_w / 2, ymax), (cx - top_w / 2, ymax)],
                "left": [(xmin, cy - top_w / 2), (xmin, cy + top_w / 2),
                         (xmin + depth, cy + bot_w / 2), (xmin + depth, cy - bot_w / 2)],
                "right": [(xmax - depth, cy - bot_w / 2), (xmax - depth, cy + bot_w / 2),
                          (xmax, cy + top_w / 2), (xmax, cy - top_w / 2)],
            }[ori])

        else:
            raise ValueError(f"Unknown rule type: {rtype}")

    return regions
