"""Parsing of crop/exclusion rules stored in the sample spreadsheets."""
import ast
import json

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
