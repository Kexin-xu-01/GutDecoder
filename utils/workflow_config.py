"""Configuration parsing shared by the existing workflow entry points."""
from pathlib import Path
from collections.abc import Mapping

import yaml


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off", "none", ""}


def coerce_bool(value):
    """Parse a config boolean; raise on anything that is not a clear yes/no (e.g. a typo)."""
    if isinstance(value, bool):
        return value
    text = "none" if value is None else str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"Expected a boolean (true/false, yes/no, 1/0); got {value!r}")


def _int_or_str_lower(value):
    if value is None or isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return str(value).lower()


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return cfg


def benchmark_arguments(cfg, *, multislide=False):
    """Preserve legacy coercions, validating values before expensive processing."""
    raw = cfg.get("args", cfg)
    if not isinstance(raw, Mapping):
        raise ValueError("args must contain a mapping")
    args = dict(raw)
    for key in (("save_dir", "base_root") if multislide else ("save_dir",)):
        if key in args:
            args[key] = Path(args[key])
    for key in ("K", "gene_k"):
        if args.get(key) is not None:
            args[key] = _int_or_str_lower(args[key])
            value = args[key]
            allowed = "loocv" if key == "K" else "auto"
            if value != allowed and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{key} must be a positive integer or {allowed!r}")
    if args.get("seed") is not None:
        try:
            args["seed"] = int(args["seed"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"seed must be an integer; got: {args['seed']}") from exc
    if "symlink" in args:
        args["symlink"] = coerce_bool(args["symlink"])
    if "slide_subdirs" in args and not isinstance(args["slide_subdirs"], (list, tuple)):
        raise ValueError("slide_subdirs must be a list of folder names")
    if "min_cells_pct" in args and not 0 <= args["min_cells_pct"] <= 1:
        raise ValueError("min_cells_pct must be between 0 and 1")
    return args


def validate_patching(settings):
    if settings["target_patch_size"] <= 0 or settings["target_pixel_size"] <= 0:
        raise ValueError("Patch size and pixel size must be positive")
    if not 0 <= settings["threshold"] <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return settings
