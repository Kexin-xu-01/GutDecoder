"""Structural completion checks compatible with existing HEST outputs."""
import json
import math
from pathlib import Path

import h5py


def patch_is_complete(path, settings=None):
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            barcode_key = "barcodes" if "barcodes" in handle else "barcode"
            if not {"img", "coords", barcode_key}.issubset(handle):
                return False
            images, coords, barcodes = handle["img"], handle["coords"], handle[barcode_key]
            n = images.shape[0]
            if images.ndim != 4 or coords.shape != (n, 2) or barcodes.shape[0] != n:
                return False
            if settings and images.shape[1:3] != (settings["target_patch_size"],) * 2:
                return False
            if settings and "target_magnification" in coords.attrs:
                if not math.isclose(float(coords.attrs["target_magnification"]),
                                    10.0 / settings["target_pixel_size"], rel_tol=1e-6):
                    return False
            # Read boundary records to catch truncated datasets without scanning gigabytes.
            if n:
                images[0], images[-1], coords[-1], barcodes[-1]
        metadata = path.with_suffix(".complete.json")
        if metadata.exists() and settings is not None:
            return json.loads(metadata.read_text())["settings"] == settings
        # Legacy outputs have no manifest: accept structural validity and patch size.
        return True
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def hest_is_complete(directory):
    directory = Path(directory)
    try:
        with h5py.File(directory / "aligned_adata.h5ad", "r") as handle:
            if not {"obs", "var", "X", "obsm"}.issubset(handle) or "spatial" not in handle["obsm"]:
                return False
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics.get("pixel_size_um_estimated", 0) <= 0:
            return False
        from tifffile import TiffFile
        with TiffFile(directory / "aligned_fullres_HE.tif") as image:
            return all(size > 0 for size in image.pages[0].shape[:2])
    except (OSError, ValueError, KeyError, TypeError):
        return False
