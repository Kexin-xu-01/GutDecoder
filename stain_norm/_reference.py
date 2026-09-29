"""
Reference image utilities.

No default reference is bundled. Call build_reference() once from your Xenium
(or other) training slides, save the result, then pass the path to
normalize_slide() or batch_normalize().
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from ._readers import read_whole_slide, write_ome_tiff

PathLike = Union[str, Path]


def build_reference(
    slide_paths: list[PathLike],
    output_path: PathLike,
    target_mpp: float = 0.5,
) -> np.ndarray:
    """
    Build a reference image from training slides and save it as OME-TIFF.

    Reads each slide as a whole image at target_mpp, resizes to the median
    dimensions across all slides, and computes the pixel-wise mean.

    Args:
        slide_paths: Paths to training slides (e.g. all Xenium H&E slides).
        output_path: Where to save the reference OME-TIFF.
        target_mpp:  Resolution for reading slides (µm/pixel, default 0.5).

    Returns:
        The reference image as (H, W, 3) uint8 RGB.
    """
    from PIL import Image

    slide_paths = [Path(p) for p in slide_paths]
    if not slide_paths:
        raise ValueError("slide_paths must not be empty.")

    print(f"[INFO] Building reference from {len(slide_paths)} slides at {target_mpp} µm/px …")

    images: list[np.ndarray] = []
    widths: list[int] = []
    heights: list[int] = []

    for i, p in enumerate(slide_paths):
        print(f"  [{i+1}/{len(slide_paths)}] Reading {p.name} …")
        img, _ = read_whole_slide(p, target_mpp=target_mpp)
        images.append(img)
        heights.append(img.shape[0])
        widths.append(img.shape[1])

    # Use median dimensions as the common canvas size
    target_w = int(np.median(widths))
    target_h = int(np.median(heights))

    accum = np.zeros((target_h, target_w, 3), dtype=np.float64)
    for img in images:
        if img.shape[:2] != (target_h, target_w):
            pil = Image.fromarray(img)
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            img = np.array(pil)
        accum += img.astype(np.float64)

    mean_img = (accum / len(images)).clip(0, 255).astype(np.uint8)
    write_ome_tiff(mean_img, output_path, mpp=target_mpp)
    print(f"[INFO] Reference saved to {output_path}")
    return mean_img


def load_reference(path: PathLike) -> np.ndarray:
    """
    Load a reference image from disk (OME-TIFF, TIFF, PNG, or JPG).

    Args:
        path: Path to the reference image file.

    Returns:
        (H, W, 3) uint8 RGB array.

    Raises:
        FileNotFoundError: if the file does not exist.
        IOError:           if the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")

    suffix = path.suffix.lower()
    name = path.name.lower()

    if name.endswith(".ome.tiff") or name.endswith(".ome.tif") or suffix in {".tiff", ".tif"}:
        import tifffile
        data = tifffile.imread(str(path))
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=-1)
        elif data.ndim == 3 and data.shape[2] == 4:
            data = data[:, :, :3]
        elif data.ndim == 3 and data.shape[0] in {1, 3, 4}:
            # channel-first format
            data = np.moveaxis(data, 0, -1)
            if data.shape[2] == 4:
                data = data[:, :, :3]
            elif data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
        return data.astype(np.uint8)
    else:
        # PNG / JPG: use PIL to handle any format
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        return np.array(img)
