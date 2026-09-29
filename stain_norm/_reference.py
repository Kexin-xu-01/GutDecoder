"""
Reference image utilities.

No default reference is bundled. Call build_reference() once from your Xenium
(or other) training slides, save the result, then pass the path to
normalize_slide() or batch_normalize().

Reference selection criterion
------------------------------
The reference sample is chosen computationally using the red-to-blue (R/B)
channel mean intensity ratio. The slide whose R/B ratio is closest to 1.0 is
selected as the reference. A ratio near 1 indicates balanced hematoxylin (blue)
and eosin (red) staining — the most "typical" stain appearance in the set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from ._readers import read_whole_slide, write_ome_tiff

PathLike = Union[str, Path]
SlideSrc = Union[PathLike, list[PathLike]]  # directory, single file, or list of files

# Low resolution used only for computing the R/B ratio (fast thumbnail pass)
_RATIO_MPP = 4.0  # ~2.5× magnification; adequate for per-channel mean

_SLIDE_EXTS = (
    ".ome.tiff", ".ome.tif", ".tiff", ".tif",
    ".czi", ".ndpi", ".svs", ".mrxs", ".scn",
)


def _collect_slides(source: SlideSrc) -> list[Path]:
    """
    Normalise the slide source into a flat list of Path objects.

    Accepts:
    - a directory path  → all slide files found inside it
    - a single file path → [that file]
    - a list of paths   → those paths
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            slides = []
            for f in sorted(p.iterdir()):
                if f.is_file() and any(
                    f.name.lower().endswith(ext) for ext in _SLIDE_EXTS
                ):
                    slides.append(f)
            if not slides:
                raise FileNotFoundError(
                    f"No supported slide files found in directory: {p}\n"
                    f"Supported extensions: {_SLIDE_EXTS}"
                )
            return slides
        else:
            return [p]
    else:
        return [Path(p) for p in source]


def _rb_ratio(image: np.ndarray) -> float:
    """Return mean(R) / mean(B) for a (H, W, 3) uint8 RGB image."""
    mean_r = float(image[:, :, 0].mean())
    mean_b = float(image[:, :, 2].mean())
    return mean_r / (mean_b + 1e-6)


def select_reference(
    slide_paths: SlideSrc,
    ratio_mpp: float = _RATIO_MPP,
) -> tuple[Path, float]:
    """
    Select the slide whose red-to-blue channel mean intensity ratio is closest to 1.

    Reads each slide at a low resolution (ratio_mpp) to compute per-channel means
    efficiently, then returns the path of the best candidate.

    Args:
        slide_paths: A directory of slides, a single slide path, or a list of
                     slide paths. Directories are scanned for all supported
                     slide formats automatically.
        ratio_mpp:   Resolution for thumbnail reads when computing ratios
                     (default 4.0 µm/px — fast, sufficient for mean intensities).

    Returns:
        Tuple of (selected_path, rb_ratio) for the chosen slide.
    """
    slide_paths = _collect_slides(slide_paths)
    if not slide_paths:
        raise ValueError("No slides found.")

    print(
        f"[INFO] Computing R/B ratios for {len(slide_paths)} slides "
        f"at {ratio_mpp} µm/px …"
    )

    best_path: Path | None = None
    best_ratio: float = float("inf")
    best_distance: float = float("inf")

    for i, p in enumerate(slide_paths):
        thumb, _ = read_whole_slide(p, target_mpp=ratio_mpp)
        ratio = _rb_ratio(thumb)
        distance = abs(ratio - 1.0)
        print(f"  [{i+1}/{len(slide_paths)}] {p.name}  R/B = {ratio:.4f}")
        if distance < best_distance:
            best_distance = distance
            best_ratio = ratio
            best_path = p

    print(
        f"[INFO] Selected reference: {best_path.name}  "  # type: ignore[union-attr]
        f"(R/B = {best_ratio:.4f}, |ratio-1| = {best_distance:.4f})"
    )
    return best_path, best_ratio  # type: ignore[return-value]


def build_reference(
    slide_paths: SlideSrc,
    output_path: PathLike,
    target_mpp: float = 0.5,
    ratio_mpp: float = _RATIO_MPP,
) -> np.ndarray:
    """
    Select the best reference slide and save it as an OME-TIFF.

    The reference is selected by finding the slide whose red-to-blue channel
    mean intensity ratio is closest to 1.0 (most balanced H&E staining).
    That slide is then read at target_mpp and saved as the reference image.

    Args:
        slide_paths: A directory of slides, a single slide path, or a list of
                     slide paths. If a single file is passed, it is used directly
                     as the reference without running the R/B selection.
        output_path: Where to save the reference OME-TIFF.
        target_mpp:  Resolution at which to save the reference (µm/pixel,
                     default 0.5 ≈ 20×).
        ratio_mpp:   Resolution used for the R/B ratio screening pass
                     (default 4.0 µm/px, much faster than target_mpp).
                     Ignored when a single file is passed.

    Returns:
        The selected reference image as (H, W, 3) uint8 RGB.
    """
    slides = _collect_slides(slide_paths)

    if len(slides) == 1:
        selected_path = slides[0]
        print(f"[INFO] Single slide provided — using directly: {selected_path.name}")
    else:
        selected_path, _ = select_reference(slides, ratio_mpp=ratio_mpp)

    print(f"[INFO] Reading reference slide at {target_mpp} µm/px …")
    img, mpp = read_whole_slide(selected_path, target_mpp=target_mpp)
    write_ome_tiff(img, output_path, mpp=mpp)
    print(f"[INFO] Reference saved to {output_path}")
    return img


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
