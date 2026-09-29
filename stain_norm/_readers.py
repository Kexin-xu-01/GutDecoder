"""
Multi-format whole-slide image reader.

Supported formats
-----------------
- ome.tiff / ome.tif / tiff / tif  → tiffslide
- ndpi / svs / mrxs / scn          → openslide
- czi                               → aicsimageio
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]

_OPENSLIDE_EXTS = frozenset({".ndpi", ".svs", ".mrxs", ".scn", ".bif", ".vms", ".vmu"})


def _get_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".ome.tiff") or name.endswith(".ome.tif"):
        return "tiff"
    suffix = path.suffix.lower()
    if suffix in {".tiff", ".tif"}:
        return "tiff"
    if suffix in _OPENSLIDE_EXTS:
        return "openslide"
    if suffix == ".czi":
        return "czi"
    raise ValueError(
        f"Unsupported slide format '{path.suffix}'. "
        "Supported: .ome.tiff, .ome.tif, .tiff, .tif, .czi, "
        ".ndpi, .svs, .mrxs, .scn"
    )


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------

def _read_tiff(path: Path, target_mpp: float) -> tuple[np.ndarray, float]:
    import tiffslide

    slide = tiffslide.TiffSlide(str(path))
    try:
        mpp_x = (
            float(slide.properties.get("tiffslide.mpp-x") or
                  slide.properties.get("openslide.mpp-x") or 0)
        )
        if mpp_x <= 0:
            mpp_x = 0.5

        downsample = max(1.0, target_mpp / mpp_x)
        level = slide.get_best_level_for_downsample(downsample)
        level_dims = slide.level_dimensions[level]  # (W, H)

        region = slide.read_region((0, 0), level, level_dims)
        img = np.array(region.convert("RGB"))
        actual_mpp = mpp_x * slide.level_downsamples[level]
    finally:
        slide.close()
    return img, actual_mpp


def _read_openslide(path: Path, target_mpp: float) -> tuple[np.ndarray, float]:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        mpp_str = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_x = float(mpp_str) if mpp_str else 0.5
        if mpp_x <= 0:
            mpp_x = 0.5

        downsample = max(1.0, target_mpp / mpp_x)
        level = slide.get_best_level_for_downsample(downsample)
        level_dims = slide.level_dimensions[level]  # (W, H)

        region = slide.read_region((0, 0), level, level_dims)
        img = np.array(region.convert("RGB"))
        actual_mpp = mpp_x * slide.level_downsamples[level]
    finally:
        slide.close()
    return img, actual_mpp


def _read_czi(path: Path, target_mpp: float) -> tuple[np.ndarray, float]:
    try:
        from aicsimageio import AICSImage
    except ImportError as e:
        raise ImportError(
            "aicsimageio is required to read CZI files. "
            "Install with: pip install aicsimageio[czi]"
        ) from e

    img_obj = AICSImage(str(path))
    mpp_x = float(img_obj.physical_pixel_sizes.X or 0.5)
    if mpp_x <= 0:
        mpp_x = 0.5

    # aicsimageio uses STCZYX ordering by default; we want Y x X x C
    arr = img_obj.get_image_data("YXC", S=0, T=0, Z=0)  # H x W x C

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    img = arr.astype(np.uint8)
    return img, mpp_x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_whole_slide(
    path: PathLike, target_mpp: float = 0.5
) -> tuple[np.ndarray, float]:
    """
    Read an entire slide at ~target_mpp resolution.

    Args:
        path:       Slide file path (.ome.tiff, .tiff, .ndpi, .czi, …).
        target_mpp: Target resolution in µm/pixel (default 0.5 ≈ 20×).

    Returns:
        Tuple of (image, actual_mpp) where image is (H, W, 3) uint8 RGB
        and actual_mpp is the effective resolution used.
    """
    path = Path(path)
    fmt = _get_format(path)
    if fmt == "tiff":
        return _read_tiff(path, target_mpp)
    if fmt == "openslide":
        return _read_openslide(path, target_mpp)
    if fmt == "czi":
        return _read_czi(path, target_mpp)
    raise ValueError(f"Unrecognised format: {fmt}")  # unreachable


def get_slide_mpp(path: PathLike) -> float:
    """Return the native MPP (µm/pixel) of a slide."""
    path = Path(path)
    fmt = _get_format(path)

    if fmt == "tiff":
        import tiffslide
        slide = tiffslide.TiffSlide(str(path))
        mpp = float(slide.properties.get("tiffslide.mpp-x") or
                    slide.properties.get("openslide.mpp-x") or 0.5)
        slide.close()
        return mpp if mpp > 0 else 0.5

    if fmt == "openslide":
        import openslide
        slide = openslide.OpenSlide(str(path))
        mpp_str = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        slide.close()
        return float(mpp_str) if mpp_str else 0.5

    if fmt == "czi":
        from aicsimageio import AICSImage
        img = AICSImage(str(path))
        mpp = float(img.physical_pixel_sizes.X or 0.5)
        return mpp if mpp > 0 else 0.5

    return 0.5


def write_ome_tiff(array: np.ndarray, path: PathLike, mpp: float) -> None:
    """
    Write a (H, W, 3) uint8 RGB image as a tiled OME-TIFF.

    The file is LZW-compressed and stores resolution metadata so downstream
    tools (Trident, QuPath, napari) can read the correct pixel spacing.

    Args:
        array: (H, W, 3) uint8 RGB image.
        path:  Output file path (created with parent dirs as needed).
        mpp:   Microns per pixel to embed in the file.
    """
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    H, W = array.shape[:2]
    px_per_cm = 1e4 / mpp  # 10,000 µm/cm ÷ mpp µm/px = px/cm

    ome_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f'<Image ID="Image:0" Name="{path.stem}">'
        '<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8"'
        f' SizeX="{W}" SizeY="{H}" SizeZ="1" SizeC="3" SizeT="1"'
        f' PhysicalSizeX="{mpp:.6f}" PhysicalSizeXUnit="µm"'
        f' PhysicalSizeY="{mpp:.6f}" PhysicalSizeYUnit="µm">'
        '<Channel ID="Channel:0:0" SamplesPerPixel="3"/>'
        '</Pixels></Image></OME>'
    )

    tifffile.imwrite(
        str(path),
        array,
        photometric="rgb",
        description=ome_xml,
        resolution=(px_per_cm, px_per_cm),
        resolutionunit=tifffile.RESUNIT.CENTIMETER,
        compression="lzw",
        tile=(256, 256),
    )
