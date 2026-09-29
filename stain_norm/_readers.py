"""
Multi-format whole-slide image reader.

Supported formats
-----------------
- ome.tiff / ome.tif / tiff / tif  → tiffslide (pyramidal WSI) or tifffile+PIL (flat TIFF)
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

def _mpp_from_tifffile(tf_page) -> float:
    """Extract MPP from a tifffile page's tags (XResolution in cm or inch)."""
    try:
        import tifffile
        res_unit = tf_page.tags.get("ResolutionUnit")
        x_res = tf_page.tags.get("XResolution")
        if res_unit is None or x_res is None:
            return 0.0
        unit_val = res_unit.value
        num, den = x_res.value if isinstance(x_res.value, tuple) else (x_res.value, 1)
        px_per_unit = num / (den or 1)
        if hasattr(unit_val, "value"):
            unit_val = unit_val.value
        if unit_val == 3:       # centimetre
            return 1e4 / px_per_unit   # µm/px
        elif unit_val == 2:     # inch
            return 25400.0 / px_per_unit
    except Exception:
        pass
    return 0.0


def _read_tiff_fallback(path: Path, target_mpp: float) -> tuple[np.ndarray, float]:
    """Read a flat (non-pyramidal) TIFF with tifffile + PIL, then downsample."""
    import tifffile
    from PIL import Image

    with tifffile.TiffFile(str(path)) as tf:
        mpp_x = _mpp_from_tifffile(tf.pages[0])
        if mpp_x <= 0:
            mpp_x = 0.5
        data = tf.asarray()

    # Normalise to (H, W, 3) uint8 RGB
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3 and data.shape[0] in {1, 3, 4}:
        data = np.moveaxis(data, 0, -1)
    if data.shape[2] == 4:
        data = data[:, :, :3]
    elif data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)

    if data.dtype != np.uint8:
        mx = data.max()
        if mx == 0:
            data = data.astype(np.uint8)
        elif mx > 255:
            # uint16 or higher bit-depth — scale to full 8-bit range
            data = (data.astype(np.float32) / mx * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)

    # Downsample to target_mpp if needed
    actual_mpp = mpp_x
    if target_mpp > mpp_x + 1e-3:
        scale = mpp_x / target_mpp
        new_w = max(1, int(data.shape[1] * scale))
        new_h = max(1, int(data.shape[0] * scale))
        data = np.array(Image.fromarray(data).resize((new_w, new_h), Image.LANCZOS))
        actual_mpp = target_mpp

    return data, actual_mpp


def _read_tiff(path: Path, target_mpp: float) -> tuple[np.ndarray, float]:
    try:
        import tiffslide
    except (ImportError, ModuleNotFoundError):
        return _read_tiff_fallback(path, target_mpp)

    try:
        slide = tiffslide.TiffSlide(str(path))
    except Exception:
        # tiffslide imported but failed to open (e.g. flat TIFF) — use fallback
        return _read_tiff_fallback(path, target_mpp)

    try:
        mpp_x = float(
            slide.properties.get("tiffslide.mpp-x") or
            slide.properties.get("openslide.mpp-x") or 0
        )
        if mpp_x <= 0:
            mpp_x = 0.5

        downsample = max(1.0, target_mpp / mpp_x)
        level = slide.get_best_level_for_downsample(downsample)
        level_dims = slide.level_dimensions[level]  # (W, H)

        region = slide.read_region((0, 0), level, level_dims)
        img = np.array(region.convert("RGB"))
        actual_mpp = mpp_x * slide.level_downsamples[level]
    except Exception:
        slide.close()
        return _read_tiff_fallback(path, target_mpp)
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
        try:
            import tiffslide
            slide = tiffslide.TiffSlide(str(path))
            mpp = float(slide.properties.get("tiffslide.mpp-x") or
                        slide.properties.get("openslide.mpp-x") or 0.0)
            slide.close()
            if mpp > 0:
                return mpp
        except Exception:
            pass
        # Fallback: read MPP from tifffile resolution tags
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            mpp = _mpp_from_tifffile(tf.pages[0])
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
        f' PhysicalSizeX="{mpp:.6f}" PhysicalSizeXUnit="&#xb5;m"'
        f' PhysicalSizeY="{mpp:.6f}" PhysicalSizeYUnit="&#xb5;m">'
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
