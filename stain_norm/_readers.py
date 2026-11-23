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
from loguru import logger

PathLike = Union[str, Path]

_OPENSLIDE_EXTS = (".ndpi", ".svs", ".mrxs", ".scn", ".bif", ".vms", ".vmu")
SLIDE_EXTENSIONS = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi") + _OPENSLIDE_EXTS


def _get_format(path: Path) -> str:
    """'tiff', 'openslide' or 'czi' for a slide path; raises for anything else."""
    suffix = path.suffix.lower()  # .ome.tif(f) ends in .tif(f)
    if suffix in {".tiff", ".tif"}:
        return "tiff"
    if suffix in _OPENSLIDE_EXTS:
        return "openslide"
    if suffix == ".czi":
        return "czi"
    raise ValueError(f"Unsupported slide format '{path.suffix}'. Supported: {', '.join(SLIDE_EXTENSIONS)}")


def to_rgb_uint8(a) -> np.ndarray:
    """
    (H, W, 3) uint8 RGB from grey, channel-first, RGBA or higher bit-depth arrays.
    Integer data is scaled by its dtype's maximum (uint16 65535 -> 255); floats are taken as 0-1.
    """
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.shape[-1] >= 4:
        a = a[..., :3]
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.integer):
        return (a.astype(np.float32) * (255.0 / np.iinfo(a.dtype).max)).clip(0, 255).astype(np.uint8)
    return (a.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)


# Format-specific readers

GUESSED_MPP = 0.5


def _mpp_or_guess(mpp, path: Path) -> float:
    """Return the stored MPP, or GUESSED_MPP with a warning if the metadata has no usable value."""
    if mpp is None or not np.isfinite(float(mpp)) or float(mpp) <= 0:
        logger.warning(
            f"No usable MPP metadata in {path} (got {mpp!r}); guessing {GUESSED_MPP} um/px. "
            "The slide will be processed at the wrong scale if that guess is wrong."
        )
        return GUESSED_MPP
    return float(mpp)


def _mpp_from_tifffile(tf_page):
    """MPP from a tifffile page's XResolution/ResolutionUnit tags, or None if they are absent."""
    res_unit = tf_page.tags.get("ResolutionUnit")
    x_res = tf_page.tags.get("XResolution")
    if res_unit is None or x_res is None:
        return None
    num, den = x_res.value if isinstance(x_res.value, tuple) else (x_res.value, 1)
    px_per_unit = num / (den or 1)
    um_per_unit = {2: 25400.0, 3: 1e4}.get(int(res_unit.value))  # 2 = inch, 3 = centimetre
    if um_per_unit is None or px_per_unit <= 0:
        return None
    return um_per_unit / px_per_unit


def _tiff_stored_mpp(path: Path, slide=None):
    """MPP stored in a TIFF: tiffslide's parsed metadata if available, else the resolution tags."""
    if slide is not None:
        mpp = slide.properties.get("tiffslide.mpp-x") or slide.properties.get("openslide.mpp-x")
        if mpp is not None:
            return mpp
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        return _mpp_from_tifffile(tf.pages[0])


def _read_tiff_fallback(path: Path, target_mpp: float) -> tuple[np.ndarray, float]:
    """Read a flat (non-pyramidal) TIFF with tifffile + PIL, then downsample."""
    import tifffile
    from PIL import Image

    with tifffile.TiffFile(str(path)) as tf:
        mpp_x = _mpp_or_guess(_mpp_from_tifffile(tf.pages[0]), path)
        data = tf.asarray()

    data = to_rgb_uint8(data)

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

    with slide:
        mpp_x = _mpp_or_guess(_tiff_stored_mpp(path, slide), path)
        try:
            downsample = max(1.0, target_mpp / mpp_x)
            level = slide.get_best_level_for_downsample(downsample)
            level_dims = slide.level_dimensions[level]  # (W, H)

            region = slide.read_region((0, 0), level, level_dims)
            img = np.array(region.convert("RGB"))
            actual_mpp = mpp_x * slide.level_downsamples[level]
        except Exception:
            return _read_tiff_fallback(path, target_mpp)
    return img, actual_mpp


def _read_openslide(path: Path, target_mpp: float) -> tuple[np.ndarray, float]:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        mpp_x = _mpp_or_guess(slide.properties.get(openslide.PROPERTY_NAME_MPP_X), path)

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
        from ._czi import open_czi
        img_obj = open_czi(path)
    except ImportError as e:
        raise ImportError(
            "aicsimageio is required to read CZI files. "
            "Install with: pip install aicsimageio[czi]"
        ) from e

    mpp_x = _mpp_or_guess(img_obj.physical_pixel_sizes.X, path)

    # RGB CZIs store colour in the Samples dim "S" (C has size 1);
    # scenes are selected via set_scene(), not an "S" index.
    if "S" in img_obj.dims.order:
        arr = img_obj.get_image_data("YXS", T=0, C=0, Z=0)  # H x W x 3
    else:
        arr = img_obj.get_image_data("YXC", T=0, Z=0)  # H x W x C

    return to_rgb_uint8(arr), mpp_x


# Public API

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
            with tiffslide.TiffSlide(str(path)) as slide:
                mpp = _tiff_stored_mpp(path, slide)
        except Exception:
            # Flat TIFFs that tiffslide cannot open: use the resolution tags.
            mpp = _tiff_stored_mpp(path)
        return _mpp_or_guess(mpp, path)

    if fmt == "openslide":
        import openslide
        with openslide.OpenSlide(str(path)) as slide:
            return _mpp_or_guess(slide.properties.get(openslide.PROPERTY_NAME_MPP_X), path)

    # czi
    from aicsimageio import AICSImage
    return _mpp_or_guess(AICSImage(str(path)).physical_pixel_sizes.X, path)


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
