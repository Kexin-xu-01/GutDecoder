"""
High-level stain normalisation pipeline.

Typical workflow
----------------
1.  Build a reference once from your training slides::

        from gutdecoder.stain_norm import build_reference
        build_reference(
            slide_paths=[...],
            output_path="xenium_reference.tiff",
        )

2.  Normalise individual slides or whole directories::

        from gutdecoder.stain_norm import normalize_slide, batch_normalize
        normalize_slide(
            slide_path="sample.czi",
            output_path="sample_normalized.ome.tiff",
            reference="xenium_reference.tiff",
        )
        batch_normalize(
            input_dir="raw_slides/",
            output_dir="normalized_slides/",
            reference="xenium_reference.tiff",
        )
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import numpy as np

from ._readers import read_whole_slide, write_ome_tiff
from ._normalizers import get_normalizer, _GPU_METHODS
from ._reference import (
    build_reference as _build_reference,
    select_reference,
    load_reference,
)

PathLike = Union[str, Path]

_SUPPORTED_EXTENSIONS = (
    ".ome.tiff", ".ome.tif",
    ".tiff", ".tif",
    ".czi",
    ".ndpi", ".svs", ".mrxs", ".scn",
)


def normalize_slide(
    slide_path: PathLike,
    output_path: PathLike,
    method: str = "macenko",
    reference: PathLike | np.ndarray | None = None,
    target_mpp: float = 0.5,
    overwrite: bool = False,
    device: str = "auto",
) -> Path:
    """
    Normalise a single whole slide image and write the result as OME-TIFF.

    Args:
        slide_path:  Source slide (.ome.tiff, .tiff, .czi, .ndpi, …).
        output_path: Destination OME-TIFF path.
        method:      Normalisation algorithm. CPU: 'macenko' (default),
                     'reinhard', 'vahadane'. GPU (requires torchstain):
                     'macenko_gpu', 'vahadane_gpu', 'reinhard_gpu'.
        reference:   Path to a reference image, or a (H,W,3) uint8 numpy array.
                     Build one with build_reference() from your training slides.
        target_mpp:  Read/normalise resolution in µm/pixel (default 0.5 ≈ 20×).
        overwrite:   Re-normalise even if output already exists (default False).
        device:      PyTorch device for GPU methods — 'auto' (default, uses CUDA
                     when available), 'cpu', 'cuda', 'cuda:0', etc.
                     Ignored for CPU numpy methods.

    Returns:
        Path of the written OME-TIFF.

    Raises:
        ValueError:        if reference is None 
        FileNotFoundError: if the slide or reference file is not found.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    if reference is None:
        raise ValueError(
            "A reference image is required. Pass a file path or numpy array. "
            "Build one from your Xenium training slides with build_reference()."
        )

    ref_img = (
        load_reference(reference) if isinstance(reference, (str, Path))
        else np.asarray(reference)
    )

    normaliser = get_normalizer(method, device=device)
    normaliser.fit(ref_img)

    slide_img, mpp = read_whole_slide(slide_path, target_mpp=target_mpp)
    normalised = normaliser.transform(slide_img)

    write_ome_tiff(normalised, output_path, mpp=mpp)
    return output_path


# ---------------------------------------------------------------------------
# Worker for parallel batch processing
# ---------------------------------------------------------------------------

def _normalize_one(args: tuple) -> tuple[Path | None, str | None]:
    """
    Worker function for ProcessPoolExecutor.
    Returns (output_path, None) on success, or (None, error_msg) on failure.
    """
    slide_path, output_path, method, reference_path, target_mpp, overwrite, device = args
    try:
        out = normalize_slide(
            slide_path=slide_path,
            output_path=output_path,
            method=method,
            reference=reference_path,
            target_mpp=target_mpp,
            overwrite=overwrite,
            device=device,
        )
        return out, None
    except Exception as exc:
        return None, f"{slide_path}: {exc}"


# ---------------------------------------------------------------------------
# Public batch API
# ---------------------------------------------------------------------------

def batch_normalize(
    input_dir: PathLike,
    output_dir: PathLike,
    method: str = "macenko",
    reference: PathLike | np.ndarray | None = None,
    extensions: list[str] | None = None,
    target_mpp: float = 0.5,
    n_workers: int = 4,
    overwrite: bool = False,
    device: str = "auto",
) -> list[Path]:
    """
    Normalise all slides in input_dir and write OME-TIFFs to output_dir.

    Args:
        input_dir:   Directory containing slides.
        output_dir:  Directory for normalised OME-TIFFs (created if absent).
        method:      Normalisation algorithm. CPU: 'macenko' (default),
                     'reinhard', 'vahadane'. GPU (requires torchstain):
                     'macenko_gpu', 'vahadane_gpu', 'reinhard_gpu'.
        reference:   Path to reference image or (H,W,3) uint8 numpy array.
                     **Required**.
        extensions:  File extensions to include (default: all supported formats).
        target_mpp:  Resolution in µm/pixel (default 0.5).
        n_workers:   Number of parallel worker processes (default 4).
                     Automatically set to 1 when using a GPU method, because
                     CUDA contexts cannot be forked across processes.
        overwrite:   Re-normalise existing outputs (default False).
        device:      PyTorch device for GPU methods — 'auto' (default), 'cpu',
                     'cuda', 'cuda:0', etc. Ignored for CPU numpy methods.

    Returns:
        List of Paths for successfully written output files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if extensions is None:
        extensions = list(_SUPPORTED_EXTENSIONS)

    # CUDA contexts cannot be forked into subprocesses — run single-threaded
    effective_workers = n_workers
    if method.lower() in _GPU_METHODS and n_workers > 1:
        print(
            f"[INFO] GPU method '{method}' — setting n_workers=1 "
            "(CUDA contexts cannot be forked across processes)."
        )
        effective_workers = 1

    # Collect slides, handling compound extensions (.ome.tiff)
    slides: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(name.endswith(ext.lower()) for ext in extensions):
            slides.append(p)

    if not slides:
        print(f"[INFO] No matching slides found in {input_dir}")
        return []

    print(f"[INFO] Found {len(slides)} slides to normalise.")

    # If reference is a numpy array, write to a temp file so workers can load it
    _tmp_ref: str | None = None
    if isinstance(reference, np.ndarray):
        import tifffile
        fd, _tmp_ref = tempfile.mkstemp(suffix=".tiff")
        os.close(fd)
        tifffile.imwrite(_tmp_ref, reference)
        reference_path: PathLike = _tmp_ref
    else:
        reference_path = reference

    tasks = [
        (
            slide,
            output_dir / (slide.stem + ".ome.tiff"),
            method,
            reference_path,
            target_mpp,
            overwrite,
            device,
        )
        for slide in slides
    ]

    results: list[Path] = []
    errors: list[str] = []

    try:
        if effective_workers <= 1:
            for task in tasks:
                out, err = _normalize_one(task)
                if out is not None:
                    results.append(out)
                else:
                    errors.append(err)
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as pool:
                futures = {pool.submit(_normalize_one, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    out, err = fut.result()
                    if out is not None:
                        results.append(out)
                    else:
                        errors.append(err)
    finally:
        if _tmp_ref and os.path.exists(_tmp_ref):
            os.unlink(_tmp_ref)

    print(f"[INFO] Normalised {len(results)}/{len(slides)} slides.")
    if errors:
        print(f"[WARN] {len(errors)} slide(s) failed:")
        for e in errors:
            print(f"  {e}")

    return results


def build_reference(
    slide_paths=None,
    output_path: PathLike | None = None,
    target_mpp: float = 0.5,
    ratio_mpp: float = 4.0,
    *,
    slide_path=None,  # alias for slide_paths (single-file convenience)
) -> Path:
    """
    Select the best reference slide from training data and save it as OME-TIFF.

    The reference is chosen by the red-to-blue (R/B) channel mean intensity
    ratio criterion: the slide whose R/B ratio is closest to 1.0 represents
    the most balanced H&E staining and is selected as the reference.

    Args:
        slide_paths: A directory of slides, a list of slide paths, or a single
                     slide path. Also accepts the keyword alias ``slide_path``
                     (singular) for the single-file case.
        output_path: Where to save the reference OME-TIFF.
        target_mpp:  Resolution at which to save the reference (µm/pixel,
                     default 0.5 ≈ 20×).
        ratio_mpp:   Resolution for the R/B screening pass (default 4.0 µm/px).
                     Ignored when only one slide is supplied.

    Returns:
        Path of the saved reference image.

    Examples::

        # Directory — auto-selects best slide by R/B ratio
        build_reference("/path/to/xenium_slides/", output_path="ref.tiff")

        # Already-selected slide
        path, ratio = select_reference("/path/to/xenium_slides/")
        build_reference(slide_path=path, output_path="ref.tiff")
    """
    src = slide_path if slide_paths is None else slide_paths
    if src is None:
        raise ValueError("Provide slide_paths (or slide_path) and output_path.")
    if output_path is None:
        raise ValueError("output_path is required.")

    output_path = Path(output_path)
    _build_reference(src, output_path, target_mpp=target_mpp, ratio_mpp=ratio_mpp)
    return output_path
