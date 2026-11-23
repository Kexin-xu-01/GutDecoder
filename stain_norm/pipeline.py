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

import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

from typing import Union


import numpy as np


from loguru import logger

from ._readers import SLIDE_EXTENSIONS, write_ome_tiff
from ._readers_tiled import SlideSource, sample_slide, write_tiled_ome_tiff

from ._normalizers import get_normalizer, _GPU_METHODS

from ._reference import (

    build_reference as _build_reference,

    select_reference,

    load_reference,

)


PathLike = Union[str, Path]


_SUPPORTED_EXTENSIONS = SLIDE_EXTENSIONS


def normalize_slide(

    slide_path: PathLike,

    output_path: PathLike,

    method: str = "macenko",

    reference: PathLike | np.ndarray | None = None,

    target_mpp: float = 0.5,

    overwrite: bool = False,

    device: str = "auto",
    memory_limit_gb: float | None = 8.0,
    overlap: int = 128,
    max_patch_px: int | None = None,

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


    with SlideSource(slide_path, target_mpp=target_mpp) as slide:
        # Source parameters come from one slide-wide sample, so the in-memory and tiled
        # paths below produce the same output.
        normaliser.fit_source(sample_slide(slide))
        # Conservative estimate of peak working memory (not just RGB bytes).
        estimated_bytes = slide.width * slide.height * 64
        budget = float('inf') if memory_limit_gb is None else memory_limit_gb * 1024**3
        if estimated_bytes <= budget:
            normalised = normaliser.transform(slide.read(0, 0, slide.width, slide.height))
            write_ome_tiff(normalised, output_path, mpp=slide.mpp)
        else:
            if method.lower() in _GPU_METHODS:
                logger.warning(
                    f"{Path(slide_path).name} is too large to normalise in memory and GPU method "
                    f"'{method}' re-estimates stains per tile, so tile seams are possible. "
                    "Use the CPU method or raise memory_limit_gb for seam-free output."
                )
            write_tiled_ome_tiff(
                slide, output_path, normaliser.transform,
                memory_limit_gb=memory_limit_gb,
                overlap=overlap,
                max_patch_px=max_patch_px,
            )

    return output_path


def _output_name(slide: Path) -> str:
    """Output file name for a slide: its name without the slide extension, plus .ome.tiff."""
    name = slide.name
    for ext in (".ome.tiff", ".ome.tif"):
        if name.lower().endswith(ext):
            return name[: -len(ext)] + ".ome.tiff"
    return slide.stem + ".ome.tiff"


def _output_collisions(slides: list[Path]) -> dict[str, list[Path]]:
    """Output names shared by more than one slide, e.g. a.czi and a.svs."""
    groups: dict[str, list[Path]] = {}
    for slide in slides:
        groups.setdefault(_output_name(slide), []).append(slide)
    return {name: group for name, group in groups.items() if len(group) > 1}


# Worker for parallel batch processing


def _normalize_one(args: tuple) -> tuple[Path | None, str | None]:

    """

    Worker function for ProcessPoolExecutor.

    Returns (output_path, None) on success, or (None, error_msg) on failure.

    """

    slide_path, output_path, method, reference_path, target_mpp, overwrite, device, memory_limit_gb, overlap, max_patch_px = args

    try:

        out = normalize_slide(

            slide_path=slide_path,

            output_path=output_path,

            method=method,

            reference=reference_path,

            target_mpp=target_mpp,

            overwrite=overwrite,

            device=device,
            memory_limit_gb=memory_limit_gb,
            overlap=overlap,
            max_patch_px=max_patch_px,

        )

        return out, None

    except Exception as exc:

        return None, f"{slide_path}: {exc}\n{traceback.format_exc()}"


# Public batch API


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
    memory_limit_gb: float | None = 8.0,
    overlap: int = 128,
    max_patch_px: int | None = None,

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

        logger.info(

            f"GPU method '{method}' — setting n_workers=1 "

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

        logger.info(f"No matching slides found in {input_dir}")

        return []


    logger.info(f"Found {len(slides)} slides to normalise.")


    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("output_dir must differ from input_dir, or outputs could overwrite source slides.")
    collisions = _output_collisions(slides)
    if collisions:
        raise ValueError(
            "These slides would write to the same output file; rename or move them first:\n"
            + "\n".join(f"  {name}: {', '.join(p.name for p in group)}" for name, group in collisions.items())
        )

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

            output_dir / _output_name(slide),

            method,

            reference_path,

            target_mpp,

            overwrite,

            device,
            memory_limit_gb,
            overlap,
            max_patch_px,

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


    logger.info(f"Normalised {len(results)}/{len(slides)} slides.")

    if errors:

        logger.warning(f"{len(errors)} slide(s) failed:")

        for e in errors:

            logger.info(f"  {e}")


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
