from pathlib import Path

import numpy as np
import pytest
import tifffile

from gutdecoder.stain_norm import batch_normalize, normalize_slide
from gutdecoder.stain_norm._normalizers import get_normalizer
from gutdecoder.stain_norm.pipeline import _output_name

MPP = 0.5


def _he_like(seed, nuclei, stroma, shape=(160, 200)):
    """White background with blobs of a nuclear and a stromal colour, plus noise."""
    rng = np.random.default_rng(seed)
    img = np.full((*shape, 3), 245.0)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for colour in [stroma] * 12 + [nuclei] * 30:
        cy, cx, r = rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(4, 25)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = colour
    img += rng.normal(0, 6, img.shape)
    return img.clip(0, 255).astype(np.uint8)


@pytest.fixture
def slide(tmp_path):
    path = tmp_path / "slide.tif"
    tifffile.imwrite(path, _he_like(0, nuclei=(90, 50, 140), stroma=(225, 140, 185)), photometric="rgb",
                     resolution=(1e4 / MPP, 1e4 / MPP), resolutionunit="CENTIMETER")
    return path


REFERENCE = _he_like(1, nuclei=(70, 40, 120), stroma=(240, 170, 200))


@pytest.mark.parametrize("method", ["reinhard", "macenko", "vahadane"])
def test_in_memory_and_tiled_output_match(slide, tmp_path, method):
    whole = normalize_slide(slide, tmp_path / "whole.ome.tiff", method=method,
                            reference=REFERENCE, target_mpp=MPP, memory_limit_gb=None)
    # A tiny budget forces the tiled path with 64 px patches.
    tiled = normalize_slide(slide, tmp_path / "tiled.ome.tiff", method=method,
                            reference=REFERENCE, target_mpp=MPP, memory_limit_gb=0.001,
                            overlap=16, max_patch_px=64)

    a, b = tifffile.imread(whole).astype(int), tifffile.imread(tiled).astype(int)
    assert a.shape == b.shape
    # Per-pixel arithmetic is identical; allow 1 level of float rounding in the uint8 cast.
    assert np.abs(a - b).max() <= 1


@pytest.mark.parametrize("method", ["reinhard", "macenko", "vahadane"])
def test_transform_requires_source_fit(method):
    normaliser = get_normalizer(method).fit(REFERENCE)

    with pytest.raises(RuntimeError, match="fit_source"):
        normaliser.transform(REFERENCE)


@pytest.mark.parametrize("name, expected", [
    ("a.svs", "a.ome.tiff"),
    ("a.ome.tiff", "a.ome.tiff"),
    ("a.OME.TIF", "a.ome.tiff"),
    ("a.b.czi", "a.b.ome.tiff"),
])
def test_output_name_strips_compound_extensions(name, expected):
    assert _output_name(Path(name)) == expected


def test_batch_refuses_slides_that_share_an_output_name(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.czi").touch()
    (src / "a.svs").touch()

    with pytest.raises(ValueError, match="a.ome.tiff: a.czi, a.svs"):
        batch_normalize(src, tmp_path / "out", reference=REFERENCE)


def test_batch_refuses_writing_into_the_input_dir(tmp_path):
    (tmp_path / "a.ome.tiff").touch()

    with pytest.raises(ValueError, match="output_dir must differ"):
        batch_normalize(tmp_path, tmp_path, reference=REFERENCE)
