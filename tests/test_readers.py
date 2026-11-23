import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from loguru import logger

from gutdecoder.stain_norm._readers import GUESSED_MPP, _get_format, get_slide_mpp, read_whole_slide, to_rgb_uint8
from gutdecoder.stain_norm._readers_tiled import SlideSource

IMG = np.full((64, 64, 3), 200, dtype=np.uint8)


@pytest.fixture
def warnings_log():
    messages = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


def test_tiff_with_resolution_tags_uses_stored_mpp(tmp_path, warnings_log):
    path = tmp_path / "with_mpp.tif"
    tifffile.imwrite(path, IMG, photometric="rgb", resolution=(1e4 / 0.25, 1e4 / 0.25), resolutionunit="CENTIMETER")

    assert get_slide_mpp(path) == pytest.approx(0.25)
    img, mpp = read_whole_slide(path, target_mpp=0.25)
    assert img.shape == IMG.shape
    assert mpp == pytest.approx(0.25)
    assert warnings_log == []


def test_tiff_without_mpp_warns_and_guesses(tmp_path, warnings_log):
    path = tmp_path / "no_mpp.tif"
    tifffile.imwrite(path, IMG, photometric="rgb")

    assert get_slide_mpp(path) == GUESSED_MPP
    _, mpp = read_whole_slide(path, target_mpp=GUESSED_MPP)
    assert mpp == GUESSED_MPP
    assert len(warnings_log) == 2
    assert all("guessing 0.5" in m and "no_mpp.tif" in m for m in warnings_log)


def test_tiled_reader_uses_stored_mpp(tmp_path, warnings_log):
    path = tmp_path / "with_mpp.tif"
    tifffile.imwrite(path, IMG, photometric="rgb", resolution=(1e4 / 0.25, 1e4 / 0.25), resolutionunit="CENTIMETER")

    with SlideSource(path, target_mpp=0.25) as slide:
        assert slide.native_mpp == pytest.approx(0.25)
    assert warnings_log == []


def test_tiled_reader_warns_and_guesses_without_mpp(tmp_path, warnings_log):
    path = tmp_path / "no_mpp.tif"
    tifffile.imwrite(path, IMG, photometric="rgb")

    with SlideSource(path, target_mpp=GUESSED_MPP) as slide:
        assert slide.native_mpp == GUESSED_MPP
    assert len(warnings_log) == 1
    assert "guessing 0.5" in warnings_log[0]


@pytest.mark.parametrize("arr, expected", [
    (np.full((2, 2), 7, np.uint8), [7, 7, 7]),                        # grey
    (np.full((3, 2, 2), 9, np.uint8), [9, 9, 9]),                     # channel-first
    (np.full((2, 2, 4), 5, np.uint8), [5, 5, 5]),                     # RGBA
    (np.full((2, 2, 3), 65535, np.uint16), [255, 255, 255]),          # 16-bit full scale
    (np.full((2, 2, 3), 300, np.uint16), [1, 1, 1]),                  # 16-bit, no 8-bit wrap-around
    (np.full((2, 2, 3), 0.5, np.float32), [127, 127, 127]),           # float 0-1
])
def test_to_rgb_uint8(arr, expected):
    out = to_rgb_uint8(arr)

    assert out.dtype == np.uint8 and out.shape == (2, 2, 3)
    assert out[0, 0].tolist() == expected


@pytest.mark.parametrize("name, fmt", [("a.ome.tiff", "tiff"), ("a.TIF", "tiff"), ("a.svs", "openslide"),
                                       ("a.vms", "openslide"), ("a.czi", "czi")])
def test_get_format(name, fmt):
    assert _get_format(Path(name)) == fmt


def test_get_format_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported slide format '.png'"):
        _get_format(Path("a.png"))


def _tiled_rgb_tiff(path, mpp=0.25, planar=False):
    img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    kwargs = dict(photometric="rgb", resolution=(1e4 / mpp, 1e4 / mpp), resolutionunit="CENTIMETER")
    if planar:
        tifffile.imwrite(path, np.moveaxis(img, -1, 0), planarconfig="separate", **kwargs)
    else:
        tifffile.imwrite(path, img, tile=(64, 64), **kwargs)
    return path


def test_slide_source_pyramid_reader_geometry(tmp_path):
    pytest.importorskip("tiffslide")
    with SlideSource(_tiled_rgb_tiff(tmp_path / "s.tif"), target_mpp=0.5) as slide:
        assert (slide.kind, slide.level, slide.downsample) == ("wsi", 0, 1.0)
        assert slide.native_mpp == pytest.approx(0.25) and slide.scale == pytest.approx(2.0)
        assert (slide.width, slide.height) == (400, 300) and slide.mpp == pytest.approx(0.25)
        assert slide.read(10, 20, 30, 40).shape == (40, 30, 3)


@pytest.mark.parametrize("planar", [False, True])
def test_slide_source_flat_reader_geometry(tmp_path, monkeypatch, planar):
    monkeypatch.setitem(sys.modules, "tiffslide", None)  # forces the memmap / decode path

    with SlideSource(_tiled_rgb_tiff(tmp_path / "s.tif", planar=planar), target_mpp=0.5) as slide:
        assert slide.kind == "flat" and slide.arr.shape[-1] == 3
        assert slide.downsample == pytest.approx(2.0)
        assert (slide.width, slide.height) == (200, 150) and slide.mpp == pytest.approx(0.5)
        assert slide.read(0, 0, 50, 40).shape == (40, 50, 3)
