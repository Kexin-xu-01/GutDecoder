import numpy as np
import pytest
import tifffile
from loguru import logger

from gutdecoder.stain_norm._readers import GUESSED_MPP, get_slide_mpp, read_whole_slide
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
