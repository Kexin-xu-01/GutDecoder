"""Regression checks for RGB mosaic stitching, without a CZI fixture."""

from types import SimpleNamespace

import numpy as np
import pytest

from gutdecoder.stain_norm._czi import _stitch_tiles


class Tile:
    def __init__(self, index):
        # S is the scene coordinate here; it is not a samples index.
        self.dimension_coordinates = {"M": index, "T": 0, "C": 0, "Z": 0, "S": 7}


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("samples", [None, 1, 3, 4])
def test_stitch_keeps_samples_and_scene_metadata(lazy, samples):
    dims = "MTCZYX" + ("S" if samples else "")
    shape = (2, 1, 1, 1, 12, 16) + ((samples,) if samples else ())
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    original = data
    if lazy:
        da = pytest.importorskip("dask.array")
        data = da.from_array(data, chunks=(1,) + shape[1:])
    first, second = Tile(0), Tile(1)
    boxes = {
        first: SimpleNamespace(x=100, y=200, w=16, h=12),
        second: SimpleNamespace(x=116, y=200, w=16, h=12),
    }
    result = _stitch_tiles(
        data, dims, {}, boxes, SimpleNamespace(x=100, y=200, w=32, h=12)
    )
    if lazy:
        result = result.compute()
    expected = np.concatenate([original[0], original[1]], axis=4)
    np.testing.assert_array_equal(result, expected)
    assert first.dimension_coordinates["S"] == 7
    assert second.dimension_coordinates["S"] == 7


def test_rgb_tile_at_reported_size():
    data = np.empty((1, 1, 1, 1, 1200, 1600, 3), dtype=np.uint8)
    data[...] = [35, 100, 210]
    box = SimpleNamespace(x=0, y=0, w=1600, h=1200)
    result = _stitch_tiles(data, "MTCZYXS", {}, {Tile(0): box}, box)
    np.testing.assert_array_equal(result, data[0])
