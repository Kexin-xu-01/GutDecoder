import numpy as np
import pytest

from gutdecoder.reader.crop_image import build_exclusion_mask_pil
from gutdecoder.reader.h5ad_reader import apply_spot_exclusions

W, H = 60, 40

RULES = [
    {"type": "corner", "corner": "top-right", "width": 0.25, "height": 0.3, "units": "frac"},
    {"type": "corner", "corner": "bottom-left", "width": 12, "height": 7},
    {"type": "strip", "side": "left", "size": 0.1, "units": "frac"},
    {"type": "strip", "side": "bottom", "size": 5},
    {"type": "strip", "axis": "x", "bounds": (0.4, 0.55), "units": "frac"},
    {"type": "strip", "axis": "y", "bounds": (10, 14)},
    {"type": "rect", "xmin": 20, "xmax": 35, "ymin": 5, "ymax": 25},
    {"type": "rect", "xmin": 5, "xmax": 10, "ymin": 5, "ymax": 10, "units": "um"},
    {"type": "trapezoid", "orientation": "top", "top_width": 0.5, "bottom_width": 0.2, "height": 0.3, "units": "frac"},
    {"type": "trapezoid", "orientation": "bottom", "top_width": 30, "bottom_width": 10, "height": 12},
    {"type": "trapezoid", "orientation": "left", "top_width": 0.5, "bottom_width": 0.1, "height": 0.2,
     "center_offset": 0.1, "units": "frac"},
    {"type": "trapezoid", "orientation": "right", "top_width": 20, "bottom_width": 6, "height": 15},
]


@pytest.mark.parametrize("rule", RULES, ids=lambda r: f"{r['type']}-{r.get('corner') or r.get('side') or r.get('axis') or r.get('orientation') or r.get('units')}")
def test_spot_exclusion_and_image_mask_agree(rule):
    ys, xs = np.mgrid[0:H, 0:W]
    keep = apply_spot_exclusions(xs.ravel(), ys.ravel(), (0, W, 0, H), [rule], mpp=0.5).reshape(H, W)
    masked = np.asarray(build_exclusion_mask_pil(W, H, [rule], mpp=0.5)) == 255

    # Compare away from region edges, including diagonal neighbours (corner pixels): PIL fills
    # rectangle end pixels inclusively, so the mask may extend one pixel past an open edge.
    padded = np.pad(keep, 1, mode="edge")
    interior = np.ones_like(keep)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= keep == padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    assert (~keep).any() and keep.any()
    np.testing.assert_array_equal(masked[interior], ~keep[interior])


def test_edge_regions_are_open_and_rects_are_closed():
    xs = np.array([9.9, 10.0, 20.0, 35.0])
    ys = np.full(4, 15.0)
    rules = [{"type": "strip", "side": "left", "size": 10},
             {"type": "rect", "xmin": 20, "xmax": 35, "ymin": 5, "ymax": 25}]

    assert apply_spot_exclusions(xs, ys, (0, W, 0, H), rules).tolist() == [False, True, False, False]


def test_um_rules_need_mpp():
    with pytest.raises(ValueError, match="mpp is required"):
        apply_spot_exclusions([1.0], [1.0], (0, W, 0, H), [RULES[7]])
