import pandas as pd

from gutdecoder.vis import _helpers, _io


def test_io_uses_the_shared_column_lookup():
    assert _io._find_col_ci is _helpers._find_col_ci


def test_count_dataset_patches_matches_columns_case_insensitively():
    meta = pd.DataFrame({
        "Sample_ID": ["XeniumPR10_a", "XeniumPR1_b", "XeniumPR10_c"],
        "NUM_PATCHES_100um": [5, 100, 7],
    })

    # XeniumPR10 must not pick up XeniumPR1 samples.
    assert _io.count_dataset_patches(meta, "XeniumPR10") == 12
