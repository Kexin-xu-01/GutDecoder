import dask.dataframe as dd
import pandas as pd
import pytest

from gutdecoder.reader import spot_pooling

DF = pd.DataFrame({"he_x": [10.0, 30.0, 20.0], "he_y": [5.0, 1.0, 9.0]})


def test_main_dependencies_are_importable():
    # main() used to call register_downscale_img without importing it.
    from hest.utils import register_downscale_img

    assert spot_pooling.register_downscale_img is register_downscale_img


@pytest.mark.parametrize("df", [DF, dd.from_pandas(DF, npartitions=2)])
def test_transcript_bbox_is_ymin_ymax_xmin_xmax(df):
    assert spot_pooling._transcript_bbox(df) == (1.0, 9.0, 10.0, 30.0)
