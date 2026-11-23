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


POINTS = pd.DataFrame({
    "feature_name": ["A", "B", "A"],
    "he_x": [5.0, 15.0, 20.0],
    "he_y": [5.0, 3.0, 20.0],
})
BBOX = (0.0, 20.0, 0.0, 20.0)  # 2 x 2 grid of 10 px spots


def _pool(df):
    return spot_pooling.pool_transcripts_xenium_bbox(df, BBOX, pixel_size_he=1.0, spot_size_um=10.0)


@pytest.mark.parametrize("to_frame", [lambda df: df, lambda df: dd.from_pandas(df, npartitions=2)])
def test_pooling_puts_max_edge_points_in_last_spot(to_frame):
    adata = _pool(to_frame(POINTS))
    counts = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    assert counts.loc["000x000", "A"] == 1
    assert counts.loc["000x001", "B"] == 1
    assert counts.loc["001x001", "A"] == 1
    assert counts.to_numpy().sum() == len(POINTS)


@pytest.mark.parametrize("to_frame", [lambda df: df, lambda df: dd.from_pandas(df, npartitions=2)])
@pytest.mark.parametrize("x, y", [(-1.0, 5.0), (25.0, 5.0), (5.0, 21.0), (float("nan"), 5.0)])
def test_pooling_rejects_points_outside_bbox(to_frame, x, y):
    df = pd.concat([POINTS, pd.DataFrame({"feature_name": ["A"], "he_x": [x], "he_y": [y]})], ignore_index=True)

    with pytest.raises(ValueError, match="outside the pooling bbox"):
        _pool(to_frame(df))


def _args(*extra):
    return spot_pooling._parse_main_args(["S1", "--output-root", "/out", "--processed-root", "/proc", *extra])


def test_roi_paths_from_spreadsheet_row():
    row = pd.Series(dict(Sample_ID="S1", Roi=2.0, Slide="slide3", run=" PR4 ", Directory="/img",
                         alignment="align.csv", h5ad="/cells.h5ad", directory="/exp"))

    paths = spot_pooling._roi_paths(row, _args(), feature_tag="cell_type")

    assert (paths["roi"], paths["slide_num"], paths["run_name"]) == (2, 3, "PR4")
    assert str(paths["alignment_file"]) == "/img/align.csv"
    assert str(paths["hest_dir"]) == "/proc/XeniumPR4/slide3/ROI2"
    assert str(paths["save_dir"]) == "/out/XeniumPR4_cell_type/slide3/ROI2"
    assert str(paths["transcript_path"]) == "/exp/transcripts.parquet"


@pytest.mark.parametrize("missing", ["Roi", "alignment", "h5ad"])
def test_roi_paths_skips_incomplete_rows(missing):
    row = dict(Sample_ID="S1", Roi=1, Slide="slide1", run="PR1", Directory="/img", alignment="a", h5ad="b",
               directory="/exp")
    row[missing] = float("nan")

    assert spot_pooling._roi_paths(pd.Series(row), _args(), feature_tag="cell_type") is None


def test_config_file_overrides_command_line(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("feature_col: niche\n")

    assert _args("--feature-col", "cell_type", "--config", str(cfg)).feature_col == "niche"
