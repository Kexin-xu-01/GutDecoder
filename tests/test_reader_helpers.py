from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from loguru import logger

matplotlib.use("Agg")

from gutdecoder.reader.h5ad_reader import (
    collect_feature_union,
    fast_spatial,
    pad_adata_to_var_union,
    pick_spatial_xy_columns,
)
from gutdecoder.reader.sample_meta import SampleRow, clean_sample_id


@pytest.fixture
def warnings_log():
    messages = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


# ----------------------------------------------------------------------------- padding

def _small(sparse=True):
    X = np.array([[1, 2], [3, 4]], dtype=float)
    a = ad.AnnData(X=sp.csr_matrix(X) if sparse else X,
                   var=pd.DataFrame({"_index": ["b", "a"]}, index=["b", "a"]),
                   obs=pd.DataFrame({"cell_centroid_x": [1.0, 2.0], "cell_centroid_y": [3.0, 4.0]}, index=["c0", "c1"]))
    a.layers["counts"] = a.X.copy()
    a.varm["loadings"] = np.array([[0.5], [0.7]])
    return a


@pytest.mark.parametrize("sparse", [True, False])
def test_padding_reorders_and_zero_fills_every_matrix(sparse):
    padded = pad_adata_to_var_union(_small(sparse), ["a", "b", "z"])
    dense = lambda m: m.toarray() if sp.issparse(m) else m

    assert list(padded.var_names) == ["a", "b", "z"]
    assert np.array_equal(dense(padded.X), [[2, 1, 0], [4, 3, 0]])
    assert np.array_equal(dense(padded.layers["counts"]), dense(padded.X))
    assert np.array_equal(padded.varm["loadings"].ravel(), [0.7, 0.5, 0.0])
    assert list(padded.obs.columns) == ["cell_centroid_x", "cell_centroid_y"]


def test_padded_file_can_be_written_despite_index_column(tmp_path):
    padded = pad_adata_to_var_union(_small(), ["a", "b", "z"])

    assert "index_value" in padded.var.columns and "_index" not in padded.var.columns
    padded.write_h5ad(tmp_path / "padded.h5ad")  # a var column named _index breaks writing


def test_padding_with_identical_features_only_renames_index_column():
    a = _small()
    same = pad_adata_to_var_union(a, ["b", "a"])

    assert list(same.var_names) == ["b", "a"]
    assert "index_value" in same.var.columns
    assert "_index" in a.var.columns  # input untouched


def test_padding_rejects_features_outside_the_target_list():
    with pytest.raises(ValueError, match="'b'"):
        pad_adata_to_var_union(_small(), ["a"])


def test_padding_rejects_duplicate_target_features():
    with pytest.raises(ValueError, match="unique"):
        pad_adata_to_var_union(_small(), ["a", "a", "b"])


# ----------------------------------------------------------------------------- feature union

def _write_labelled(path, labels):
    obs = pd.DataFrame({"predicted.id": labels}, index=[f"c{i}" for i in range(len(labels))])
    ad.AnnData(np.zeros((len(labels), 1)), obs=obs).write_h5ad(path)
    return str(path)


def test_feature_union_collects_sorted_labels_and_warns_on_missing_file(tmp_path, warnings_log):
    rows = pd.DataFrame({"h5ad": [
        _write_labelled(tmp_path / "a.h5ad", ["T cell", "B cell"]),
        _write_labelled(tmp_path / "b.h5ad", ["Epithelial", "T cell", None]),
        str(tmp_path / "missing.h5ad"),
        str(tmp_path / "missing.h5ad"),   # warned once only
        np.nan,
    ]})

    assert collect_feature_union(rows, "h5ad") == ["B cell", "Epithelial", "T cell"]
    assert sum("missing.h5ad" in m for m in warnings_log) == 1


def test_feature_union_fails_when_no_labels_found(tmp_path):
    with pytest.raises(ValueError, match="No cell type"):
        collect_feature_union(pd.DataFrame({"h5ad": [str(tmp_path / "missing.h5ad")]}), "h5ad")


def test_pick_spatial_columns_prefers_cell_centroid_pair():
    obs = pd.DataFrame(columns=["x", "y", "cell_centroid_x", "cell_centroid_y"])
    assert pick_spatial_xy_columns(obs) == ("cell_centroid_x", "cell_centroid_y")
    with pytest.raises(KeyError, match="centroid_only_x"):
        pick_spatial_xy_columns(pd.DataFrame(columns=["centroid_only_x"]))


# ----------------------------------------------------------------------------- fast_spatial

def _spatial_adata(n=200):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.poisson(2, (n, 2)).astype(float), var=pd.DataFrame(index=["CD3E", "EPCAM"]),
                   obs=pd.DataFrame({"cell_type": rng.choice(["T", "B"], n)}, index=[f"c{i}" for i in range(n)]))
    a.obsm["spatial"] = rng.uniform(0, 100, (n, 2))
    a.obs["cx"], a.obs["cy"] = a.obsm["spatial"][:, 0] + 1000, a.obsm["spatial"][:, 1]
    return a


def test_fast_spatial_default_uses_obsm_spatial(tmp_path):
    assert fast_spatial(_spatial_adata(), color="cell_type", save=tmp_path / "d.png") == tmp_path / "d.png"


@pytest.fixture
def scattered(monkeypatch):
    """Record the x/y arrays of every ax.scatter call made through plt.subplots."""
    calls = []
    real_subplots = matplotlib.pyplot.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        real_scatter = ax.scatter

        def scatter(x, y, **kw):
            calls.append((np.asarray(x), np.asarray(y)))
            return real_scatter(x, y, **kw)

        ax.scatter = scatter
        return fig, ax

    monkeypatch.setattr(matplotlib.pyplot, "subplots", spy_subplots)
    return calls


def test_fast_spatial_obs_columns_gene_colour_and_subsample(tmp_path, scattered):
    fast_spatial(_spatial_adata(), color="CD3E", x="cx", y="cy", max_cells=50, save=tmp_path / "g.png")

    xs, _ = scattered[0]
    assert len(scattered) == 1    # numeric (gene) colour: one scatter call
    assert len(xs) == 50          # subsampled
    assert xs.min() >= 1000       # coordinates came from obs["cx"], not obsm["spatial"]


def test_fast_spatial_subsample_is_reproducible(tmp_path, scattered):
    for i in range(2):
        fast_spatial(_spatial_adata(), max_cells=10, save=tmp_path / f"s{i}.png")

    assert np.array_equal(scattered[0][0], scattered[1][0])


def test_fast_spatial_rejects_half_coordinates_and_unknown_colour(tmp_path):
    with pytest.raises(ValueError, match="both x and y"):
        fast_spatial(_spatial_adata(), x="cx", save=tmp_path / "a.png")
    with pytest.raises(ValueError, match="NOPE"):
        fast_spatial(_spatial_adata(), color="NOPE", save=tmp_path / "b.png")


# ----------------------------------------------------------------------------- SampleRow

def test_sample_row_parses_metadata_and_builds_folder():
    row = SampleRow.from_row(pd.Series({"Sample_ID": " XeniumPR1S1ROI3\t", "run": "PR1 ", "Slide": "slide1", "Roi": 3.0}))

    assert row == SampleRow("XeniumPR1S1ROI3", "PR1", 1, 3)
    assert row.hest_dir("/r") == Path("/r/XeniumPR1/slide1/ROI3")
    assert row.hest_dir("/r", prefix="Visium", suffix="_cell_l0") == Path("/r/VisiumPR1_cell_l0/slide1/ROI3")


@pytest.mark.parametrize("slide, roi, expected", [(2.0, "4", (2, 4)), ("S1", 1, (1, 1))])
def test_sample_row_accepts_numeric_and_text_fields(slide, roi, expected):
    row = SampleRow.from_row({"Sample_ID": "a", "Slide": slide, "Roi": roi})
    assert (row.slide, row.roi, row.run) == (*expected, "")


@pytest.mark.parametrize("slide, roi", [("unknown", 1), ("slide1", np.nan)])
def test_sample_row_refuses_incomplete_folder(slide, roi):
    with pytest.raises(ValueError, match="cannot build"):
        SampleRow.from_row({"Sample_ID": "x", "run": "R2", "Slide": slide, "Roi": roi}).hest_dir("/r")


def test_clean_sample_id_normalises_unicode_and_whitespace():
    assert clean_sample_id("Ｘenium PR1\n") == "XeniumPR1"
    assert clean_sample_id(np.nan) == ""
