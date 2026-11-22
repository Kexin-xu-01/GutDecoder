import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from loguru import logger

matplotlib.use("Agg")

from gutdecoder.vis.plot import plot_mean_std_ranked
from gutdecoder.vis.pred_figures import (
    layer_values,
    save_one_sample_shared_cbar,
    save_two_samples_shared_cbar,
    shared_vmin_vmax,
    show_pred_target,
)


@pytest.fixture
def warnings_log():
    messages = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


def _adata(seed=0, sample_id="S1"):
    rng = np.random.default_rng(seed)
    n, genes = 50, ["CD3E", "EPCAM", "ALLNAN"]
    target = rng.poisson(3, (n, 3)).astype(float)
    pred = target + rng.normal(0, 1, (n, 3))
    target[:, 2] = pred[:, 2] = np.nan
    a = ad.AnnData(X=target.copy(), var=pd.DataFrame(index=genes),
                   obs=pd.DataFrame({"sample_id": sample_id}, index=[f"c{i}" for i in range(n)]))
    a.layers["pred"], a.layers["target"] = pred, target
    a.obsm["spatial"] = rng.uniform(0, 100, (n, 2))
    a.uns["spatial"] = {sample_id: {
        "images": {"downscaled_fullres": rng.uniform(0, 1, (100, 100, 3))},
        "scalefactors": {"tissue_downscaled_fullres_scalef": 1.0, "spot_diameter_fullres": 2.0},
    }}
    return a


def test_layer_values_returns_gene_column_and_rejects_unknown_gene():
    a = _adata()
    assert np.array_equal(layer_values(a, "pred", "EPCAM"), a.layers["pred"][:, 1])
    with pytest.raises(KeyError, match="NOPE"):
        layer_values(a, "pred", "NOPE")


def test_shared_scale_spans_all_arrays_and_respects_overrides():
    assert shared_vmin_vmax([np.array([1.0, np.nan, 5.0]), np.array([-2.0])]) == (-2.0, 5.0)
    assert shared_vmin_vmax([np.array([1.0, 5.0])], vmin=0, vmax=10) == (0.0, 10.0)


def test_shared_scale_widens_a_constant_range():
    vmin, vmax = shared_vmin_vmax([np.array([3.0, 3.0])])
    assert vmin == 3.0 and vmax > vmin


def test_shared_scale_warns_on_all_nan(warnings_log):
    assert shared_vmin_vmax([np.array([np.nan])], label="ALLNAN") == (0.0, 1.0)
    assert any("ALLNAN" in m for m in warnings_log)


def test_show_pred_target_plots_every_gene_and_warns_on_missing(warnings_log, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    a = _adata()
    obs_before = list(a.obs.columns)

    scales = show_pred_target(a, ["CD3E", "EPCAM", "NOPE"])

    # The notebook copy returned inside the loop and only ever plotted the first gene.
    assert set(scales) == {"CD3E", "EPCAM"}
    assert any("NOPE" in m for m in warnings_log)
    assert list(a.obs.columns) == obs_before  # temporary _pred/_target columns removed


def test_one_sample_figure_writes_pdf_and_png(tmp_path):
    vmin, vmax = save_one_sample_shared_cbar(_adata(), "CD3E", tmp_path / "one.pdf", label="Crohn's")

    assert (tmp_path / "one.pdf").exists() and (tmp_path / "one.png").exists()
    assert vmin < vmax


@pytest.mark.parametrize("layout", ["rows", "cols"])
def test_two_sample_figure_writes_both_layouts(tmp_path, layout):
    out = tmp_path / f"{layout}.pdf"

    save_two_samples_shared_cbar(_adata(0, "S1"), _adata(1, "S2"), "CD3E", out, layout=layout)

    assert out.exists() and out.with_suffix(".png").exists()


def test_two_sample_rows_without_row_heights(tmp_path):
    # The notebook copy raised NameError when row_heights was omitted.
    save_two_samples_shared_cbar(_adata(0, "S1"), _adata(1, "S2"), "CD3E", tmp_path / "r.pdf", labels=["A", "B"])


def test_two_sample_cols_warns_that_labels_are_not_drawn(tmp_path, warnings_log):
    save_two_samples_shared_cbar(_adata(0, "S1"), _adata(1, "S2"), "CD3E", tmp_path / "c.pdf",
                                 layout="cols", labels=["A", "B"])
    assert any("labels" in m for m in warnings_log)


def test_two_sample_rejects_bad_layout_and_label_count(tmp_path):
    with pytest.raises(ValueError, match="layout"):
        save_two_samples_shared_cbar(_adata(), _adata(1), "CD3E", tmp_path / "x.pdf", layout="grid")
    with pytest.raises(ValueError, match="labels"):
        save_two_samples_shared_cbar(_adata(), _adata(1), "CD3E", tmp_path / "x.pdf", labels=["only one"])


def test_mean_std_ranked_sorts_descending_and_thins_labels(tmp_path, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    df = pd.DataFrame({"gene": [f"G{i}" for i in range(60)],
                       "mean_corr": np.linspace(0, 0.9, 60), "std_corr": 0.05})

    fig, ax = plot_mean_std_ranked(df, max_labels=20, savepath=tmp_path / "ranked")

    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "G59"                 # highest mean first
    assert len(labels) == 20                  # every 60 // 20 = 3rd label
    assert (tmp_path / "ranked.pdf").exists() and (tmp_path / "ranked.png").exists()


def test_mean_std_ranked_names_missing_columns():
    with pytest.raises(ValueError, match="std_corr"):
        plot_mean_std_ranked(pd.DataFrame({"gene": ["a"], "mean_corr": [0.1]}))
