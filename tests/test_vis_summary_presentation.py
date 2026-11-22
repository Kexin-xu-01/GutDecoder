import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
from loguru import logger

matplotlib.use("Agg")

from gutdecoder.vis._summary import compare_models, plot_dataset_violin, plot_scaling_relationships
from gutdecoder.vis._io import load_pred_adata, load_run_with_model_meta, stack_gene_corrs


@pytest.fixture
def warnings_log():
    messages = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


def _run(tmp_path, name="run_1"):
    run = tmp_path / name
    run.mkdir()
    (run / "dataset_results.json").write_text(json.dumps({"results": [{
        "dataset_name": "DS",
        "results": [{"encoder_name": e, "pearson_mean": m, "pearson_std": 0.01}
                    for e, m in [("uni", 0.3), ("conch", 0.25)]],
    }]}))
    return name


def test_compare_models_defaults_keep_title_and_label(tmp_path):
    fig = compare_models(_run(tmp_path), runs_root=str(tmp_path))
    ax = fig.axes[0]

    assert ax.get_ylabel() == "Pearson correlation"
    assert ax.get_title().startswith("Dataset: DS")
    assert tuple(fig.get_size_inches()) == (12, 6)


def test_compare_models_presentation_options(tmp_path):
    out = tmp_path / "compare.png"

    fig = compare_models(_run(tmp_path), runs_root=str(tmp_path), figsize=(11, 3),
                         ylabel="Prediction correlation", title=False, save_path=out)

    assert fig.axes[0].get_ylabel() == "Prediction correlation"
    assert fig.axes[0].get_title() == ""
    assert tuple(fig.get_size_inches()) == (11, 3)
    assert out.exists()


def _merged():
    return pd.DataFrame({
        "encoder_name": ["uni", "conch", "hopt", "nometa"],
        "pearson_mean": [0.30, 0.25, 0.35, 0.20],
        "number_of_parameters": [3e8, 9e7, 1.1e9, np.nan],
        "number_of_patches": [1e8, 1e6, 5e8, 1e7],
    })


@pytest.mark.parametrize("labels", ["drop_overlapping", "all", "adjust"])
def test_scaling_plot_logs_dropped_rows_and_saves(tmp_path, warnings_log, labels):
    out = tmp_path / f"scaling_{labels}.png"

    fig = plot_scaling_relationships(_merged(), labels=labels, panel_titles=("a.", "b."), save_path=out, show=False)

    assert out.exists()
    assert [ax.get_title() for ax in fig.axes] == ["a.", "b."]
    assert any("nometa" in m for m in warnings_log)


def test_scaling_plot_drop_overlapping_keeps_fewer_labels_than_all():
    df = pd.DataFrame({  # identical points: every label lands on the same spot
        "encoder_name": [f"enc{i}" for i in range(5)], "pearson_mean": 0.3,
        "number_of_parameters": 1e8, "number_of_patches": 1e6,
    })
    df.loc[0, "pearson_mean"] = 0.31            # keep a non-degenerate regression
    df.loc[0, "number_of_parameters"] = 2e8
    df.loc[0, "number_of_patches"] = 2e6

    n_texts = lambda fig: len(fig.axes[0].texts)
    assert n_texts(plot_scaling_relationships(df, labels="drop_overlapping", show=False)) < \
           n_texts(plot_scaling_relationships(df, labels="all", show=False))


def test_scaling_plot_rejects_unknown_label_mode():
    with pytest.raises(ValueError, match="labels"):
        plot_scaling_relationships(_merged(), labels="nope", show=False)


def _violin_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "dataset_label": np.repeat(["Visium", "Xenium", "Xenium 5K"], 20),
        "mean_corr": rng.uniform(0, 0.6, 60),
    })


def test_violin_writes_counts_above_each_dataset(tmp_path):
    ax = plot_dataset_violin(_violin_df(), n_labels=["n = 34", "n = 33", "n = 29"],
                             shaded_ranges=[(0, 0.2, 0, 1 / 3, 0.15)], save_path=tmp_path / "v.png", show=False)

    assert [t.get_text() for t in ax.texts] == ["n = 34", "n = 33", "n = 29"]
    assert (tmp_path / "v.png").exists()


def test_violin_rejects_wrong_number_of_count_labels():
    with pytest.raises(ValueError, match="n_labels"):
        plot_dataset_violin(_violin_df(), n_labels=["n = 1"], show=False)


# ----------------------------------------------------------------------------- presentation loaders

def test_stack_gene_corrs_labels_each_dataset_with_its_gene_count():
    frames = {"Visium": pd.DataFrame({"gene": ["a", "b"], "mean_corr": [0.1, 0.2]}),
              "Xenium 5K": pd.DataFrame({"gene": ["a", "b", "c"], "mean_corr": [0.3, 0.4, 0.5]})}

    df = stack_gene_corrs(frames)

    assert df["dataset"].tolist() == ["Visium"] * 2 + ["Xenium 5K"] * 3
    assert df["dataset_label"].unique().tolist() == ["Visium\n(2 genes)", "Xenium 5K\n(3 genes)"]
    assert "dataset" not in frames["Visium"].columns  # inputs untouched


def test_stack_gene_corrs_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        stack_gene_corrs({})


def test_load_pred_adata_reads_run_pred_folder(tmp_path):
    import anndata as ad

    (tmp_path / "run_1" / "pred").mkdir(parents=True)
    ad.AnnData(np.ones((2, 3))).write_h5ad(tmp_path / "run_1" / "pred" / "S1.h5ad")

    assert load_pred_adata("run_1", "S1", runs_root=str(tmp_path)).shape == (2, 3)


def test_load_pred_adata_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="S2.h5ad"):
        load_pred_adata("run_1", "S2", runs_root=str(tmp_path))


def test_load_run_with_model_meta_merges_and_warns_on_unmatched(tmp_path, warnings_log):
    pytest.importorskip("openpyxl")
    meta = tmp_path / "models.xlsx"
    pd.DataFrame({"name": ["uni"], "number_of_parameters": [3e8]}).to_excel(meta, index=False)

    merged = load_run_with_model_meta(_run(tmp_path), model_meta_xlsx=str(meta), runs_root=str(tmp_path))

    assert merged.set_index("encoder_name").loc["uni", "number_of_parameters"] == 3e8
    assert np.isnan(merged.set_index("encoder_name").loc["conch", "number_of_parameters"])  # kept, not dropped
    assert any("conch" in m for m in warnings_log)
