import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gutdecoder.vis.plot_slide_emb_umap as umap_mod

PACKAGE_ROOT = Path(umap_mod.__file__).resolve().parents[1]
PRE_MERGE_COMMIT = "a3654ba"  # last commit with separate 2-group / 3-group plotting functions

rng = np.random.default_rng(0)
EMB = rng.normal(size=(60, 2))
CATS = pd.Series(rng.choice(["A", "B", None], 60))
NUMS = pd.Series(np.where(rng.random(60) < 0.1, np.nan, rng.random(60)))
G2 = pd.Series(rng.choice(["xenium", "adjacent"], 60))
G3 = pd.Series(rng.choice(["xenium", "adjacent", "nec"], 60))


def _calls(fn, *args):
    """Run a plotting function on mock axes; return its scatter calls and legend handles, comparably."""
    ax, legend_ax = mock.MagicMock(), mock.MagicMock()
    module = sys.modules[fn.__module__]
    with mock.patch.object(module.plt, "colorbar"):
        if "categorical" in fn.__name__:
            fn(ax, legend_ax, *args)
        else:
            fn(ax, *args)

    def plain(v):
        if isinstance(v, (np.ndarray, pd.Series)):
            return np.asarray(v).tolist()
        if isinstance(v, matplotlib.colors.Normalize):
            return ("Normalize", float(v.vmin), float(v.vmax))
        if isinstance(v, matplotlib.colors.Colormap):
            return ("Colormap", v.name)
        return v

    # an explicit None (e.g. edgecolors=None) draws the same as leaving the argument out
    scatters = [([plain(a) for a in c.args], {k: plain(v) for k, v in c.kwargs.items() if v is not None})
                for c in ax.scatter.call_args_list]
    handles = [[(h.get_label(), h.get_marker()) for h in c.kwargs.get("handles", [])]
               for c in ax.legend.call_args_list + legend_ax.legend.call_args_list]
    return scatters, handles


@pytest.fixture(scope="module")
def old_module():
    try:
        src = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "show", f"{PRE_MERGE_COMMIT}:vis/plot_slide_emb_umap.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"pre-merge commit {PRE_MERGE_COMMIT} not available")
    module = types.ModuleType("plot_slide_emb_umap_pre_merge")
    sys.modules[module.__name__] = module
    exec(compile(src, module.__name__, "exec"), module.__dict__)
    yield module
    sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize("name, args", [
    ("plot_categorical_by_group", (EMB, CATS, G2, "xenium", "adjacent", "UMAP colored by tissue")),
    ("plot_numeric_by_group", (EMB, NUMS, G2, "xenium", "adjacent", "UMAP colored by age")),
    ("plot_categorical_by_three_groups", (EMB, CATS, G3, ["xenium", "adjacent", "nec"], "UMAP colored by tissue")),
    ("plot_numeric_by_three_groups", (EMB, NUMS, G3, ["xenium", "adjacent", "nec"], "UMAP colored by age")),
])
def test_wrappers_draw_exactly_what_the_pre_merge_functions_drew(old_module, name, args):
    assert _calls(getattr(umap_mod, name), *args) == _calls(getattr(old_module, name), *args)


def test_n_group_categorical_supports_more_than_three_groups():
    groups = pd.Series(rng.choice(["a", "b", "c", "d"], 60))
    fig, (ax, legend_ax) = plt.subplots(1, 2)

    umap_mod.plot_categorical_by_groups(ax, legend_ax, EMB, CATS, groups, ["a", "b", "c", "d"], "UMAP colored by tissue")

    group_legend = legend_ax.get_legend()
    assert [t.get_text() for t in group_legend.get_texts()] == ["a", "b", "c", "d"]
    plt.close(fig)


def test_n_group_functions_reject_too_many_groups():
    labels = [f"g{i}" for i in range(len(umap_mod.GROUP_MARKERS) + 1)]
    with pytest.raises(ValueError, match="at most"):
        umap_mod.plot_numeric_by_groups(mock.MagicMock(), EMB, NUMS, G2, labels, "t")


def test_numeric_all_nan_only_sets_title():
    ax = mock.MagicMock()
    umap_mod.plot_numeric_by_groups(ax, EMB, pd.Series([np.nan] * 60), G2, ["xenium", "adjacent"], "t")

    ax.set_title.assert_called_once_with("t")
    ax.scatter.assert_not_called()


def test_pca_elbow_returns_fitted_pca_and_saves(tmp_path):
    X = np.random.default_rng(1).normal(size=(30, 8))

    pca, fig, ax = umap_mod.plot_pca_elbow(X, out_path=tmp_path / "elbow.png", max_components=5)

    assert pca.n_components_ == 5
    assert np.isclose(np.cumsum(pca.explained_variance_ratio_)[-1], pca.explained_variance_ratio_.sum())
    assert (tmp_path / "elbow.png").exists()
    plt.close(fig)


@pytest.mark.parametrize("X", [np.zeros(5), np.zeros((1, 3))])
def test_pca_elbow_rejects_bad_input(X):
    with pytest.raises(ValueError):
        umap_mod.plot_pca_elbow(X)
