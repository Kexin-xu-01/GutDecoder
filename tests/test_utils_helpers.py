import os
import sys

import h5py
import matplotlib
import numpy as np
import pytest

from gutdecoder.utils import runs as runs_mod
from gutdecoder.utils.env import disable_cupy
from gutdecoder.utils.h5 import print_h5_tree
from gutdecoder.vis.style import PALETTES, apply_style


@pytest.fixture
def cupy_slot(monkeypatch):
    """Run with cupy absent from sys.modules; restore sys.modules and the env var afterwards."""
    had, prev = "cupy" in sys.modules, sys.modules.get("cupy")
    monkeypatch.delenv("DASK_NO_IMPORT_CUPY", raising=False)
    sys.modules.pop("cupy", None)
    yield
    if had:
        sys.modules["cupy"] = prev
    else:
        sys.modules.pop("cupy", None)


def test_disable_cupy_blocks_import_and_tells_dask(cupy_slot):
    disable_cupy()

    with pytest.raises(ImportError):
        import cupy  # noqa: F401
    assert os.environ["DASK_NO_IMPORT_CUPY"] == "1"


def test_disable_cupy_warns_when_cupy_already_imported(cupy_slot):
    sys.modules["cupy"] = object()

    with pytest.warns(RuntimeWarning, match="already imported"):
        disable_cupy()
    assert sys.modules["cupy"] is None


def test_print_h5_tree_lists_groups_and_datasets(tmp_path, capsys):
    path = tmp_path / "patches.h5"
    with h5py.File(path, "w") as f:
        f["coords"] = np.zeros((3, 2), dtype=np.int32)
        f.create_group("meta")["barcode"] = np.array([b"a", b"b", b"c"])

    print_h5_tree(path)
    out = capsys.readouterr().out

    assert "- DATASET: coords  shape=(3, 2) dtype=int32" in out
    assert "- GROUP: meta" in out
    assert "  - DATASET: meta/barcode" in out


def _registry(tmp_path, text):
    path = tmp_path / "runs.yaml"
    path.write_text(text)
    return path


def test_load_runs_reads_registry_and_checks_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(runs_mod, "RESULTS_ROOT", str(tmp_path))
    (tmp_path / "run_a").mkdir()
    path = _registry(tmp_path, "visium: run_a  # comment\n")

    assert runs_mod.load_runs(path) == {"visium": "run_a"}


def test_load_runs_names_missing_run_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(runs_mod, "RESULTS_ROOT", str(tmp_path))
    path = _registry(tmp_path, "visium: run_a\nxenium: run_b\n")

    with pytest.raises(FileNotFoundError, match="run_a.*run_b|run_b.*run_a"):
        runs_mod.load_runs(path)
    assert runs_mod.load_runs(path, check_exists=False) == {"visium": "run_a", "xenium": "run_b"}


@pytest.mark.parametrize("text", ["- run_a\n", "visium: 3\n"])
def test_load_runs_rejects_malformed_registry(tmp_path, text):
    with pytest.raises(ValueError, match="map labels"):
        runs_mod.load_runs(_registry(tmp_path, text), check_exists=False)


@pytest.mark.parametrize("kind, label_size", [("oral", 11), ("poster", 12)])
def test_apply_style_sets_sizes_and_editable_text(kind, label_size):
    with matplotlib.rc_context():
        apply_style(kind)
        assert matplotlib.rcParams["axes.labelsize"] == label_size
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["axes.spines.right"] is False


def test_apply_style_rejects_unknown_kind_and_leaves_rcparams_alone():
    before = matplotlib.rcParams["axes.labelsize"]
    with pytest.raises(ValueError, match="kind"):
        apply_style("talk")
    assert matplotlib.rcParams["axes.labelsize"] == before
    assert PALETTES["dataset"] == "#8ecae6"
