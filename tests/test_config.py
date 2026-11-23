import os
import runpy

import gutdecoder.config as config


def _fresh_config(monkeypatch, **env):
    """Evaluate config.py with the given environment, without touching the imported module."""
    for name in list(os.environ):
        if name.startswith("GUTDECODER_"):
            monkeypatch.delenv(name)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return runpy.run_path(config.__file__)


def test_every_path_follows_hest_root(monkeypatch):
    cfg = _fresh_config(monkeypatch, GUTDECODER_HEST_ROOT="/data/hest")

    assert cfg["RESULTS_ROOT"] == "/data/hest/eval/ST_pred_results"
    assert cfg["XENIUM_DIRECTORY_XLSX"] == "/data/hest/metadata/xenium_directory.xlsx"
    assert cfg["EMBEDDING_ROOT"] == "/data/hest/eval/ST_data_emb"


def test_single_path_override_wins(monkeypatch):
    cfg = _fresh_config(monkeypatch, GUTDECODER_HEST_ROOT="/data/hest", GUTDECODER_DATA_ROOT="/fast/data")

    assert cfg["DATA_ROOT"] == "/fast/data"
    assert cfg["RESULTS_ROOT"] == "/data/hest/eval/ST_pred_results"


def test_defaults_are_unchanged(monkeypatch):
    cfg = _fresh_config(monkeypatch)

    assert cfg["RESULTS_ROOT"] == "/project/gutdecoder/kxu/hest/eval/ST_pred_results"
    assert cfg["CURATED_XLSX"] == "/project/gutdecoder/kxu/hest/metadata/curated_gene_list.xlsx"
    assert cfg["BROAD_METADATA_CSV"] == "/project/gutdecoder/kxu/hest/metadata/broad_directory.csv"
