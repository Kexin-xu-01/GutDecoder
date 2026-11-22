import pandas as pd
import pytest

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


def test_safe_read_json_names_the_malformed_file(tmp_path):
    bad = tmp_path / "enc_results.json"
    bad.write_text("{not json")

    with pytest.raises(ValueError, match="enc_results.json"):
        _helpers._safe_read_json(bad)


def test_merge_rejects_gene_table_without_expected_columns():
    with pytest.raises(KeyError, match="corr_per_split"):
        _io.merge_kfold_gene_corrs_with_test_metadata(pd.DataFrame({"gene": ["g1"]}), pd.DataFrame())


def _fake_run(monkeypatch, missing=False):
    def load_run(run, runs_root):
        if missing:
            raise FileNotFoundError("dataset_results.json")
        return "DS", None, None

    monkeypatch.setattr(_io, "load_run", load_run)
    monkeypatch.setattr(_io, "get_test_splits", lambda run, runs_root, splits_root: pd.DataFrame())
    monkeypatch.setattr(_io, "extract_best_model_gene_corrs", lambda run, runs_root, verbose: (None, None, None))
    monkeypatch.setattr(_io, "merge_kfold_gene_corrs_with_test_metadata", lambda genes, splits: pd.DataFrame({
        "test_sample": ["A", "A", "A", "B"],
        "corr": [0.2, 0.4, None, 0.9],
    }))


def test_sample_corr_stats_for_run(tmp_path, monkeypatch):
    _fake_run(monkeypatch)

    stats = _io.sample_corr_stats_for_run("run_1", ["A"], str(tmp_path), str(tmp_path)).iloc[0]

    assert stats["sample"] == "A" and stats["run"] == "run_1" and stats["dataset"] == "DS"
    assert stats["mean_pearson"] == pytest.approx(0.3)
    assert stats["n_genes"] == 3


def test_sample_corr_stats_skips_unfinished_run_with_warning(tmp_path, monkeypatch):
    _fake_run(monkeypatch, missing=True)

    with pytest.warns(UserWarning, match="Skipping run_1"):
        assert _io.sample_corr_stats_for_run("run_1", ["A"], str(tmp_path), str(tmp_path)) is None
