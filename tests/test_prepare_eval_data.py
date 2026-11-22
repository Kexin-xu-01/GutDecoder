import os

import pytest

from gutdecoder.utils.prepare_eval_data import _load_patient_map, _plan_and_transfer_assets, _transfer


def test_patient_map_none_means_infer(tmp_path):
    assert _load_patient_map(None) == ({}, None)


def test_patient_map_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata_csv=None"):
        _load_patient_map(str(tmp_path / "nope.csv"))


def test_patient_map_missing_columns_raises(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("sample,patient\nA,P1\n")

    with pytest.raises(ValueError, match="'sample_id' and 'patient_id'"):
        _load_patient_map(str(csv))


def test_patient_map_reads_and_strips(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("sample_id,patient_id\n A ,P1 \n")

    patient_map, _ = _load_patient_map(str(csv))

    assert patient_map == {"A": "P1"}


def test_transfer_replaces_broken_symlink(tmp_path):
    src = tmp_path / "src.h5"
    src.write_text("data")
    dst = tmp_path / "out" / "A.h5"
    dst.parent.mkdir()
    os.symlink(tmp_path / "gone.h5", dst)  # broken link left by an earlier run

    _transfer(src, dst, "patch", symlink=True, missing_list=[])

    assert dst.resolve() == src.resolve()


def test_plan_reports_sources_that_vanished(tmp_path):
    adata = tmp_path / "A.h5ad"
    adata.write_text("x")
    sources = {"A": {"adata": [adata], "patch": [tmp_path / "gone.h5"]}}

    missing = _plan_and_transfer_assets(["A"], sources, tmp_path / "out", symlink=False)

    assert missing == [("A", "patch")]
    assert (tmp_path / "out" / "adata" / "A.h5ad").exists()
