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


def test_assign_conditions_by_run_number_then_name():
    from gutdecoder.utils.prepare_eval_data import _DEFAULT_CONDITION_MAP, _assign_conditions

    ids = ["XeniumPR2S1ROI1", "XeniumPR10S1ROI2", "NEC_extra", "other"]

    assert _assign_conditions(ids, _DEFAULT_CONDITION_MAP) == ["IBD", "NEC", "NEC", None]


def test_condition_splits_hold_out_one_condition_each(tmp_path):
    import pandas as pd
    from gutdecoder.utils.prepare_eval_data import _write_condition_splits

    meta = pd.DataFrame({"id": ["a", "b", "c", "d"], "condition": ["IBD", "IBD", "NEC", None]})

    _write_condition_splits(meta, tmp_path, drop_unassigned_from_splits=True)

    assert pd.read_csv(tmp_path / "test_0.csv")["sample_id"].tolist() == ["a", "b"]
    assert pd.read_csv(tmp_path / "train_0.csv")["sample_id"].tolist() == ["c"]
    assert pd.read_csv(tmp_path / "train_1.csv")["patches_path"].tolist() == ["patches/a.h5", "patches/b.h5"]
    assert pd.read_csv(tmp_path / "condition_to_split.csv")["condition"].tolist() == ["IBD", "NEC"]


def test_ex_val_alias_uses_default_condition_groups(monkeypatch):
    from gutdecoder.utils import prepare_eval_data as ped

    calls = {}
    monkeypatch.setattr(ped, "create_benchmark_data_multirun", lambda *a, **kw: calls.update(kw))

    ped.create_benchmark_data_multirun_ex_val("out", 5, ["eval"])

    assert calls["condition_map"] == ped._DEFAULT_CONDITION_MAP


def test_patient_kfold_loocv_makes_one_fold_per_patient(tmp_path):
    import pandas as pd
    from gutdecoder.utils.prepare_eval_data import _write_patient_kfold_splits

    meta = pd.DataFrame({"id": ["a", "b", "c", "d"], "patient": ["P1", "P1", "P2", "P3"],
                         "dataset_title": "XeniumPR"})

    _write_patient_kfold_splits(meta, "loocv", seed=0, splits_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.glob("test_*.csv")) == ["test_0.csv", "test_1.csv", "test_2.csv"]
