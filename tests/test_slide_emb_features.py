import h5py
import numpy as np
import pytest

from gutdecoder.vis.plot_slide_emb_umap import load_features


def _write(path, data):
    with h5py.File(path, "w") as f:
        f["features"] = data


def test_load_features_reads_requested_samples(tmp_path):
    _write(tmp_path / "A.h5", np.arange(4.0))
    _write(tmp_path / "B.h5", np.ones((1, 4)))

    df = load_features(tmp_path, sample_ids=["A", "B"])

    assert df["sample_id"].tolist() == ["A", "B"]
    assert df["features"].iloc[0].shape == (4,)


def test_load_features_fails_on_missing_requested_sample(tmp_path):
    _write(tmp_path / "A.h5", np.arange(4.0))

    # Used to print "Skipping" and silently leave the sample out of the UMAP.
    with pytest.raises(FileNotFoundError):
        load_features(tmp_path, sample_ids=["A", "missing"])
