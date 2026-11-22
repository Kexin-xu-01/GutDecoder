import h5py
import numpy as np
import pytest

from gutdecoder.wrappers import patching_wrapper, segment_wrapper


def _broken_loader(sample_dir):
    raise FileNotFoundError(f"aligned_adata.h5ad missing in {sample_dir}")


def test_patching_reports_every_failed_sample(tmp_path, monkeypatch):
    (tmp_path / "A").mkdir()
    monkeypatch.setattr(patching_wrapper, "load_hest_sample", _broken_loader)

    with pytest.raises(RuntimeError, match="Patching failed for 2 of 2 samples") as err:
        patching_wrapper.patch_hest_samples(tmp_path, ids=["A", "missing"], verbose=0)

    assert "A: FileNotFoundError" in str(err.value)
    assert "missing: folder does not exist" in str(err.value)


def test_segmentation_reports_every_failed_sample(tmp_path, monkeypatch):
    (tmp_path / "A").mkdir()
    monkeypatch.setattr(segment_wrapper, "load_hest_sample", _broken_loader)

    with pytest.raises(RuntimeError, match="Segmentation failed for 1 of 1 samples"):
        segment_wrapper.segment_hest_tissue(tmp_path, ids=["A"])


def test_count_patches_does_not_skip_unreadable_files(tmp_path):
    (tmp_path / "A" / "patches").mkdir(parents=True)
    with h5py.File(tmp_path / "A" / "patches" / "A.h5", "w") as f:
        f["img"] = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    (tmp_path / "B" / "patches").mkdir(parents=True)
    (tmp_path / "B" / "patches" / "B.h5").write_text("not hdf5")

    with pytest.raises(OSError):
        patching_wrapper.count_patches(tmp_path)
