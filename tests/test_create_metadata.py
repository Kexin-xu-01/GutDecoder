import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from gutdecoder.utils.create_metadata import _read_barcodes_from_patch_h5, add_qc_from_all_runs, return_qc


def _write_sample(sample_dir, barcodes_in_patches):
    sample_dir.mkdir(parents=True)
    obs = pd.DataFrame(
        {"total_counts": [10.0, 20.0, 30.0, 40.0], "n_genes_by_counts": [1.0, 2.0, 3.0, 4.0]},
        index=["c1", "c2", "c3", "c4"],
    )
    ad.AnnData(np.zeros((4, 2)), obs=obs).write_h5ad(sample_dir / "aligned_adata.h5ad")
    (sample_dir / "patches").mkdir()
    with h5py.File(sample_dir / "patches" / "patches.h5", "w") as f:
        f["barcode"] = np.array([[b.encode()] for b in barcodes_in_patches])


def test_return_qc_uses_patch_barcodes_as_tissue_mask(tmp_path):
    _write_sample(tmp_path / "slide" / "A", ["c1", "c3"])

    row = return_qc(tmp_path / "slide").iloc[0]

    assert row["sample_id"] == "A"
    assert row["n_obs"] == 4
    assert row["n_obs_in_tissue"] == 2
    assert row["mean_total_counts_in_tissue"] == 20.0
    assert row["pct_in_tissue"] == 50.0


def test_add_qc_reports_every_failed_slide_after_processing_the_rest(tmp_path):
    run = tmp_path / "xenium" / "XeniumPR1"
    _write_sample(run / "slide1" / "A", ["c1"])
    broken = run / "slide2" / "B"
    broken.mkdir(parents=True)
    (broken / "aligned_adata.h5ad").write_text("not an h5ad file")

    with pytest.raises(RuntimeError, match=r"QC failed on 1 slide\(s\):\n.*slide2"):
        add_qc_from_all_runs(pd.DataFrame({"sample_id": []}), tmp_path / "xenium", tmp_path / "visium",
                             save_csv_name=None, verbose=False)


@pytest.mark.parametrize("key, data", [
    ("barcode", np.array([[b"c1 "], [b"c2"]])),        # bytes, (N, 1)
    ("barcodes", np.array(["c1", "c2"], dtype="S")),   # bytes, (N,)
    ("meta/barcode", np.array([b"c1", b"c2"])),        # one group down
])
def test_read_barcodes_handles_known_layouts(tmp_path, key, data):
    path = tmp_path / "patches.h5"
    with h5py.File(path, "w") as f:
        f[key] = data

    assert _read_barcodes_from_patch_h5(path) == ["c1", "c2"]


def test_read_barcodes_warns_instead_of_guessing_another_dataset(tmp_path):
    path = tmp_path / "patches.h5"
    with h5py.File(path, "w") as f:
        f["coords"] = np.zeros((2, 1))  # used to be mistaken for barcodes
        f["names"] = np.array([b"x", b"y"])

    with pytest.warns(UserWarning, match="No barcode dataset"):
        assert _read_barcodes_from_patch_h5(path) == []


def test_read_barcodes_warns_when_barcode_is_a_group(tmp_path):
    path = tmp_path / "patches.h5"
    with h5py.File(path, "w") as f:
        f.create_group("barcode")

    with pytest.warns(UserWarning, match="is a group"):
        assert _read_barcodes_from_patch_h5(path) == []


@pytest.mark.filterwarnings("error::FutureWarning")
def test_add_qc_appends_new_samples_without_pandas_concat_warning(tmp_path):
    _write_sample(tmp_path / "xenium" / "XeniumPR1" / "slide1" / "A", ["c1", "c2"])
    metadata = pd.DataFrame({"sample_id": ["Other"], "tissue": ["colon"]})

    meta = add_qc_from_all_runs(metadata, tmp_path / "xenium", tmp_path / "visium",
                                save_csv_name=None, verbose=False)["metadata"]

    new = meta.set_index("sample_id").loc["XeniumPR1S1A"]
    assert new["n_obs_in_tissue"] == 2
    assert pd.isna(new["tissue"])
    assert meta.set_index("sample_id").loc["Other", "tissue"] == "colon"
