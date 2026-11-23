import anndata as ad
import numpy as np
import pandas as pd
import pytest
from loguru import logger

from gutdecoder.vis.visualise_pred import _clean_barcode_raw, add_formatted_inference_to_adata, save_adata_from_list


def _adata(sample):
    return ad.AnnData(np.ones((2, 3)), obs=pd.DataFrame({"sample_id": [sample] * 2}, index=["a", "b"]))


def test_save_reports_saved_then_overwrote(tmp_path):
    messages = []
    handler = logger.add(messages.append, format="{level} {message}")
    try:
        paths = save_adata_from_list([_adata("s1")], str(tmp_path), "run")
        save_adata_from_list([_adata("s1")], str(tmp_path), "run")
    finally:
        logger.remove(handler)

    assert len(paths) == 1 and (tmp_path / "ST_pred_results" / "run" / "pred" / "s1.h5ad").exists()
    assert any(m.startswith("INFO Saved s1") for m in messages)
    assert any(m.startswith("INFO Overwrote s1") for m in messages)
    assert not any(m.startswith("ERROR") for m in messages)


def test_save_skips_existing_without_force(tmp_path):
    paths = save_adata_from_list([_adata("s1"), _adata("s2")], str(tmp_path), "run",
                                 existing_pred_samples={"s1"}, force_overwrite=False)

    assert [p.rsplit("/", 1)[-1] for p in paths] == ["s2.h5ad"]


@pytest.mark.parametrize("raw", [b"007x211", "007x211", ["007x211"], "b'007x211'", np.array([[b"007x211"]])])
def test_clean_barcode_raw(raw):
    assert _clean_barcode_raw(raw) == "007x211"


def _sample_adata(n_spots=3):
    obs = pd.DataFrame({"sample_id": ["S1"] * n_spots}, index=[f"bc{i}" for i in range(n_spots)])
    return ad.AnnData(np.zeros((n_spots, 2)), obs=obs, var=pd.DataFrame(index=["g1", "g2"]))


def _inference(barcodes):
    preds = pd.DataFrame({"g1": np.arange(len(barcodes), dtype=float), "g2": 1.0}, index=barcodes)
    return {"S1": {"preds": preds, "targets": preds * 2}}


def test_predictions_are_placed_by_barcode():
    adata = _sample_adata()

    summary = add_formatted_inference_to_adata([adata], _inference(["bc2", "bc0"]), verbose=False)

    assert summary["S1"]["method"] == "match_by_obs_names"
    np.testing.assert_array_equal(adata.layers["pred"][:, 0], [1.0, np.nan, 0.0])


def test_unmatchable_predictions_raise_instead_of_pairing_by_position():
    with pytest.raises(ValueError, match="by position"):
        add_formatted_inference_to_adata([_sample_adata(3)], _inference(["x", "y"]), verbose=False)


def test_sample_found_by_normalised_name():
    adata = _sample_adata()
    adata.obs["sample_id"] = "S_1"

    summary = add_formatted_inference_to_adata([adata], {"s-1": _inference(["bc0"])["S1"]}, verbose=False)

    assert summary["s-1"]["adata_index"] == 0


def test_unknown_sample_matched_by_shared_barcodes():
    other = _sample_adata()
    other.obs_names = ["x0", "x1", "x2"]

    summary = add_formatted_inference_to_adata([_sample_adata(), other], {"unknown": _inference(["x2", "x0"])["S1"]},
                                               verbose=False)

    assert summary["unknown"]["adata_index"] == 1


def test_barcode_column_used_when_obs_names_differ():
    adata = _sample_adata()
    adata.obs["barcode"] = ["c0", "c1", "c2"]

    summary = add_formatted_inference_to_adata([adata], _inference(["c2", "c0"]), verbose=False)

    assert summary["S1"]["method"] == "match_by_obs['barcode']"
    np.testing.assert_array_equal(adata.layers["pred"][:, 0], [1.0, np.nan, 0.0])


def test_duplicate_barcodes_are_averaged():
    adata = _sample_adata()

    add_formatted_inference_to_adata([adata], _inference(["bc1", "bc1", "bc1"]), verbose=False)

    assert adata.layers["pred"][1, 0] == 1.0  # mean of 0, 1, 2
