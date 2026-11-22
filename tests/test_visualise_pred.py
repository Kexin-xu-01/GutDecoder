import anndata as ad
import numpy as np
import pandas as pd

from gutdecoder.vis.visualise_pred import save_adata_from_list


def _adata(sample):
    return ad.AnnData(np.ones((2, 3)), obs=pd.DataFrame({"sample_id": [sample] * 2}, index=["a", "b"]))


def test_save_reports_saved_then_overwrote(tmp_path, capsys):
    paths = save_adata_from_list([_adata("s1")], str(tmp_path), "run")
    save_adata_from_list([_adata("s1")], str(tmp_path), "run")

    out = capsys.readouterr().out
    assert len(paths) == 1 and (tmp_path / "ST_pred_results" / "run" / "pred" / "s1.h5ad").exists()
    assert "[OK] Saved s1" in out and "[OK] Overwrote s1" in out
    assert "[ERROR]" not in out


def test_save_skips_existing_without_force(tmp_path):
    paths = save_adata_from_list([_adata("s1"), _adata("s2")], str(tmp_path), "run",
                                 existing_pred_samples={"s1"}, force_overwrite=False)

    assert [p.rsplit("/", 1)[-1] for p in paths] == ["s2.h5ad"]
