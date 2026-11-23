import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from gutdecoder.inference import spatial_plot as sp


@pytest.mark.parametrize("to_matrix", [np.asarray, sparse.csr_matrix])
def test_clip_negative_values_to_zero(to_matrix):
    adata = ad.AnnData(to_matrix(np.array([[-1.0, 2.0], [0.5, -0.1]])))

    sp.clip_negative_values_to_zero(adata, "s")

    x = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    np.testing.assert_array_equal(x, [[0.0, 2.0], [0.5, 0.0]])


class FakeWSI:
    width, height = 4000, 2000

    def get_thumbnail(self, width, height):
        return np.zeros((height, width, 4), dtype=np.uint8)


def test_manual_registration_uses_wsi_size_and_thumbnail():
    adata = ad.AnnData(np.zeros((2, 1)), obsm={"spatial": np.array([[0.0, 0.0], [224.0, 0.0]])})

    thumb, factor = sp.manual_register_downscale_img(adata, FakeWSI(), sample_name="s", max_side=2000)

    assert thumb.shape == (1000, 2000, 3)
    assert factor == 2.0
    assert adata.uns["spatial"]["s"]["scalefactors"]["tissue_hires_scalef"] == 0.5


def test_process_run_raises_after_trying_every_sample(tmp_path, monkeypatch):
    (tmp_path / "adata").mkdir()
    for name in ("a", "b", "c"):
        ad.AnnData(np.zeros((1, 1))).write_h5ad(tmp_path / "adata" / f"{name}.h5ad")

    attempted = []

    def fake_add(h5ad, wsi_path, run_dir, pixel_size):
        attempted.append(h5ad)
        raise ValueError("bad slide")

    monkeypatch.setattr(sp, "_find_matching_wsi", lambda name, root: None if name == "b" else f"{name}.ndpi")
    monkeypatch.setattr(sp, "add_image_to_adata_and_save", fake_add)

    with pytest.raises(RuntimeError, match="3 of 3 samples failed") as err:
        sp.process_run(str(tmp_path), wsi_root="wsis")

    assert len(attempted) == 2
    assert "b: no unambiguous matching WSI" in str(err.value)
    assert "a: ValueError: bad slide" in str(err.value)
