import json

import h5py
import joblib
import numpy as np
import pytest
import scanpy as sc
from sklearn.linear_model import Ridge

from gutdecoder.bench.fusion import fuse_embeddings
from gutdecoder.inference.inference import infer

GENES = ["g1", "g2"]


def test_fuse_embeddings_concat_repeats_slide_vector():
    fused = fuse_embeddings(np.ones((3, 4)), np.array([7.0, 8.0]))

    assert fused.shape == (3, 6)
    np.testing.assert_array_equal(fused[:, 4:], [[7, 8]] * 3)


def _write_run(tmp_path, slide_emb_root):
    split = tmp_path / "results" / "run" / "split0"
    split.mkdir(parents=True)
    config = {"slide_emb_root": slide_emb_root, "fusion": "concat" if slide_emb_root else None}
    (split / "config.json").write_text(json.dumps(config))
    (split / "summary.json").write_text(json.dumps({"pearson_corrs": [{"name": g} for g in GENES]}))

    n_features = 6 if slide_emb_root else 4
    rng = np.random.default_rng(0)
    reg = Ridge().fit(rng.normal(size=(20, n_features)), rng.normal(size=(20, len(GENES))))
    joblib.dump({"regression_model": reg}, split / "model.pkl")

    embed_dir = tmp_path / "tiles"
    embed_dir.mkdir()
    with h5py.File(embed_dir / "s1.h5", "w") as f:
        f["features"] = rng.normal(size=(5, 4)).astype(np.float32)
        f["coords"] = np.arange(10).reshape(5, 2) * 224

    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()
    with h5py.File(slide_dir / "s1.h5", "w") as f:
        f["embeddings"] = np.array([[1.0, 2.0]], dtype=np.float32)

    return embed_dir, slide_dir


def test_infer_fuses_slide_embedding_when_trained_with_fusion(tmp_path):
    embed_dir, slide_dir = _write_run(tmp_path, slide_emb_root="/train/slides")

    infer(str(embed_dir), "run", str(tmp_path / "out"),
          results_root=str(tmp_path / "results"), slide_emb_dir=str(slide_dir))

    adata = sc.read_h5ad(tmp_path / "out" / "adata" / "s1.h5ad")
    assert list(adata.var_names) == GENES


def test_infer_requires_slide_emb_dir_for_fusion_models(tmp_path):
    embed_dir, _ = _write_run(tmp_path, slide_emb_root="/train/slides")

    with pytest.raises(ValueError, match="slide_emb_dir is required"):
        infer(str(embed_dir), "run", str(tmp_path / "out"), results_root=str(tmp_path / "results"))


def test_infer_rejects_slide_emb_dir_for_tile_only_models(tmp_path):
    embed_dir, slide_dir = _write_run(tmp_path, slide_emb_root=None)

    with pytest.raises(ValueError, match="tile embeddings only"):
        infer(str(embed_dir), "run", str(tmp_path / "out"),
              results_root=str(tmp_path / "results"), slide_emb_dir=str(slide_dir))
