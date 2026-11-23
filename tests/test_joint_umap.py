import h5py
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from gutdecoder.vis import plot_slide_emb_umap as umap_pdf


def _group(tmp_path, name, shift, n=6):
    rng = np.random.default_rng(ord(name))
    root = tmp_path / name
    root.mkdir()
    rows = []
    for i in range(n):
        sid = f"{name}{i}"
        with h5py.File(root / f"{sid}.h5", "w") as f:
            f["features"] = rng.normal(loc=shift, size=8)
        rows.append(dict(sid=sid, Sample_ID=f"P{i}", tissue=["colon", "ileum"][i % 2], age=30 + i))
    meta = tmp_path / f"{name}.csv"
    pd.DataFrame(rows).to_csv(meta, index=False)
    return root, meta


def test_load_joint_groups_keeps_group_order_and_labels(tmp_path):
    (a_root, a_meta), (b_root, b_meta) = _group(tmp_path, "a", 0), _group(tmp_path, "b", 1, n=3)

    joint = umap_pdf._load_joint_groups(
        [dict(h5_root=a_root, metadata_path=a_meta, id_col="sid", label="A", sample_ids=None),
         dict(h5_root=b_root, metadata_path=b_meta, id_col="sid", label="B", sample_ids=None)],
        feature_key="features",
    )

    assert joint["source"].tolist() == ["A"] * 6 + ["B"] * 3
    assert joint["tissue"].notna().all()


def test_two_group_pdf_with_matched_distances(tmp_path):
    # Harmony uses about n_samples / 30 clusters, so it needs a few dozen samples.
    (a_root, a_meta), (b_root, b_meta) = _group(tmp_path, "a", 0, n=30), _group(tmp_path, "b", 1, n=30)
    out = tmp_path / "out" / "umap.pdf"

    joint = umap_pdf.make_joint_group_umap_pdf(
        a_root, b_root, a_meta, b_meta, out, "CZI", "NDPI", "sid", "sid",
        categorical_cols=["tissue"], numeric_cols=["age"], match_col="Sample_ID",
        use_harmony=True, harmony_n_pca=4,
    )

    assert out.exists() and (out.parent / "umap_distances.csv").exists()
    assert {"UMAP1", "UMAP2", "Harmony_1"} <= set(joint.columns)
    assert set(joint.attrs["distance_summary"]["embedding_space"]) == {"original", "pre_harmony", "post_harmony"}


def test_three_group_pdf(tmp_path):
    groups = [_group(tmp_path, name, shift) for name, shift in (("a", 0), ("b", 1), ("c", -1))]
    out = tmp_path / "out" / "umap3.pdf"

    joint = umap_pdf.make_joint_three_group_umap_pdf(
        *(g[0] for g in groups), *(g[1] for g in groups), out, "A", "B", "C", "sid", "sid", "sid",
        categorical_cols=["tissue"], numeric_cols=["age"],
    )

    assert out.exists()
    assert joint["source"].value_counts().to_dict() == {"A": 6, "B": 6, "C": 6}
    assert joint[["UMAP1", "UMAP2"]].notna().all().all()


def test_harmony_needs_enough_components(tmp_path):
    (a_root, a_meta), (b_root, b_meta) = _group(tmp_path, "a", 0), _group(tmp_path, "b", 1)

    with pytest.raises(ValueError, match="Not enough samples/components"):
        umap_pdf.make_joint_group_umap_pdf(a_root, b_root, a_meta, b_meta, tmp_path / "u.pdf", "A", "B", "sid", "sid",
                                           use_harmony=True, harmony_n_pca=1)
