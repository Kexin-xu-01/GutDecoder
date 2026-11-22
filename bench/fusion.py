"""Tile + slide embedding fusion, shared by training (benchmark.py) and inference."""
import os

import h5py
import numpy as np


def read_slide_embedding(slide_h5_path, preferred_keys=("embeddings", "features", "feature")):
    """Read a slide-level h5 holding one feature vector of shape (D,), (1, D) or (D, 1)."""
    if not os.path.isfile(slide_h5_path):
        raise FileNotFoundError(f"Slide-level H5 not found: {slide_h5_path}")

    with h5py.File(slide_h5_path, "r") as f:
        key = next((k for k in preferred_keys if k in f), None)
        if key is None:
            raise KeyError(
                f"No slide embedding dataset found in {slide_h5_path}. Available keys: {list(f.keys())}"
            )
        slide_emb = np.squeeze(np.asarray(f[key][:]))

    if slide_emb.ndim != 1:
        raise ValueError(
            f"Expected slide embedding to squeeze to 1D, got shape {slide_emb.shape} from {slide_h5_path}"
        )
    return slide_emb.astype(np.float32)


def fuse_embeddings(tile_emb, slide_emb, fusion="concat"):
    """Fuse every tile embedding (n_tiles, d_tile) with one slide embedding (d_slide,)."""
    tile_emb = np.asarray(tile_emb, dtype=np.float32)
    if tile_emb.ndim != 2:
        raise ValueError(f"Tile embeddings should be 2D, got shape {tile_emb.shape}")

    slide_emb_repeated = np.repeat(slide_emb[None, :], tile_emb.shape[0], axis=0)
    if fusion == "concat":
        return np.concatenate([tile_emb, slide_emb_repeated], axis=1)
    raise ValueError(f"Unsupported fusion method: {fusion}")
