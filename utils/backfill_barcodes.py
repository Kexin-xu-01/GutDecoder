#!/usr/bin/env python3
"""
Backfill barcodes into embedding H5 files that were generated without them.

Embedding H5 files produced by the old pipeline have:
    barcodes  (N, 1)  |S7   e.g. b'005x029'
    coords    (N, 2)  int
    embeddings (N, D) float32

Newer runs omitted barcodes because numpy unicode strings silently fail
to save in h5py. This script recovers barcodes by nearest-neighbour
matching between the H5 coords and the adata's pixel coordinates
(obs/pxl_col_in_fullres, obs/pxl_row_in_fullres).

Usage:
    python -m gutdecoder.utils.backfill_barcodes \
        --h5    /path/to/embeddings/SAMPLE.h5 \
        --adata /path/to/expression/SAMPLE.h5ad \
        [--max_dist 10]   # pixels; tiles further than this raise a warning
"""

import argparse
import numpy as np
import h5py
import scanpy as sc
from scipy.spatial import cKDTree


def backfill_barcodes(h5_path: str, adata_path: str, max_dist: float = 50.0):
    adata = sc.read_h5ad(adata_path)

    # adata pixel coordinates
    if "pxl_col_in_fullres" in adata.obs and "pxl_row_in_fullres" in adata.obs:
        adata_coords = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    elif "spatial" in adata.obsm:
        adata_coords = adata.obsm["spatial"]
    else:
        raise ValueError("adata has neither pxl_col/row_in_fullres nor obsm['spatial']")

    adata_barcodes = adata.obs_names.to_numpy()

    with h5py.File(h5_path, "r") as f:
        if "barcodes" in f:
            print(f"[INFO] {h5_path} already has 'barcodes' — skipping.")
            return
        if "coords" not in f:
            raise ValueError(f"No 'coords' key in {h5_path}")
        h5_coords = f["coords"][:]

    # match each H5 tile coord to the nearest adata spot
    tree = cKDTree(adata_coords)
    dists, idxs = tree.query(h5_coords, k=1)

    n_far = (dists > max_dist).sum()
    if n_far > 0:
        print(f"[WARN] {n_far}/{len(dists)} tiles have nearest-spot distance > {max_dist}px "
              f"(max={dists.max():.1f}). Check coordinate alignment.")

    matched_barcodes = adata_barcodes[idxs]
    bc_bytes = np.array([b.encode() for b in matched_barcodes]).reshape(-1, 1)

    with h5py.File(h5_path, "a") as f:
        f.create_dataset("barcodes", data=bc_bytes)

    print(f"[OK] Written {len(bc_bytes)} barcodes to {h5_path} "
          f"(median dist={np.median(dists):.2f}px)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",       required=True, help="Embedding H5 file to patch")
    parser.add_argument("--adata",    required=True, help="Expression H5AD with pixel coords")
    parser.add_argument("--max_dist", type=float, default=50.0,
                        help="Warn if nearest-spot distance exceeds this (pixels)")
    args = parser.parse_args()
    backfill_barcodes(args.h5, args.adata, args.max_dist)


if __name__ == "__main__":
    main()
