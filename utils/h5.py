"""Inspect HDF5 files (patch .h5, embeddings) without loading datasets."""
from __future__ import annotations

from pathlib import Path

__all__ = ["print_h5_tree"]


def print_h5_tree(path: str | Path) -> None:
    """Print every group and dataset in an .h5 file, indented by depth, with dataset shape and dtype."""
    import h5py

    def visit(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}- DATASET: {name}  shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{indent}- GROUP: {name}")

    with h5py.File(path, "r") as f:
        print(f"{path}")
        f.visititems(visit)
