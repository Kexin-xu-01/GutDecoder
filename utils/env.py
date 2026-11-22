"""Runtime environment switches shared by notebooks and scripts."""
from __future__ import annotations

import os
import sys

__all__ = ["disable_cupy"]


def disable_cupy() -> None:
    """
    Stop CuPy from being imported (by dask, scanpy, hest, ...) in this Python process.

    Call it before importing anything that may pull in CuPy: it makes ``import cupy``
    raise ImportError and tells dask not to try. Use on nodes where the CuPy build does not
    match the GPU/driver, which otherwise crashes at import time.
    """
    if sys.modules.get("cupy") is not None:
        import warnings
        warnings.warn("cupy was already imported, so disable_cupy() cannot unload it; "
                      "call it before importing dask/scanpy/hest", RuntimeWarning, stacklevel=2)
    sys.modules["cupy"] = None
    os.environ["DASK_NO_IMPORT_CUPY"] = "1"
