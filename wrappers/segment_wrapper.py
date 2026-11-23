import math
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import dask
import torch

import warnings
import numpy as np
import tifffile as tiff


import scanpy as sc


from gutdecoder.utils.patch.qc_patch import apply_qc_patch
apply_qc_patch(fallback=None)

from gutdecoder.wrappers._common import load_hest_sample, show_images  # noqa: F401  (re-exported for notebooks)
from loguru import logger


def segment_hest_tissue(hest_root: Path, ids=None, method="deep",target_pxl_size=1):
    """
    Perform tissue segmentation on HEST Xenium samples inside hest/xenium_data/broad,
    saving results into a tissue_seg folder inside each sample folder.
    Suppresses pyogrio CRS warnings.

    Args:
        hest_root (Path): root directory of HEST dataset
        ids (list, optional): list of sample folder names to segment
        method (str): tissue segmentation method ("otsu" recommended)
    """
    hest_root = Path(hest_root)
    broad_dir = hest_root 

    # auto-detect sample folders if not provided
    if ids is None:
        ids = [p.name for p in broad_dir.iterdir() if p.is_dir()]
        logger.info(f"Auto-detected {len(ids)} samples from {broad_dir}: {ids}")

    if not ids:
        raise ValueError(f"No sample folders found for segmentation in {broad_dir}")

    failures = []
    for sample_id in ids:
        sample_dir = broad_dir / sample_id
        tissue_dir = sample_dir / "tissue_seg"
        tissue_dir.mkdir(exist_ok=True, parents=True)

        try:
            # load the HEST object (using aligned_adata.h5ad + associated files)
            st = load_hest_sample(sample_dir)  
            logger.info('hest sample loaded')

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*'crs' was not provided.*",
                    category=UserWarning,
                )

                st.segment_tissue(method=method,target_pxl_size=target_pxl_size)
                logger.info('tissue segmentation completed')
                st.save_tissue_contours(tissue_dir, sample_id)
                logger.info('tissue contour saved')
                st.save_tissue_vis(tissue_dir, sample_id)
                logger.info('tissue visualisation saved')

            logger.info(f"Segmentation complete for {sample_id} → {tissue_dir}")
        except Exception as e:
            failures.append(f"{sample_id}: {type(e).__name__}: {e}")

    if failures:
        raise RuntimeError(f"Segmentation failed for {len(failures)} of {len(ids)} samples:\n" + "\n".join(failures))
