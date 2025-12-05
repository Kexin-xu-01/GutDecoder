import math
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import dask
import hest
from hest.HESTData import read_HESTData
from pathlib import Path
import torch

import warnings
import numpy as np
import tifffile as tiff

from hestcore.wsi import NumpyWSI
import scanpy as sc


def load_hest_sample(sample_dir: Path):
    """
    Load a HEST sample from a Xenium sample folder.

    Args:
        sample_dir (Path): Path to sample folder containing aligned_adata.h5ad, 
                           aligned_fullres_HE.tif, and aligned_cells.

    Returns:
        st (HESTSample): Loaded HEST sample object.
    """
    sample_dir = Path(sample_dir)

    # files
    adata_path = sample_dir / "aligned_adata.h5ad"
    image_path = sample_dir / "aligned_fullres_HE.tif"
    metrics_path = sample_dir / "metrics.json"
    
    # Look for any .geojson under tissue_seg
    tissue_seg_dir = sample_dir / "tissue_seg"
    geojson_files = list(tissue_seg_dir.glob("*.geojson")) if tissue_seg_dir.exists() else []
    tissue_contours_path = geojson_files[0] if geojson_files else None

    if not adata_path.exists():
        raise FileNotFoundError(f"Missing {adata_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing {image_path}")

    # load AnnData
    adata = sc.read_h5ad(adata_path)

    # construct HEST object (assuming HESTSample or similar API exists)
    st = read_HESTData(
        adata_path=str(adata_path),
        img=str(image_path),
        metrics_path=str(metrics_path),
        #tissue_contours_path = str(tissue_contours_path)
    )
    
    print(st)

    return st


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
        print(f"[INFO] Auto-detected {len(ids)} samples from {broad_dir}: {ids}")

    if not ids:
        raise ValueError(f"No sample folders found for segmentation in {broad_dir}")

    # iterate through each sample
    for sample_id in ids:
        sample_dir = broad_dir / sample_id
        tissue_dir = sample_dir / "tissue_seg"
        tissue_dir.mkdir(exist_ok=True, parents=True)

        try:
            # load the HEST object (using aligned_adata.h5ad + associated files)
            st = load_hest_sample(sample_dir)  # <-- replace with your loader

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*'crs' was not provided.*",
                    category=UserWarning,
                )

                st.segment_tissue(method=method,target_pxl_size=target_pxl_size)
                st.save_tissue_contours(tissue_dir, sample_id)
                st.save_tissue_vis(tissue_dir, sample_id)

            print(f"[INFO] Segmentation complete for {sample_id} → {tissue_dir}")
        except Exception as e:
            print(f"[ERROR] Failed segmentation for {sample_id}: {e}")

import math
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def show_images(
    broad_dir: Path,
    ids=None,
    subfolders=("tissue_seg",),
    recursive=True,
    max_cols=3,
    figsize=(4, 4),
    max_images=None,
    max_images_per_sample=None,
    max_display_px=1024,
    save_dir: Path | str | None = None,   # <-- NEW
    save_name: str = "image_grid.png",    # <-- NEW
    dpi: int = 200,                       # <-- NEW (optional quality)
):
    """
    Display image files from flexible subfolders under each sample directory,
    with automatic downsampling and optional saving of the final grid image.

    If save_dir is provided, the final grid will also be saved as an image there.
    """
    broad_dir = Path(broad_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

    # auto-detect sample folders if not provided
    if ids is None:
        ids = [p.name for p in broad_dir.iterdir() if p.is_dir()]
    if not ids:
        raise ValueError(f"No sample folders found in {broad_dir}")

    images, labels = [], []

    for sample_id in ids:
        sample_dir = broad_dir / sample_id
        if not sample_dir.is_dir():
            continue

        found_for_sample = []
        for sub in subfolders:
            sub_path = sample_dir / sub
            if not sub_path.exists():
                continue
            search_iter = sub_path.rglob("*") if recursive else sub_path.glob("*")
            for f in search_iter:
                if f.is_file() and f.suffix.lower() in valid_exts:
                    found_for_sample.append(f)

        found_for_sample = sorted(set(found_for_sample))
        if max_images_per_sample is not None:
            found_for_sample = found_for_sample[:max_images_per_sample]

        images.extend(found_for_sample)
        labels.extend([sample_id] * len(found_for_sample))

    if not images:
        print("[WARN] No images found across all samples.")
        return

    if max_images is not None:
        images = images[:max_images]
        labels = labels[:len(images)]

    # grid layout
    n_images = len(images)
    n_cols = min(max_cols, n_images)
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
    )
    axes = axes.flatten() if n_images > 1 else [axes]

    for ax, img_path, sample_id in zip(axes, images, labels):
        try:
            img = Image.open(img_path)
            w, h = img.size
            scale = min(max_display_px / max(w, h), 1.0)
            if scale < 1.0:
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.LANCZOS)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{sample_id}\n{img_path.name}", fontsize=8)
        except Exception as e:
            ax.axis("off")
            ax.set_title(f"Failed: {img_path.name}", fontsize=8)
            print(f"[ERROR] Could not open {img_path}: {e}")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # --- NEW: Save output image if requested ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / save_name
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Saved figure to {save_path}")

    plt.close(fig)
