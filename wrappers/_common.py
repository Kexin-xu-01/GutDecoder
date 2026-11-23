"""Sample loading and image-grid display shared by the patching and segmentation wrappers."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
from hest import HESTData
from PIL import Image
from loguru import logger


def load_hest_sample(sample_dir: Path):
    """
    Load a HEST sample from a Xenium sample folder containing aligned_adata.h5ad,
    aligned_fullres_HE.tif, optional metrics.json and an optional tissue_seg/*.geojson.
    """
    sample_dir = Path(sample_dir)
    adata_path = sample_dir / "aligned_adata.h5ad"
    image_path = sample_dir / "aligned_fullres_HE.tif"
    metrics_path = sample_dir / "metrics.json"

    tissue_seg_dir = sample_dir / "tissue_seg"
    geojson_files = list(tissue_seg_dir.glob("*.geojson")) if tissue_seg_dir.exists() else []
    tissue_contours_path = geojson_files[0] if geojson_files else None

    if not adata_path.exists():
        raise FileNotFoundError(f"Missing {adata_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing {image_path}")

    st = HESTData.from_paths(
        adata_path=str(adata_path),
        img=str(image_path),
        metrics_path=str(metrics_path),
        tissue_contours_path=str(tissue_contours_path) if tissue_contours_path else None,
    )
    print(st)
    return st


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
    save_dir: Path | str | None = None,
    save_name: str = "image_grid.png",
    dpi: int = 200,
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
        logger.warning("No images found across all samples.")
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
        except OSError as e:  # unreadable or not an image; show the gap in the grid
            ax.axis("off")
            ax.set_title(f"Failed: {img_path.name}", fontsize=8)
            logger.error(f"Could not open {img_path}: {e}")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / save_name
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    plt.close(fig)
