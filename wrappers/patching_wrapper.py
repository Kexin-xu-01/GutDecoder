import pandas as pd
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import dask


import numpy as np
import math

import scanpy as sc

# import monkeypatch for qc
from gutdecoder.utils.patch.qc_patch import apply_qc_patch
apply_qc_patch(fallback=5) 

import hest
from hest.HESTData import read_HESTData
from hest import iter_hest

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
        tissue_contours_path=str(tissue_contours_path)
        # xenium_cell_path=str(cells_path) # no need
    )
    
    print(st)

    return st

def patch_hest_samples(
    root: Path = Path("/project/simmons_hts/kxu/hest/xenium_data/broad/"),
    ids=None,
    target_patch_size: int = 224,
    target_pixel_size: float = 0.5,
    threshold: float = 0.15,
    verbose: int = 1,
):
    """
    Extract H&E patches for Xenium-format HEST samples and save under
    <sample_dir>/patches and <sample_dir>/patches_vis (per sample).

    Expected layout for each sample folder:
        <sample_id>/
          aligned_adata.h5ad
          aligned_fullres_HE.tif
          aligned_cells.h5ad
          metrics.json (optional)
          tissue_seg/                 (optional; used as mask if present)
    """
    root = Path(root)

    # auto-detect sample IDs (folder names)
    if ids is None:
        ids = sorted([p.name for p in root.iterdir() if p.is_dir()])
        if verbose:
            print(f"[INFO] Auto-detected {len(ids)} sample folders: {ids}")

    if not ids:
        raise ValueError("No sample folders found to patch under the specified root.")

    for sample_id in ids:
        sample_dir = root / sample_id
        if not sample_dir.exists():
            print(f"[WARN] Skipping {sample_id}: folder does not exist at {sample_dir}")
            continue

     # --- skip if any patch .h5 already exists under the sample (e.g. ROI/*/patches/*.h5) ---
        existing_patch_files = list(sample_dir.glob("**/patches/*.h5"))
        if existing_patch_files:
            if verbose:
                print(f"[INFO] Skipping {sample_id}: found existing patch files ({len(existing_patch_files)}) — e.g. {existing_patch_files[0]}")
            continue

        try:
            # Load the sample via your helper
            st = load_hest_sample(sample_dir)

            # Per-sample output dirs
            sample_patches_dir = sample_dir / "patches"
            sample_patches_vis_dir = sample_dir / "patches_vis"
            sample_patches_dir.mkdir(exist_ok=True, parents=True)
            sample_patches_vis_dir.mkdir(exist_ok=True, parents=True)

            # Where files will end up
            patch_save_path = sample_patches_dir / f"{sample_id}.h5"
            vis_save_path = sample_patches_vis_dir / f"{sample_id}_patch_vis.png"

            # Optional tissue segmentation mask
            tissue_seg_dir = sample_dir / "tissue_seg"
            has_tissue_seg = tissue_seg_dir.exists() and tissue_seg_dir.is_dir()
            if verbose:
                print(f"[INFO] {sample_id}: tissue_seg present = {has_tissue_seg}")

            dump_kwargs = dict(
                patch_save_dir=str(sample_patches_dir),
                name=sample_id,
                target_patch_size=target_patch_size,
                target_pixel_size=target_pixel_size,
                verbose=verbose,
                dump_visualization=True,
                use_mask=bool(has_tissue_seg),
                threshold=threshold,
            )

            st.dump_patches(**dump_kwargs)

            # Move visualization into sample's patches_vis
            default_vis_path = sample_patches_dir / f"{sample_id}_patch_vis.png"
            if default_vis_path.exists():
                default_vis_path.rename(vis_save_path)

            if verbose:
                print(f"[INFO] Patches saved for {sample_id}: {patch_save_path}")
                print(f"[INFO] Patch visualization saved: {vis_save_path}")

        except FileNotFoundError as e:
            print(f"[ERROR] Missing required file(s) for {sample_id}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to patch {sample_id}: {e}")
            

def show_images(
    broad_dir: Path,
    ids=None,
    subfolders=("tissue_seg",),   # one or many subfolders under each sample
    recursive=True,               # search recursively inside each subfolder
    max_cols=3,
    figsize=(4, 4),               # size of each grid cell (in inches)
    max_images=None,              # optional global cap on total images
    max_images_per_sample=None,   # optional cap per sample
    save_dir: Path | str | None = None,   # <-- NEW
    save_name: str = "patches_vis.png",  
    max_display_px=1024,          # max width/height for displayed images
):
    """
    Display image files from flexible subfolders under each sample directory,
    with automatic downsampling to ensure grid display fits nicely.

    Args:
        broad_dir (Path): path to xenium_data/broad directory (contains sample folders)
        ids (list[str], optional): specific sample folder names; if None, auto-detect all
        subfolders (tuple[str]): subfolder names under each sample
        recursive (bool): recurse within each subfolder
        max_cols (int): max columns in the grid
        figsize (tuple): (width, height) in inches for each subplot
        max_images (int|None): cap total number of images displayed
        max_images_per_sample (int|None): cap images per sample
        max_display_px (int): maximum width/height in pixels for display
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

            # downsample large images for display only
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

    # hide unused axes
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # --- NEW: Save output image if requested ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / save_name
        fig.savefig(save_path, bbox_inches="tight")
        print(f"✅ Saved figure to {save_path}")

    plt.close(fig)


def count_patches(broad_root, save_csv=None):
    """
    Count patches for each sample when patches are stored inside
    <sample_dir>/patches/*.h5.

    Args:
        broad_root (str or Path): root directory containing sample folders
        save_csv (str or Path, optional): if given, save counts table to this CSV

    Returns:
        pandas.DataFrame
    """
    broad_root = Path(broad_root)
    results = []

    # iterate over sample folders
    for sample_dir in sorted(broad_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        patches_dir = sample_dir / "patches"
        if not patches_dir.exists():
            continue

        # expect one .h5 per sample (but support >1 just in case)
        for h5_file in patches_dir.glob("*.h5"):
            sample_id = sample_dir.name  # folder name is sample ID
            try:
                with h5py.File(h5_file, "r") as f:
                    num_patches = f["img"].shape[0]
                results.append({"sample_id": sample_id, "num_patches": num_patches})
            except Exception as e:
                print(f"[ERROR] Failed reading {h5_file}: {e}")

    df = pd.DataFrame(results).sort_values("sample_id")
    total_patches = df["num_patches"].sum()

    print(df)
    print(f"\nTotal patches across all samples: {total_patches}")

    if save_csv:
        df.to_csv(broad_root / save_csv, index=False)
        print(f"[INFO] Saved counts to {broad_root/save_csv}")

    return df



def visualise_patch_block(
    h5_file_path,
    jpeg_path,                       # path to downscaled_fullres.jpeg
    fullres_size=None,               # (H_full, W_full); if None we infer from coords
    block_size=3,
    highlight_color="yellow",
    max_display_side_px=2000,        # cap JPEG display size for plotting
    selection_method="auto",         # 'auto' (try strict then fallback), 'strict', or 'nearest'
):
    """
    Visualize a consecutive block of patches and their locations using *downscaled_fullres.jpeg*,
    with patch coordinates stored in ORIGINAL WSI pixels (no giant TIFF reads).

    Args:
        h5_file_path (str|Path): patch .h5 file (expects datasets: img, coords/xy, optional barcode)
        jpeg_path (str|Path): path to downscaled_fullres.jpeg (background)
        fullres_size (tuple|None): (H_full, W_full) of the original WSI. If None, infer from coords.
        block_size (int): block side length (block_size x block_size)
        highlight_color (str): rectangle color for selected patches
        max_display_side_px (int): cap long edge of displayed JPEG to keep figure reasonable
        selection_method (str): 'auto' (default) tries strict-grid then falls back to nearest-neighbour,
                                'strict' forces the original exact-grid behaviour (may raise if no block),
                                'nearest' forces nearest-neighbour selection.
    """
    h5_file_path = Path(h5_file_path)
    jpeg_path = Path(jpeg_path)

    # ---------- Load minimal H5 metadata (NO full arrays) ----------
    with h5py.File(h5_file_path, "r") as f:
        # coords: accept several common keys
        if "coords" in f:
            coords = f["coords"][...]            # (N, 2) top-left (x, y) at FULL-RES pixels
        elif "xy" in f:
            coords = f["xy"][...]
        else:
            raise KeyError("No 'coords' or 'xy' dataset in the H5 file.")

        # patch size from img dataset shape (N, H, W, [C])
        ish = f["img"].shape
        patch_h, patch_w = int(ish[1]), int(ish[2])

        has_barcodes = "barcode" in f
        barcodes_ds = f["barcode"] if has_barcodes else None

        # Build helpful lookup structures
        xs = np.unique(coords[:, 0]); xs.sort()
        ys = np.unique(coords[:, 1]); ys.sort()
        coord_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}

        desired_n = block_size * block_size
        chosen_indices = np.array([], dtype=int)
        chosen_xy = np.zeros((0, 2), dtype=int)
        used_fallback = False

        def strict_grid_selection():
            """Attempt strict grid selection using unique xs/ys indices."""
            max_x_start = len(xs) - block_size
            max_y_start = len(ys) - block_size
            sel_idx = []
            sel_xy = []
            if max_x_start > 0 and max_y_start > 0:
                start_x_idx = np.random.randint(0, max_x_start)
                start_y_idx = np.random.randint(0, max_y_start)
                for yi in range(start_y_idx, start_y_idx + block_size):
                    for xi in range(start_x_idx, start_x_idx + block_size):
                        x = int(xs[xi]); y = int(ys[yi])
                        if (x, y) in coord_to_idx:
                            idx = coord_to_idx[(x, y)]
                            sel_idx.append(idx)
                            sel_xy.append((x, y))
            return np.array(sel_idx, dtype=int), np.array(sel_xy, dtype=int)

        def nearest_neighbour_block():
            """Select nearest `desired_n` neighbours around a random seed and order them row->col."""
            if coords.shape[0] < desired_n:
                # Not enough patches in total
                nn_idx = np.argsort(((coords - coords.mean(axis=0))**2).sum(axis=1))
                nn_idx = nn_idx[:coords.shape[0]]
            else:
                seed_idx = np.random.randint(0, coords.shape[0])
                seed_xy = coords[seed_idx].astype(np.float64)
                diffs = coords.astype(np.float64) - seed_xy[None, :]
                d2 = (diffs ** 2).sum(axis=1)
                nn_idx = np.argsort(d2)[:desired_n]

            nn_coords = coords[nn_idx].astype(int)

            # Try to create a row ordering. Compute approximate row height from y-values
            y_vals_unique = np.unique(nn_coords[:, 1])
            if len(y_vals_unique) > 1:
                sorted_y = np.sort(y_vals_unique)
                dy = np.diff(sorted_y)
                # if all dy equal zero (unlikely) fallback to patch_h
                median_dy = int(np.median(dy)) if len(dy) >= 1 else patch_h
                row_height = max(1, median_dy)
            else:
                row_height = patch_h

            row_ids = ((nn_coords[:, 1] - nn_coords[:, 1].min()) // row_height).astype(int)
            order = np.lexsort((nn_coords[:, 0], row_ids))  # sort by row (y), then x
            ordered_idx = nn_idx[order]
            return np.array(ordered_idx, dtype=int), coords[ordered_idx].astype(int)

        # Selection strategy
        if selection_method == "strict":
            chosen_indices, chosen_xy = strict_grid_selection()
            used_fallback = False
        elif selection_method == "nearest":
            chosen_indices, chosen_xy = nearest_neighbour_block()
            used_fallback = True
        else:  # auto: try strict, then fallback
            chosen_indices, chosen_xy = strict_grid_selection()
            if chosen_indices.size < desired_n:
                chosen_indices, chosen_xy = nearest_neighbour_block()
                used_fallback = True

        # If even after fallback we have fewer than desired, clamp to available
        if chosen_indices.size == 0:
            # nothing selected; raise a clearer error
            raise ValueError(f"Could not select any patches (found {coords.shape[0]} total). "
                             "Check that coords are present and non-empty.")
        if chosen_indices.size < desired_n:
            print(f"[WARN] only {chosen_indices.size}/{desired_n} patches available for display; "
                  "display will be partially empty.")
        print(f"Selected {len(chosen_indices)} patches (fallback used: {used_fallback})")

        # ---------- (1) Patch block grid (stream from HDF5) ----------
        plt.figure(figsize=(3 * block_size, 3 * block_size))
        k = 0
        for j in range(block_size):
            for i in range(block_size):
                ax = plt.subplot(block_size, block_size, j * block_size + i + 1)
                if k < len(chosen_indices):
                    idx = int(chosen_indices[k])
                    patch = f["img"][idx]  # read only this patch
                    ax.imshow(patch)
                    if has_barcodes:
                        b = barcodes_ds[idx]
                        title = b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b)
                        ax.set_title(title, fontsize=6)
                else:
                    # empty cell
                    ax.axis("off")
                k += 1
        plt.suptitle(f"Patch Block (method='{selection_method}', fallback={used_fallback})")
        plt.tight_layout()
        plt.show()

    # ---------- (2) Overlay on downscaled_fullres.jpeg (no TIFF) ----------
    jpeg_img = Image.open(jpeg_path).convert("RGB")
    Wj, Hj = jpeg_img.size  # PIL: (width, height)

    # Determine full-res WSI (H_full, W_full) → used ONLY for scaling coords to JPEG space
    if fullres_size is not None:
        Hf, Wf = map(int, fullres_size)
    else:
        # Infer from max coords + patch size (works well if coords cover slide)
        xmax = int(coords[:, 0].max() + patch_w)
        ymax = int(coords[:, 1].max() + patch_h)
        Wf, Hf = xmax, ymax
        print("[INFO] fullres_size not provided; inferred from coords as "
              f"(H_full={Hf}, W_full={Wf}). Pass fullres_size=(H,W) for exact scaling.")

    # Scale factors: FULL-RES → JPEG
    sx = Wj / float(Wf) if Wf > 0 else 1.0
    sy = Hj / float(Hf) if Hf > 0 else 1.0

    # Map selected coords + patch size to JPEG space
    # chosen_xy may contain fewer than desired_n; that's OK
    chosen_xy_disp = chosen_xy.astype(np.float64).copy()
    chosen_xy_disp[:, 0] *= sx
    chosen_xy_disp[:, 1] *= sy
    pw_disp = patch_w * sx
    ph_disp = patch_h * sy

    # Downscale the displayed JPEG if it's huge so the figure always renders fully
    long_edge = max(Wj, Hj)
    disp_scale = 1.0 if long_edge <= max_display_side_px else (max_display_side_px / float(long_edge))
    if disp_scale < 1.0:
        new_w = int(Wj * disp_scale)
        new_h = int(Hj * disp_scale)
        jpeg_disp = jpeg_img.resize((new_w, new_h), Image.LANCZOS)
    else:
        jpeg_disp = jpeg_img
        new_w, new_h = Wj, Hj

    # Apply display scale to coords/size
    chosen_xy_disp *= disp_scale
    pw_d = pw_disp * disp_scale
    ph_d = ph_disp * disp_scale

    # Draw only the chosen block on the JPEG background (fast)
    dpi = 100
    fig_w_in, fig_h_in = new_w / dpi, new_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax.imshow(jpeg_disp)
    ax.set_axis_off()

    for (x, y) in chosen_xy_disp:
        ax.add_patch(Rectangle((x, y), pw_d, ph_d, fill=False,
                               linewidth=1.0, edgecolor=highlight_color))
        ax.plot(x + pw_d / 2, y + ph_d / 2, "o", ms=3, mec="black", mfc=highlight_color)

    plt.tight_layout(pad=0)
    plt.show()


def visualise_first_n_patches(h5_file_path, wsi_path=None, max_patches=20, highlight_color="yellow"):
    """
    Visualize patches and their locations on the original WSI.

    Args:
        h5_file_path (str or Path): path to the patch .h5 file
        wsi_path (str or Path, optional): path to the downscaled WSI image for background
        max_patches (int): maximum number of patches to display in image grid
        highlight_color (str): color to highlight selected patches on WSI
    """
    h5_file_path = Path(h5_file_path)
    
    with h5py.File(h5_file_path, "r") as f:
        patches = f["img"][:]         # shape: (num_patches, H, W, C)
        coords = f["coords"][:]       # shape: (num_patches, 2) top-left x, y
        barcodes = f["barcode"][:]    # barcodes (optional)

    num_patches = patches.shape[0]
    print(f"Loaded {num_patches} patches, showing first {min(max_patches, num_patches)}")

    # -------------------
    # 1️⃣ Show a few patches
    # -------------------
    n_show = min(max_patches, num_patches)
    ncols = min(5, n_show)
    nrows = (n_show + ncols - 1) // ncols
    plt.figure(figsize=(3*ncols, 3*nrows))
    for i in range(n_show):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(patches[i])
        plt.axis("off")
        plt.title(barcodes[i].decode("utf-8") if isinstance(barcodes[i], bytes) else str(barcodes[i]), fontsize=8)
    plt.suptitle("Sample Patches")
    plt.tight_layout()
    plt.show()

    # -------------------
    # 2️⃣ Show patch locations on WSI with highlights
    # -------------------
    if wsi_path is not None:
        wsi = Image.open(wsi_path)
        wsi = np.array(wsi)

        plt.figure(figsize=(8,8))
        plt.imshow(wsi)

        # plot ALL patch locations in red
        plt.scatter(coords[:,0], coords[:,1], s=10, c="red", alpha=0.3, label="All patches")

        # highlight SELECTED patches (first n_show) in different color
        selected_coords = coords[:n_show]
        plt.scatter(selected_coords[:,0], selected_coords[:,1],
                    s=40, c=highlight_color, edgecolor="black", label="Highlighted patches")

        plt.title("Patch Locations on WSI")
        plt.axis("off")
        plt.legend()
        plt.show()
    else:
        print("[INFO] wsi_path not provided, skipping WSI location overlay.")
