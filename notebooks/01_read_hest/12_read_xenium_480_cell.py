#!/usr/bin/env python3
"""
Process Xenium imaging data using HEST (optionally cuCIM) for a single SampleID.

- Loads metadata from EXCEL_PATH and filters rows matching the provided SampleID.
- For each ROI: builds paths, optionally skips already-processed ROIs,
  loads Xenium dataset, aligns labelled cells, pools cells into pseudospots,
  filter to only the WSI area, computes summary counts, and writes outputs.
"""

from pathlib import Path
import argparse
import re

import pandas as pd
import numpy as np
import scanpy as sc
import ast

from loguru import logger

# HEST / Gutdecoder imports 
import hest
from hest import XeniumReader
from hest.HESTData import *
from hest.readers import pool_transcripts_xenium
from hest.utils import *
from gutdecoder.reader.h5ad_reader import *

# Optional: Dask for parallel/large-data workflows (not required)
import dask

# Make cuCIM optional: use it when available, silently fall back otherwise.
try:
    import cucim  # optional GPU-accelerated image operations
    logger.info("cuCIM available — GPU image ops enabled")
except ImportError:
    logger.info("cuCIM not available — falling back to CPU ops")

# Path to metadata Excel file
EXCEL_PATH = "/project/simmons_hts/kxu/hest/xenium_directory.xlsx"

# change cell_type field and the base_out_dir to save the cell adata
rename_map = {
        "cell_centroid_x": "x_centroid",
        "cell_centroid_y": "y_centroid",
        "cell": "cell_id",
        "CellAnnotation.Level0": "cell_type",
        "n_transcripts": "transcript_counts",
    }

def main():
    """Main entrypoint: parse SampleID, load metadata, and process each matching ROI."""
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Process Xenium imaging data for a given SampleID")
    parser.add_argument("SampleID", type=str, help="SampleID to process")
    args = parser.parse_args()
    sample_id = args.SampleID

    # --- Read metadata (spreadsheet with directories and ROI info) ---
    metadata = pd.read_excel(EXCEL_PATH)

    # --- Filter down to rows that match the requested SampleID ---
    sample_rows = metadata.loc[metadata["Sample_ID"] == sample_id]
    if sample_rows.empty:
        # No rows for provided SampleID -> exit early
        print(f"❌ No entry found for SampleID: {sample_id}")
        return

    # --- Iterate through all ROIs / rows for this SampleID ---
    for _, row in sample_rows.iterrows():
        sample = row["Sample_ID"]
        print(sample)

        roi = row["Roi"]
        if pd.notnull(roi):
            roi = int(float(roi))

        # Build base paths from spreadsheet columns
        exp_stem = Path(row["directory"])          # experiment root used by load_xenium_dataset
        base_img_dir = Path(row["Directory"])      # folder that contains H&E and alignment files
        image_name = Path(row["PostHnE"])          # H&E image filename (relative to base_img_dir)
        alignment_file = Path(row.get('alignment'))  # alignment filename (relative to base_img_dir)
        h5ad_path = Path(row.get('h5ad'))

        # Slide/run handling: normalize to integer slide number and optional run name
        slide_str = str(row["Slide"])
        match = re.search(r'\d+', slide_str)
        slide_num = int(match.group()) if match else None
        run_name = str(row.get("run", "")).strip()
        

        # Output root for processed Xenium outputs 
        base_out_dir = Path("/project/simmons_hts/kxu/hest/xenium_data") / f"Xenium{run_name}_cell_l0" / f"slide{slide_num}"
        hest_dir = Path("/project/simmons_hts/kxu/hest/xenium_data") / f"Xenium{run_name}" / f"slide{slide_num}"/ f"ROI{roi}"

        print(f"Processing sample={sample}, roi={roi}, slide={slide_num}, run={run_name}...")

        # --- Build absolute paths for image and alignment files ---
        img_path = base_img_dir / image_name
        alignment_file = base_img_dir / alignment_file

        # Per-ROI output folder (one folder per ROI)
        save_dir = base_out_dir / f"ROI{roi}"

        # --- Skip processing if outputs already exist (safety to avoid rework) ---
        if save_dir.exists():
            existing = list(save_dir.glob("*.h5ad")) + list(save_dir.glob("*.png"))
            if existing:
                print(f"⚠️ {sample} ({roi}) already processed at {save_dir}, skipping.")
                continue

        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Load Xenium dataset ---
        # return an object with:
        #   - .meta (dict with pixel size, spot diameter, etc.)
        #   - .wsi (whole-slide image abstraction with .width/.height)
        #   - .adata (AnnData-like structure, overwritten later)
        st = read_HESTData(
            str(hest_dir / 'aligned_adata.h5ad'), 
            str(hest_dir / 'aligned_fullres_HE.tif'), 
            str(hest_dir / 'metrics.json')
        )

        # --- Load labelled single-cell data ---
        adata_labelled = sc.read_h5ad(h5ad_path)

        # Ensure spatial coordinates are in obsm["spatial"] as Nx2 array
        adata_labelled.obs[['cell_centroid_x']] = adata_labelled.obs[["x"]]
        adata_labelled.obs[['cell_centroid_y']] = adata_labelled.obs[["y"]]
        adata_labelled.obsm["spatial"] = adata_labelled.obs[["cell_centroid_x", "cell_centroid_y"]].to_numpy()

        # --- Filter spatial transcriptomics data to WSI extent and remove codeword features ---
        # update_st_with_filtered_and_labelled modifies `st` and attaches additional fields.
        update_st_with_filtered_and_labelled(st, adata_labelled, drop_codeword=True)

        # --- Align labelled cells from their coordinate system to H&E pixel coordinates ---
        # Uses `alignment_file` and pixel size in `st.meta` to map labelled coordinates -> H&E pixels
        px_um = st.meta["pixel_size"]
        adata_labelled = align_labelled_to_he(adata_labelled, alignment_file, pixel_size_um=px_um)

        # Standardize column names (he_x / he_y etc.) in adata_labelled.obs
        # standardize column names / annotate
        adata_labelled = standardize_obs_columns(adata_labelled, rename_map = rename_map)

        # --- Build a simple cell-level dataframe expected by pooling function ---
        # `feature_name` is the transcript/feature (here: cell_type), and he_x/he_y are H&E pixel coords
        cell_df = adata_labelled.obs[["cell_type", "he_x", "he_y"]].rename(columns={"cell_type": "feature_name"})

        # --- Pool transcripts to cell-level 'spots' using HEST helper ---
        # pool_transcripts_xenium expects (df, pixel_size_um, key_x, key_y, spot_size_um)
        adata_cells_pooled = pool_transcripts_xenium(
            cell_df,
            st.meta["pixel_size_um_estimated"],
            key_x="he_x",
            key_y="he_y",
            spot_size_um=st.meta["spot_diameter"],
        )

        # Add total counts and log1p(total_counts) to obs for downstream QC/filtering
        adata_cells_pooled.obs["total_counts"] = np.asarray(adata_cells_pooled.X.sum(axis=1)).ravel()
        adata_cells_pooled.obs["log1p_total_counts"] = np.log1p(adata_cells_pooled.obs["total_counts"])

        # --- Register / downscale the full-resolution WSI to pooled grid (for overlays) ---
        downscaled_fullres, downscale_factor = register_downscale_img(
            adata_cells_pooled, st.wsi, st.meta["pixel_size_um_estimated"]
        )

        # --- Compute spot exclusion mask across the full WSI extent ---
        xy = adata_cells_pooled.obsm["spatial"]
        sx, sy = xy[:, 0], xy[:, 1]

        # WSI bounds in pixel coordinates
        xmin, xmax, ymin, ymax = 0, st.wsi.width, 0, st.wsi.height

        # Collect rules from spreadsheet row (e.g., crop_100_um)      
        rule_col = row["crop_100_um"]

        # Normalize rule_col → list[dict]
        if rule_col is None or (isinstance(rule_col, float) and pd.isna(rule_col)):
            rules = []
        elif isinstance(rule_col, dict):
            rules = [rule_col]
        elif isinstance(rule_col, list):
            rules = rule_col
        elif isinstance(rule_col, str):
            s = rule_col.strip()
            if s == "":
                rules = []
            else:
                # If the whole string is a JSON list / single JSON object, try to parse it directly
                parsed = None
                try:
                    parsed = json.loads(s)
                except Exception:
                    parsed = None

                if parsed is not None:
                    # parsed can be a dict or list
                    rules = parsed if isinstance(parsed, list) else [parsed]
                else:
                    # Fallback: extract all {...} blocks and parse each separately
                    # non-greedy match to avoid grabbing across blocks
                    dict_blocks = re.findall(r"\{.*?\}", s, flags=re.S)
                    if not dict_blocks:
                        # last resort: try ast.literal_eval on full string (may raise)
                        parsed = ast.literal_eval(s)
                        rules = parsed if isinstance(parsed, list) else [parsed]
                    else:
                        rules = []
                        for blk in dict_blocks:
                            # remove trailing commas/newlines inside the block
                            blk_clean = re.sub(r",\s*$", "", blk.strip(), flags=re.S)
                            obj = None
                            try:
                                # try JSON first (handles double quotes)
                                obj = json.loads(blk_clean)
                            except Exception:
                                try:
                                    # safe python literal parsing (handles single quotes)
                                    obj = ast.literal_eval(blk_clean)
                                except Exception as e:
                                    raise ValueError(f"Failed to parse rule block: {blk_clean!r}") from e

                            if isinstance(obj, dict):
                                rules.append(obj)
                            else:
                                # If a parsed block is a list or other, extend or wrap accordingly
                                if isinstance(obj, list):
                                    rules.extend(obj)
                                else:
                                    rules.append(obj)
        else:
            raise ValueError(f"Unsupported crop_100_um format: {type(rule_col)}")

        print("Applying exclusions:")
        for r in rules:
            print(" ", r)

        # apply_spot_exclusions returns a boolean mask indicating which spots to keep
        final_keep = apply_spot_exclusions(sx, sy, (xmin, xmax, ymin, ymax), rules)

        print(f"Spots kept after rules: {final_keep.sum()} / {len(final_keep)}")

        # --- Attach filtered pooled adata back to st and refresh metadata summaries ---
        st.adata = adata_cells_pooled[final_keep]

        # refresh_meta_counts should update st.meta with counts/summary stats and return before/after stats
        stats = refresh_meta_counts(st)
        print("Before/After:", stats)

        # --- Save outputs (overlays, .h5ad, images) via save_all helper ---
        overlay_path = save_all(st, save_dir, pyramidal=True)
        print("✔ Saved results to ", overlay_path)


if __name__ == "__main__":
    main()

