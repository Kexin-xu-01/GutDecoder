#!/usr/bin/env python3
"""
Script to process Xenium imaging data using Hest and CuCIM,
with metadata loaded from an Excel file and SampleID provided as a CLI argument.
"""

from pathlib import Path
import pandas as pd
import hest
from hest import XeniumReader
import argparse
import re

# Optional: Dask for parallel processing
import dask

from hest.HESTData import *
from hest import h5ad_reader
from hest.h5ad_reader import *


# Path to metadata Excel file
EXCEL_PATH = "/project/simmons_hts/kxu/hest/xenium_directory.xlsx"


def main():
    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="Process Xenium imaging data for a given SampleID")
    parser.add_argument("SampleID", type=str, help="SampleID to process")
    args = parser.parse_args()
    sample_id = args.SampleID

    # --- Load metadata ---
    metadata = pd.read_excel(EXCEL_PATH)

    # --- Filter metadata ---
    sample_rows = metadata.loc[metadata["Sample_ID"] == sample_id]

    if sample_rows.empty:
        print(f"❌ No entry found for SampleID: {sample_id}")
        return

    # --- Process each ROI for that SampleID ---
    for _, row in sample_rows.iterrows():
        sample = row["Sample_ID"]
        print(sample)
        roi = row["Roi"]
        if pd.notnull(roi):
            roi = int(float(roi))
        exp_stem = Path(row["directory"])
        base_img_dir = Path(row["Directory"])

        # Choose slide directory (slide1, slide2, etc.)
        slide_str = str(row["Slide"])  # ensure it's a string
        match = re.search(r'\d+', slide_str)
        if match:
            slide_num = int(match.group())
        run_name = str(row.get("run", "")).strip()
        base_out_dir = Path("/project/simmons_hts/kxu/hest/xenium_data") / f"Xenium{run_name}" / f"slide{slide_num}" 

        print(f"Processing sample={sample}, roi={roi}, slide={slide_num}, run={run_name}...")
        
        # --- Build paths ---
        img_path = base_img_dir / f"{sample}.ome.tif"
        alignment_file = base_img_dir / f"{sample}_alignment_files" / "matrix.csv"

        save_dir = base_out_dir / f"ROI{roi}"

            # --- Skip existing ---
        if save_dir.exists():
            existing = list(save_dir.glob("*.h5ad")) + list(save_dir.glob("*.png"))
            if existing:
                print(f"⚠️ {sample} ({roi}) already processed at {save_dir}, skipping.")
                continue

        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Run reader ---
        reader = XeniumReader().read(
            img_path=str(img_path),
            experiment_path=str(exp_stem / "experiment.xenium"),
            alignment_file_path=str(alignment_file),
            feature_matrix_path=str(exp_stem / "cell_feature_matrix.h5"),
            transcripts_path=str(exp_stem / "transcripts.parquet"),
            cells_path=str(exp_stem / "cells.parquet"),
            nucleus_bound_path=str(exp_stem / "nucleus_boundaries.parquet"),
            cell_bound_path=str(exp_stem / "cell_boundaries.parquet"),
            dapi_path=str(exp_stem / "morphology_focus/morphology_focus_0000.ome.tif"),
            use_dask=True,
        )

        # filter st.adata to WSI extent + drop Codeword features
        update_st_with_filtered_and_labelled(reader, drop_codeword=True)

        # --- Save outputs ---
        reader.save_spatial_plot(save_dir)
        reader.save(save_dir, pyramidal=True)
        print(f"✔ Saved results to {save_dir}")


if __name__ == "__main__":
    main()
