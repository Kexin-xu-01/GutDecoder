#!/usr/bin/env python3
"""
Script to process Visium  data using Hest and CuCIM,
with metadata loaded from an Excel file and SampleID provided as a CLI argument.
"""

from pathlib import Path
import pandas as pd
import hest
from hest import VisiumReader
import argparse
import re

# Optional: Dask for parallel processing
import dask

from hest.HESTData import *


# Path to metadata Excel file
EXCEL_PATH = "/project/simmons_hts/kxu/hest/visium_directory.xlsx"


def main():
    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="Process Visium data for a given SampleID")
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
        #directory = row['Directory']
        sample_name = row['Sample_name']
        directory = Path("/project/simmons_hts/kxu/deepspace_CD/test_data") / sample_name / Path('SpaceRanger/')
        roi = row["Roi"]
        if pd.notnull(roi):
            roi = int(float(roi))

        # Choose slide directory (slide1, slide2, etc.)
        slide_num = str(int(row['Slide']))
        run_name = str(row.get("run", "")).strip()
        save_dir = Path("/project/simmons_hts/kxu/hest/visium_data") / f"Visium{run_name}" / f"slide{slide_num}" / f"ROI{roi}"

        print(f"Processing sample={sample}, roi={roi}, slide={slide_num}, run={run_name}...")
        
            # --- Skip existing ---
        if save_dir.exists():
            existing = list(save_dir.glob("*.h5ad")) + list(save_dir.glob("*.png"))
            if existing:
                print(f"⚠️ {sample} ({roi}) already processed at {save_dir}, skipping.")
                continue

        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Run reader ---
        print('loading data from directory', directory)
        reader = VisiumReader().auto_read(str(directory))

        # --- Save outputs ---
        reader.save_spatial_plot(save_dir)
        reader.save(save_dir, pyramidal=True)
        print(f"✔ Saved results to {save_dir}")


if __name__ == "__main__":
    main()
