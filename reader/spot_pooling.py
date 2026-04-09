#!/usr/bin/env python3
"""
Unified Xenium/HEST processing script.

This combines the repeated logic from the per-case scripts into one configurable
entrypoint.

Key configurable pieces:
- which spreadsheet column contains the h5ad path
- which obs column should be used as the pooled feature_name
- how the output folder is named
- which processed HEST root to read from
- which output root to write to

Example usages:

  # cell-level, source annotation is in CellAnnotation.Level0, pooled as cell_type
  python combine_xenium_scripts.py SAMPLE1 \
    --h5ad-col h5ad \
    --annotation-source-col CellAnnotation.Level0 \
    --feature-col cell_type \
    --feature-tag cell_l0 \
    --output-root /project/gutdecoder/kxu/hest/xenium_data \
    --processed-root /project/gutdecoder/kxu/hest/xenium_data

  # niche-level, the h5ad already has a niche column
  python combine_xenium_scripts.py SAMPLE1 \
    --h5ad-col h5ad_niche_kmeans_c12_n80 \
    --annotation-source-col niche \
    --feature-col niche \
    --feature-tag niche \
    --output-root /project/gutdecoder/kxu/hest/xenium_data \
    --processed-root /project/gutdecoder/kxu/hest/xenium_data
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
import re
from typing import Any
import yaml

import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

# HEST / Gutdecoder imports
import hest
from hest.HESTData import *  # noqa: F401,F403
from hest.readers import pool_transcripts_xenium
from hest.utils import *  # noqa: F401,F403
from gutdecoder.reader.h5ad_reader import *  # noqa: F401,F403

# Optional dependencies
try:
    import cucim  # noqa: F401
    logger.info("cuCIM available — GPU image ops enabled")
except ImportError:
    logger.info("cuCIM not available — falling back to CPU ops")


EXCEL_PATH_DEFAULT = "/project/simmons_hts/kxu/hest/xenium_directory.xlsx"

# Always keep these structural columns normalized.
BASE_RENAME_MAP = {
    "cell_centroid_x": "x_centroid",
    "cell_centroid_y": "y_centroid",
    "cell": "cell_id",
    "n_transcripts": "transcript_counts",
}


def _parse_rules(rule_col: Any) -> list[dict]:
    """Normalize crop/exclusion rules stored in the spreadsheet cell."""
    if rule_col is None or (isinstance(rule_col, float) and pd.isna(rule_col)):
        return []
    if isinstance(rule_col, dict):
        return [rule_col]
    if isinstance(rule_col, list):
        return rule_col
    if isinstance(rule_col, str):
        s = rule_col.strip()
        if not s:
            return []

        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            pass

        dict_blocks = re.findall(r"\{.*?\}", s, flags=re.S)
        if not dict_blocks:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else [parsed]

        rules: list[dict] = []
        for blk in dict_blocks:
            blk_clean = re.sub(r",\s*$", "", blk.strip(), flags=re.S)
            try:
                obj = json.loads(blk_clean)
            except Exception:
                obj = ast.literal_eval(blk_clean)

            if isinstance(obj, dict):
                rules.append(obj)
            elif isinstance(obj, list):
                rules.extend(obj)
            else:
                rules.append(obj)
        return rules

    raise ValueError(f"Unsupported crop/exclusion rule format: {type(rule_col)}")


def _build_rename_map(annotation_source_col: str, feature_col: str) -> dict[str, str]:
    """Build a rename map that standardizes the requested annotation column."""
    rename_map = dict(BASE_RENAME_MAP)
    rename_map[annotation_source_col] = feature_col
    return rename_map


def _resolve_path(base: Path, maybe_path: Any) -> Path:
    p = Path(str(maybe_path))
    return p if p.is_absolute() else base / p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Xenium imaging data for a given SampleID with configurable feature/output settings",
    )
    parser.add_argument("SampleID", type=str, help="SampleID to process")
    parser.add_argument(
        "--excel-path",
        default=EXCEL_PATH_DEFAULT,
        help="Metadata Excel file path",
    )
    parser.add_argument(
        "--h5ad-col",
        default="h5ad",
        help="Spreadsheet column name containing the h5ad path",
    )
    parser.add_argument(
        "--annotation-source-col",
        default="CellAnnotation.Level0",
        help="Column in adata.obs to rename into feature-col before pooling",
    )
    parser.add_argument(
        "--feature-col",
        default="cell_type",
        help="Standardized obs column to use as the pooled feature (e.g. cell_type, niche)",
    )
    parser.add_argument(
        "--feature-tag",
        default=None,
        help="Output tag used in folder naming. Defaults to feature-col.",
    )
    parser.add_argument(
        "--output-root",
        default="/project/simmons_hts/kxu/hest/xenium_data",
        help="Root folder for processed outputs",
    )
    parser.add_argument(
        "--processed-root",
        default="/project/simmons_hts/kxu/hest/xenium_data",
        help="Root folder containing processed HEST inputs (aligned_adata.h5ad, aligned_fullres_HE.tif, metrics.json)",
    )
    parser.add_argument(
        "--input-root-col",
        default="directory",
        help="Spreadsheet column containing the experiment root for load_xenium_dataset (kept for compatibility)",
    )
    parser.add_argument(
        "--img-dir-col",
        default="Directory",
        help="Spreadsheet column containing the folder with the H&E image and alignment file",
    )
    parser.add_argument(
        "--image-col",
        default="PostHnE",
        help="Spreadsheet column containing the H&E filename",
    )
    parser.add_argument(
        "--alignment-col",
        default="alignment",
        help="Spreadsheet column containing the alignment filename",
    )
    parser.add_argument(
        "--slide-col",
        default="Slide",
        help="Spreadsheet column containing the slide identifier",
    )
    parser.add_argument(
        "--roi-col",
        default="Roi",
        help="Spreadsheet column containing the ROI identifier",
    )
    parser.add_argument(
        "--run-col",
        default="run",
        help="Spreadsheet column containing the run name",
    )
    parser.add_argument(
        "--crop-col",
        default="crop_100_um",
        help="Spreadsheet column containing cropping/exclusion rules",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ROIs whose output folder already appears complete",
    )
    parser.add_argument(
        "--output-pattern",
        default="Xenium{run_name}_{feature_tag}/slide{slide_num}",
        help=(
            "Folder pattern under output-root. Available placeholders: "
            "{run_name}, {feature_tag}, {slide_num}"
        ),
    )
    parser.add_argument(
        "--processed-pattern",
        default="Xenium{run_name}/slide{slide_num}/ROI{roi}",
        help=(
            "Folder pattern under processed-root for reading aligned HEST inputs. "
            "Available placeholders: {run_name}, {slide_num}, {roi}"
        ),
    )
    parser.add_argument(
    "--config",
    type=str,
    help="Path to YAML config file"
    )
    args = parser.parse_args()

    # --- Load YAML config if provided ---
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # overwrite args with config values if present
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    sample_id = args.SampleID
    feature_tag = args.feature_tag or args.feature_col

    metadata = pd.read_excel(args.excel_path)
    sample_rows = metadata.loc[metadata["Sample_ID"] == sample_id]
    if sample_rows.empty:
        print(f"❌ No entry found for SampleID: {sample_id}")
        return

    rename_map = _build_rename_map(args.annotation_source_col, args.feature_col)

    for _, row in sample_rows.iterrows():
        sample = row["Sample_ID"]
        roi = row.get(args.roi_col)
        if pd.notnull(roi):
            roi = int(float(roi))
        else:
            print(f"⚠️ Missing ROI for sample={sample}; skipping row")
            continue

        slide_str = str(row[args.slide_col])
        match = re.search(r"\d+", slide_str)
        slide_num = int(match.group()) if match else None
        run_name = str(row.get(args.run_col, "")).strip()

        base_img_dir = Path(row[args.img_dir_col])
        image_name = Path(row[args.image_col])
        alignment_name = row.get(args.alignment_col)
        if pd.isna(alignment_name) or alignment_name is None:
            print(f"⚠️ Missing alignment file for sample={sample}, roi={roi}; skipping")
            continue

        h5ad_value = row.get(args.h5ad_col)
        if pd.isna(h5ad_value) or h5ad_value is None:
            print(f"⚠️ Missing h5ad path in column '{args.h5ad_col}' for sample={sample}, roi={roi}; skipping")
            continue

        exp_stem = Path(row[args.input_root_col])
        h5ad_path = Path(h5ad_value)
        img_path = _resolve_path(base_img_dir, image_name)
        alignment_file = _resolve_path(base_img_dir, alignment_name)

        processed_rel = args.processed_pattern.format(run_name=run_name, slide_num=slide_num, roi=roi)
        hest_dir = Path(args.processed_root) / processed_rel

        output_rel = args.output_pattern.format(
            run_name=run_name,
            feature_tag=feature_tag,
            slide_num=slide_num,
        )
        base_out_dir = Path(args.output_root) / output_rel
        save_dir = base_out_dir / f"ROI{roi}"

        print(f"Processing sample={sample}, roi={roi}, slide={slide_num}, run={run_name}...")

        if args.skip_existing and save_dir.exists():
            required = [
                save_dir / "aligned_adata.h5ad",
                save_dir / "aligned_fullres_HE.tif",
                save_dir / "metrics.json",
            ]
            if all(p.exists() for p in required):
                print(f"⚠️ {sample} ({roi}) already processed at {save_dir}, skipping.")
                continue

        save_dir.mkdir(parents=True, exist_ok=True)

        st = read_HESTData(
            str(hest_dir / "aligned_adata.h5ad"),
            str(hest_dir / "aligned_fullres_HE.tif"),
            str(hest_dir / "metrics.json"),
        )

        adata_labelled = sc.read_h5ad(h5ad_path)

        # Ensure spatial coordinates are present before update/align.
        if {"cell_centroid_x", "cell_centroid_y"}.issubset(adata_labelled.obs.columns):
            adata_labelled.obsm["spatial"] = adata_labelled.obs[["cell_centroid_x", "cell_centroid_y"]].to_numpy()
        elif {"x", "y"}.issubset(adata_labelled.obs.columns):
            adata_labelled.obs["cell_centroid_x"] = adata_labelled.obs["x"]
            adata_labelled.obs["cell_centroid_y"] = adata_labelled.obs["y"]
            adata_labelled.obsm["spatial"] = adata_labelled.obs[["cell_centroid_x", "cell_centroid_y"]].to_numpy()
        else:
            raise KeyError(
                "Could not find centroid columns. Expected either 'cell_centroid_x/y' or 'x/y' in adata.obs."
            )

        update_st_with_filtered_and_labelled(st, adata_labelled, drop_codeword=True)

        px_um = st.meta["pixel_size"]
        adata_labelled = align_labelled_to_he(adata_labelled, alignment_file, pixel_size_um=px_um)
        adata_labelled = standardize_obs_columns(adata_labelled, rename_map=rename_map)

        if args.feature_col not in adata_labelled.obs.columns:
            raise KeyError(
                f"'{args.feature_col}' not found in adata_labelled.obs after standardization. "
                f"Available columns include: {list(adata_labelled.obs.columns[:25])} ..."
            )

        cell_df = adata_labelled.obs[[args.feature_col, "he_x", "he_y"]].rename(
            columns={args.feature_col: "feature_name"}
        )

        adata_cells_pooled = pool_transcripts_xenium(
            cell_df,
            st.meta["pixel_size_um_estimated"],
            key_x="he_x",
            key_y="he_y",
            spot_size_um=st.meta["spot_diameter"],
        )

        adata_cells_pooled.obs["total_counts"] = np.asarray(adata_cells_pooled.X.sum(axis=1)).ravel()
        adata_cells_pooled.obs["log1p_total_counts"] = np.log1p(adata_cells_pooled.obs["total_counts"])

        register_downscale_img(adata_cells_pooled, st.wsi, st.meta["pixel_size_um_estimated"])

        xy = adata_cells_pooled.obsm["spatial"]
        sx, sy = xy[:, 0], xy[:, 1]
        xmin, xmax, ymin, ymax = 0, st.wsi.width, 0, st.wsi.height

        rule_col = row.get(args.crop_col)
        rules = _parse_rules(rule_col)
        print("Applying exclusions:")
        for r in rules:
            print(" ", r)

        final_keep = apply_spot_exclusions(sx, sy, (xmin, xmax, ymin, ymax), rules)
        print(f"Spots kept after rules: {final_keep.sum()} / {len(final_keep)}")

        st.adata = adata_cells_pooled[final_keep]
        stats = refresh_meta_counts(st)
        print("Before/After:", stats)

        # save_all differs a little between script variants; keep this flexible.
        try:
            overlay_path = save_all(st, save_dir, pyramidal=True, cell_label_key=args.feature_col)
        except TypeError:
            overlay_path = save_all(st, save_dir, pyramidal=True)

        print("✔ Saved results to", overlay_path)


if __name__ == "__main__":
    main()
