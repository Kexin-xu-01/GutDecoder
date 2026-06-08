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
import dask
import math
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

from hest.utils import load_wsi, read_xenium_alignment
import dask.dataframe as dd

# HEST / Gutdecoder imports
import hest
from hest import HESTData, XeniumReader
from hest.HESTData import XeniumHESTData
from hest.readers import pool_transcripts_xenium
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

def __transcript_bbox(self, transcript_df, key_x: str = "he_x", key_y: str = "he_y"):
    """
    Return transcript bbox as (ymin, ymax, xmin, xmax).
    Works for both pandas and dask dataframes.
    """
    if isinstance(transcript_df, dd.DataFrame):
        xmin, xmax, ymin, ymax = dask.compute(
            transcript_df[key_x].min(),
            transcript_df[key_x].max(),
            transcript_df[key_y].min(),
            transcript_df[key_y].max(),
        )
    else:
        xmin = float(transcript_df[key_x].min())
        xmax = float(transcript_df[key_x].max())
        ymin = float(transcript_df[key_y].min())
        ymax = float(transcript_df[key_y].max())

    return float(ymin), float(ymax), float(xmin), float(xmax)

def get_transcript_bbox(
    self,
    img_path: str,
    experiment_path: str,
    alignment_file_path: str = None,
    feature_matrix_path: str = None, 
    transcripts_path: str = None,
    cells_path: str = None,
    nucleus_bound_path: str = None,
    cell_bound_path: str = None,
    dapi_path = None,
    load_img=True,
    use_dask=True,
    spot_size_um=100.,
    nb_partitions=30):
    """ Read a Xenium sample

    Args:
        img_path (str): path to the WSI
        experiment_path (str): path to a `experiment.xenium` file
        alignment_file_path (str, optional): path to a DAPI->H&E matrix/keypoints alignment file, None if the H&E is already aligned with the DAPI. Defaults to None.
        feature_matrix_path (str, optional): path to a `cell_feature_matrix.h5`. Defaults to None.
        transcripts_path (str, optional): path to a transcripts.parquet, None to not load the transcripts. Defaults to None.
        cells_path (str, optional): path to a `cells.parquet` file, None to not load the cells. Defaults to None.
        nucleus_bound_path (str, optional): path to a `nucleus_boundaries.parquet` file. Defaults to None.
        cell_bound_path (str, optional): path to a `cell_boundaries` file. Defaults to None.
        dapi_path (_type_, optional): path to a `morphology_focus_0000.ome.tif`/`morphology_focus.ome.tif` file. Defaults to None.
        load_img (bool, optional): whenever to load the WSI. Defaults to True.
        use_dask (bool, optional): whenever to load the transcript dataframe with DASK (recommended if the transcript dataframe does not fit into the RAM). Defaults to False.
        spot_size_um (float, optional): transcripts are pooled into squares of spot_size_um x spot_size_um mirometers and then stored in `HESTData.adata`
        nb_partitions (int, optional): number of dask partition to use if use_dask is True. Defaults to 30

    Returns:
        XeniumHESTData: Xenium sample
    """

    if load_img:
        print("Loading the WSI... (can be slow for large images)")
        img, pixel_size_embedded = load_wsi(img_path)
    else:
        img, pixel_size_embedded = wsi_factory(np.zeros((1, 1, 3)), mpp=1.0), None

    dict = {}
    dict['pixel_size_um_embedded'] = pixel_size_embedded

    with open(experiment_path) as f:
        dict_exp = json.load(f)
        pixel_size_morph = dict_exp['pixel_size']
    dict = {**dict, **dict_exp}

    shapes = []


    alignment_matrix = read_xenium_alignment(alignment_file_path) if alignment_file_path else None
    dict['pixel_size_um_estimated'] = self._XeniumReader__xenium_estimate_pixel_size(pixel_size_morph, alignment_matrix)
    if cell_bound_path is not None:
        shapes.append(LazyShapes(cell_bound_path, 'tenx_cell', 'dapi', 
                                 reader=XeniumParquetCellReader, 
                                 reader_kwargs={'pixel_size_morph': pixel_size_morph}))
        if alignment_matrix is not None:
            shapes.append(LazyShapes(cell_bound_path, 'tenx_cell', 'he', 
                                     reader=XeniumParquetCellReader, 
                                     reader_kwargs={
                                         'pixel_size_morph': pixel_size_morph, 
                                         'alignment_matrix': alignment_matrix}))

    if nucleus_bound_path is not None:
        shapes.append(LazyShapes(nucleus_bound_path, 'tenx_nucleus', 'dapi', 
                                 reader=XeniumParquetCellReader, 
                                 reader_kwargs={'pixel_size_morph': pixel_size_morph}))
        if alignment_matrix is not None:
            shapes.append(LazyShapes(nucleus_bound_path, 'tenx_nucleus', 'he', 
                                     reader=XeniumParquetCellReader, 
                                     reader_kwargs={
                                         'pixel_size_morph': pixel_size_morph, 
                                         'alignment_matrix': alignment_matrix}))


    if transcripts_path is not None:
        print('Loading transcripts...')
        transcript_df = self._XeniumReader__load_transcripts(transcripts_path, alignment_matrix, pixel_size_morph, 
                                                use_dask)
        
        transcript_bbox = self.__transcript_bbox(transcript_df)
        print(f"Transcript bbox (ymin, ymax, xmin, xmax): {transcript_bbox}")

    return transcript_bbox

XeniumReader.get_transcript_bbox = get_transcript_bbox
XeniumReader.__transcript_bbox = __transcript_bbox

def get_indices_chunk(partition, key_x, key_y, 
                      x_min, y_min, spot_size_um, pixel_size_he, n, spot_grid_columns):
    a = np.floor((partition[key_x] - x_min) / (spot_size_um / pixel_size_he)).astype(int)
    b = np.floor((partition[key_y] - y_min) / (spot_size_um / pixel_size_he)).astype(int)

    c = b * n + a
    cols = spot_grid_columns.get_indexer(partition['feature_name'])
    return pd.DataFrame({'c': c, 'cols': cols}, index=partition.index)

def pool_transcripts_xenium_bbox(
    df: Union[pd.DataFrame, dd.DataFrame], 
    bbox,
    pixel_size_he: float,
    spot_size_um=100.,
    key_x='he_x',
    key_y='he_y'
) -> sc.AnnData: # type: ignore
    """ Pool a xenium transcript dataframe by square spots of `spot_size_um` micrometers.

    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): xenium transcipts (dask) dataframe containing columns:

            - 'he_x' and 'he_y' indicating the pixel coordinates of each transcripts in the morphology image
            - 'feature_name' indicating the transcript name
        pixel_size_he (float): pixel size in um/px of 'he_x' and 'he_y'
        spot_size_um: pooling rectangle width in um
        key_x: column name of pixel x coordinate of each transcript in `df`
        key_y: column name of pixel y coordinate of each transcript in `df`
        

    Returns:
        sc.AnnData: AnnData object, each row in .obs represents a bin, each row in `.X` represents the sum of transcripts within that bin. Center coordinates of each bin (in pixel on WSI) are in adata.obsm['spatial']
    """
    import scanpy as sc
    import dask.dataframe as dd
    import dask

    y_max = bbox[1]
    y_min = bbox[0]
    x_max = bbox[3]
    x_min = bbox[2]
    
    m = ((y_max - y_min) / (spot_size_um / pixel_size_he))
    n = ((x_max - x_min) / (spot_size_um / pixel_size_he))

    unique_features = df['feature_name'].unique()
    
    if isinstance(df, dd.DataFrame):
        m, n, unique_features, x_min, y_min = dask.compute(
            m, n, unique_features, x_min, y_min)

    m = math.ceil(m)
    n = math.ceil(n)
    spot_grid = pd.DataFrame(0, index=range(m * n), columns=unique_features)
    spot_grid_np = spot_grid.values.astype(np.uint32)
    
    if isinstance(df, dd.DataFrame):
        import dask.array as da
            
        cols_c = df.map_partitions(get_indices_chunk, key_x, key_y, 
                      x_min, y_min, spot_size_um, pixel_size_he, n, spot_grid.columns,
                      meta={'c': 'int64', 'cols': 'int64'})
        
        num_rows = m * n
        num_cols = len(unique_features)
        c_da = cols_c['c'].to_dask_array(lengths=True)
        cols_da = cols_c['cols'].to_dask_array(lengths=True)
        h, xedges, yedges = da.histogram2d(
            c_da, 
            cols_da, 
            bins=[np.arange(num_rows + 1), np.arange(num_cols + 1)]
        )
        spot_grid_np = h.astype(np.uint32).compute()
        
    else:
        cols_c = get_indices_chunk(df,key_x, key_y, 
                      x_min, y_min, spot_size_um, pixel_size_he, n, spot_grid.columns)

        c = cols_c['c']
        cols = cols_c['cols']
        np.add.at(spot_grid_np, (c, cols), 1)
    
    
    if isinstance(spot_grid.columns.values[0], bytes):
        spot_grid.columns = [i.decode('utf-8') for i in spot_grid.columns]
    

    expression_df = pd.DataFrame(spot_grid_np, columns=spot_grid.columns)
    
    coord_df = expression_df.copy()

    coord_df['x'] = x_min + (coord_df.index % n) * (spot_size_um / pixel_size_he) + ((spot_size_um / 2) / pixel_size_he)
    coord_df['y'] = y_min + np.floor(coord_df.index / n) * (spot_size_um / pixel_size_he) + ((spot_size_um / 2) / pixel_size_he)
    coord_df = coord_df[['x', 'y']]
    
    expression_df.index = [str(i) for i in expression_df.index]
    
    adata = sc.AnnData(expression_df)
    adata.var_names = adata.var_names.astype(str)
    adata.obsm['spatial'] = coord_df[['x', 'y']].values
    adata.obs['in_tissue'] = True
    adata.obs['pxl_col_in_fullres'] = coord_df['x'].values
    adata.obs['pxl_row_in_fullres'] = coord_df['y'].values
    adata.obs['array_col'] = np.arange(len(adata.obs)) % n
    adata.obs['array_row'] = np.arange(len(adata.obs)) // n
    adata.obs.index = [str(row).zfill(3) + 'x' + str(col).zfill(3) for row, col in  zip(adata.obs['array_row'], adata.obs['array_col'])]
    sc.pp.filter_cells(adata, min_counts=1)
    
    return adata


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
        "--pool-by-transcript-coord",
        action="store_true",
        help="If set, pool cell/niche data on the transcript coordinate grid using the transcript bbox.",
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
        transcript_path = exp_stem / "transcripts.parquet"

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

        st = HESTData.from_paths(
            adata_path=str(hest_dir / "aligned_adata.h5ad"),
            img=str(hest_dir / "aligned_fullres_HE.tif"),
            metrics_path=str(hest_dir / "metrics.json"),
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


        if args.pool_by_transcript_coord:
            # Use transcript bbox to force the same grid as transcript-level pooling
            bbox = XeniumReader().get_transcript_bbox(
                img_path=str(img_path),
                experiment_path=exp_stem / "experiment.xenium",
                alignment_file_path=alignment_file,
                transcripts_path=transcript_path,
            )

            adata_cells_pooled = pool_transcripts_xenium_bbox(
                cell_df,
                bbox=bbox,
                pixel_size_he=st.meta["pixel_size_um_estimated"],
                key_x="he_x",
                key_y="he_y",
                spot_size_um=st.meta["spot_diameter"],
            )
        else:

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
