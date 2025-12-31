#!/usr/bin/env python3

import spatialdata as sd
import pandas as pd
import numpy as np
from pathlib import Path

from hest.readers import pool_transcripts_xenium
from hest.HESTData import XeniumHESTData
from hest.utils import register_downscale_img
from hestcore.wsi import wsi_factory

from spatialdata.transformations import get_transformation


def read_xenium_zarr(zarr_path: str, spot_size_um: float = 100.0) -> XeniumHESTData:
    """
    Read a Xenium Zarr sample and create a HESTData object.
    """
    # --- Load the Zarr ---
    sdata = sd.read_zarr(zarr_path)

    # --- Load transcript table ---
    transcripts = sdata.points['transcripts']
    if hasattr(transcripts, "compute"):
        transcripts = transcripts.compute()
    transcripts = pd.DataFrame(transcripts)

    # --- Pool transcripts into pseudo-visium spots ---
    scale_um_to_px = get_transformation(
        sdata["transcripts"], to_coordinate_system="scale_um_to_px"
    ).scale[0]
    pixel_size = 1 / scale_um_to_px
    print(f"[INFO] Pixel size (um): {pixel_size}")

    transcripts["he_x"] = transcripts["x"] / pixel_size
    transcripts["he_y"] = transcripts["y"] / pixel_size

    adata = pool_transcripts_xenium(
        transcripts,
        pixel_size_he=pixel_size,
        spot_size_um=spot_size_um,
        key_x="he_x",
        key_y="he_y",
    )

    # --- Load HE_registered image and downscale ---
    img = np.array(sdata.images["HE_registered"]).transpose(1, 2, 0)  # (H, W, C)
    img, pixel_size_embedded = wsi_factory(img), None
    register_downscale_img(adata, img, pixel_size)

    # metadata
    meta = {
        "pixel_size_um_embedded": pixel_size_embedded,
        "pixel_size_um_estimated": pixel_size,
        "spot_diameter": spot_size_um,
        "inter_spot_dist": spot_size_um,
        "spots_under_tissue": len(adata.obs),
        'tissue_downscaled_fullres_scalef': adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef'],
        'spot_diameter_fullres': adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']

    }

    # --- Optional: keep DAPI path if available ---
    dapi_path = sdata.images["DAPI"] if "DAPI" in sdata.images else None

    # --- Cell-level AnnData if available ---
    cell_adata = None
    if "anucleus" in sdata.tables:
        cell_adata = sdata.tables["anucleus"]
        meta["cells_under_tissue"] = len(cell_adata.obs)

    # --- Build HESTData object ---
    st_object = XeniumHESTData(
        adata=adata,
        img=img,
        pixel_size=pixel_size,
        meta=meta,
        transcript_df=transcripts,
        cell_adata=cell_adata,
        dapi_path=dapi_path,
    )

    return st_object


def process_folder(input_dir: str, output_dir: str, spot_size_um: float = 100.0):
    """
    Loop through all Zarr samples in a folder and save processed HESTData.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for zarr_file in sorted(input_path.glob("*.zarr")):
        print(f"[INFO] Processing {zarr_file.name}...")
        try:
            st = read_xenium_zarr(str(zarr_file), spot_size_um=spot_size_um)
            save_dir = output_path / zarr_file.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            st.save_spatial_plot(save_dir)
            st.save(save_dir, pyramidal=True)
            print(f"[INFO] Saved results to {save_dir}")
        except Exception as e:
            print(f"[ERROR] Failed on {zarr_file.name}: {e}")


if __name__ == "__main__":
    # Example usage
    input_dir = "/project/simmons_hts/kxu/hest/data/broad/zarr"
    output_dir = "/project/simmons_hts/kxu/hest/xenium_data/broad"
    process_folder(input_dir, output_dir, spot_size_um=100.0)
