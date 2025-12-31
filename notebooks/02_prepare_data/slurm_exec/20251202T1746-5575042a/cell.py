import dask
import hest
print(hest.__version__)
from hest.HESTData import read_HESTData
from pathlib import Path
import torch

import warnings
import numpy as np
import tifffile as tiff

from hestcore.wsi import NumpyWSI

# cutomised function
from hest.segmentation.segment_wrapper import *

base_dir = Path("/project/simmons_hts/kxu/hest/visium_data")

for i in range(1, 7):  # VisiumR1–R6
    xenium_dir = base_dir / f"VisiumR{i}"
    if not xenium_dir.exists():
        continue

    for slide_dir in sorted(xenium_dir.glob("slide*")):
        if slide_dir.is_dir():
            print(f"🚀 Running segmentation for: {slide_dir}")
            segment_hest_tissue(str(slide_dir))
