from pathlib import Path
import dask
import hest
print(hest.__version__)
from hest.HESTData import read_HESTData
from hest import iter_hest
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import math

import scanpy as sc

from hest.patching_wrapper import *

for i in range(6, 7):
    base = Path(f"/project/simmons_hts/kxu/hest/visium_data/VisiumR{i}")
    
    # check both possible subfolders
    for slide in ["slide1", "slide2"]:
        slide_path = base / slide
        if slide_path.exists():
            print(f"Processing {slide_path}...")
            patch_hest_samples(
                broad_root=slide_path,
                ids=None,
                target_patch_size=224,
                target_pixel_size=0.5,
                threshold=0.15,
            )
        else:
            print(f"Skipping {slide_path} (not found)")
