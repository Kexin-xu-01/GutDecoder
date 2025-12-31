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

# cutomised function
from hest.patching_wrapper import *

patch_hest_samples(
    broad_root= Path("/project/simmons_hts/kxu/hest/xenium_data/XeniumPR1_25um/slide1/"),
    ids=None,
    target_patch_size = 224,
    target_pixel_size = 0.5,
    threshold = 0.15
)

patch_hest_samples(
    broad_root= Path("/project/simmons_hts/kxu/hest/xenium_data/XeniumPR1_25um/slide2/"),
    ids=None,
    target_patch_size = 224,
    target_pixel_size = 0.5,
    threshold = 0.15
)
