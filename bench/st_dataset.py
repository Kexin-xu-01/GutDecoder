from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import Dataset
import h5py
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from PIL import Image

def log_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    #print(adata.obs)
    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            neighbors_index = adata.obs[((adata.obs['array_row'] >= row - 1) & (adata.obs['array_row'] <= row + 1)) & \
                ((adata.obs['array_col'] >= col - 1) & (adata.obs['array_col'] <= col + 1))].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg
    
    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)

    return filtered_adata

def library_size_norm_adata(adata: sc.AnnData) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    
    # library size normalization
    sc.pp.normalize_total(filtered_adata, target_sum=1e6)

    return filtered_adata

def load_adata(expr_path, genes = None, barcodes = None, normalize=False, library_size_normalize=False):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]
    if library_size_normalize:
        adata = library_size_norm_adata(adata)
    if normalize:
        adata = log_adata(adata)
    return adata.to_df()
