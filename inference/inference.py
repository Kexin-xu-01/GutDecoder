# Import libraries
# Core Python Libraries
import os
import json
import warnings
import gc
import joblib
from types import SimpleNamespace
from operator import itemgetter
from abc import abstractmethod
from tqdm import tqdm
import glob
from pathlib import Path

# Core Data manipulation Libraries
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning
import math

# Visualization Library
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# Spatial Data Processing
import h5py  
import scanpy as sc

# Frameworks for ML and DL models
import torch
import timm  
from sklearn.linear_model import Ridge  
from sklearn.decomposition import PCA

from hest.bench.utils.file_utils import read_assets_from_h5


def load_models_from_directories(base_path):
    """
    Load all trained regression models (one model for each cross-validation split)
    Load 'model.pkl' from each directory within the base_path.

    :param base_path: The parent directory containing split subdirectories.
    :return: A dictionary where keys are directory names and values are the loaded models.
    """

    models = {}
    for name in os.listdir(base_path):
        dir_path = os.path.join(base_path, name)
        if os.path.isdir(dir_path):
            model_path = os.path.join(dir_path, 'model.pkl')
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"'model.pkl' not found in {dir_path}")

    return models


def load_gene_list(dir_models_and_results):
    """
    Load the gene list from a specified JSON file for a given dataset.

    Args:
        dataset_name (str): Dataset name (folder under eval_data_dir).
        gene_list_name (str): Name of the JSON file inside the dataset folder (e.g. "full_panel_genes.json").
        eval_data_dir (str or Path): Root directory containing dataset folders.

    Returns:
        gene_list: list of genes.
    """
    gene_json_path = Path(os.path.join(dir_models_and_results, "split0/summary.json"))
    print('loading gene list from ', str(gene_json_path))

    if not gene_json_path.exists():
        raise FileNotFoundError(f"summary.json not found at {gene_json_path}")
    
    with open(gene_json_path, "r") as f:
        data = json.load(f)   

    gene_names = [entry["name"] for entry in data["pearson_corrs"]]

    print(f"Loaded '{len(gene_names)}' genes")
    return gene_names

def predict_and_aggregate_models(X_test, results_dir):
    """
    Load models from the given directory, make predictions on the test set (X_test),
    aggregate the predictions by averaging, and set negative predictions to 0.

    Args:
    - X_test (np.array): The test data to make predictions on.
    - results_dir (str): Directory containing the saved models.

    Returns:
    - np.array: The aggregated predictions.
    """

    # Load models from the specified directory
    models = load_models_from_directories(results_dir)

    predictions = []
    model_names = []

    for split_name in models.keys():
        model_names.append(split_name)
        if "pca_pipeline" in models[split_name]:
            print('perform Scaling + PCA')
            X_test_transformed = models[split_name]['pca_pipeline'].transform(X_test)
        else:
            X_test_transformed = X_test

        preds = models[split_name]['regression_model'].predict(X_test_transformed)
        predictions.append(preds)

    # Stack the predictions 
    predictions = np.stack(predictions)
    print("predictions has the shape ", predictions.shape)

    # Aggregate predictions by calculating the mean across all models
    average_predictions = np.mean(predictions, axis=0)

    sd_predictions = np.std(predictions, axis=0)

    # Set any negative predictions to 0
    # average_predictions = np.where(average_predictions < 0, 0.0, average_predictions)

    del models

    return average_predictions, sd_predictions, predictions, model_names


def infer(
    test_embed_dir: str,  # either a directory containing sample_name.h5 files or a single .h5 file
    model_directory_path: str,  # Path with trained model and results (same as before)
    out_dir: str # path to save output prediction, e.g. /project/simmons_hts/kxu/hest/inference/uni_v2
):
    """
    Find all sample_name.h5 files under `test_embed_dir`, run inference for each sample,
    and save a CSV named '<sample_name>_predictions.csv' next to the .h5 file.
    Returns a list of saved file paths.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Resolve sample file list
    if os.path.isdir(test_embed_dir):
        sample_paths = sorted(glob.glob(os.path.join(test_embed_dir, "*.h5")))
    else:
        raise ValueError(f"test_embed_dir must be a directory. Got: {test_embed_dir!r}")

    if not sample_paths:
        raise FileNotFoundError(f"No .h5 files found in {test_embed_dir!r}")

    # Previous directory where models and results are stored 
    dir_models_and_results = os.path.join("/project/simmons_hts/kxu/hest/eval/ST_pred_results", model_directory_path)

    print("Using models from ", dir_models_and_results)

    # Load training configuration parameters once
    config_path = os.path.join(dir_models_and_results, "split0", "config.json")
    with open(config_path, "r") as f:
        args_dict = json.load(f)
    args = SimpleNamespace(**args_dict)

    # Directory for processed test dataset (temporary storage) 
    dir_processed_dataset_test = os.path.join("/tmp", "processed_dataset_test")
    os.makedirs(dir_processed_dataset_test, exist_ok=True)

    # Set device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_paths = []

    for sample_path in sample_paths:
        # Extract sample name (without extension) from the file path
        name_data = os.path.splitext(os.path.basename(sample_path))[0]
        print(f"\n-- {name_data} INFERENCE ---------------------------------------------------------------\n")
        print(sample_path)

        print(f"\n-- {name_data} FETCHING EMBEDDING--\n")
        assets, _ = read_assets_from_h5(sample_path)

        # Extract embeddings features for prediction
        X_test = assets['features']
        print("Embedding shape (X_test):", X_test.shape)

        print(f"\n-- {name_data} REGRESSION PREDICTIONS--\n")
        # Make predictions and aggregate results across cross-validation regression models
        average_predictions, sd_predictions, per_model_predictions, model_names = predict_and_aggregate_models(X_test, dir_models_and_results)

        # Convert the predictions to a DataFrame (rounded to 2 decimal places)
        gene_names = load_gene_list(dir_models_and_results)
        # prediction = pd.DataFrame(np.round(average_predictions, 2), index=coords, columns=gene_names)
        # prediction = prediction.reset_index(names="coords")

        coords = assets["coords"]  # numpy array shape (N,2)
        print("coords shape:", coords.shape)
        assert len(coords) == average_predictions.shape[0], "coords and predictions must have same number of rows"

        # Build prediction DataFrame 
        prediction = pd.DataFrame(
            np.round(average_predictions, 5),
            columns=gene_names
        )

        meta = pd.DataFrame({
            "x": coords[:, 0].astype(int),  # or .astype(float) if subpixel
            "y": coords[:, 1].astype(int)
        })

        meta["barcode"] = meta.apply(lambda r: f"{int(r['x'])}x{int(r['y'])}", axis=1)

        # Normalise coordinates
        patch_pixel = 224

        y_max = meta['y'].max()
        y_min = meta['y'].min()
        x_max = meta['x'].max()
        x_min = meta['x'].min()

        #n = ((x_max - x_min) / (spot_size_um / pixel_size_he))
        n = ((x_max - x_min) / patch_pixel)
        n = math.ceil(n)

        adata = sc.AnnData(prediction)
        adata.var_names = adata.var_names.astype(str)
        print('Some adata var_names are ', adata.var_names[0:6])
        adata.obsm['spatial'] = meta[["x","y"]].values
        adata.obs['in_tissue'] = True
        adata.obs['pxl_col_in_fullres'] = meta['x'].values
        adata.obs['pxl_row_in_fullres'] = meta['y'].values
        adata.obs['array_col'] = np.arange(len(adata.obs)) % n
        adata.obs['array_row'] = np.arange(len(adata.obs)) // n
        adata.obs.index = [str(row).zfill(3) + 'x' + str(col).zfill(3) for row, col in  zip(adata.obs['array_row'], adata.obs['array_col'])]

        # --- Add SD into layers ---
        # layers must be same shape as adata.X: (n_obs, n_vars)
        adata.layers['pred_sd'] = sd_predictions

        # Set any negative predictions to 0
        # average_predictions = np.where(average_predictions < 0, 0.0, average_predictions)
        # sc.pp.filter_cells(adata, min_counts=1) # this doesn't work with lots of negative prediction

        # filter out spots where all genes are predicted zero
        nonzero_mask = (np.abs(adata.X).sum(axis=1) > 0)
        adata = adata[nonzero_mask].copy()

        pred_sd = np.asarray(adata.layers["pred_sd"])

        adata.var["mean_expression"] = prediction.mean(axis=0)

        # ---- Mean SD per gene (across spots) ----
        adata.var["mean_pred_sd_per_gene"] = pred_sd.mean(axis=0)

        # ---- Mean SD per spot (across genes) ----
        adata.obs["mean_pred_sd_per_spot"] = pred_sd.mean(axis=1)

        #adata.obs.index = [str(row).zfill(7) + 'x' + str(col).zfill(7) for row, col in  zip(adata.obs['pxl_row_in_fullres'], adata.obs['pxl_col_in_fullres'])]

        out_filename = f"{name_data}.h5ad"
        out_path = os.path.join(out_dir, 'adata')
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, out_filename)
        adata.write(out_file)

        print(f"\n-- {name_data} PREDICTION ADATA SAVED: {out_path}\n")

        del average_predictions, X_test, assets, prediction, sd_predictions, per_model_predictions, model_names
        gc.collect()

        saved_paths.append(out_path)

    print(f"\n-- Finished processing {len(sample_paths)} sample(s).\n")
    return saved_paths