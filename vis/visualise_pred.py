import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import re
import h5py
import math
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from scipy.stats import pearsonr
from hest.vis.plot import *


RUN_ROOT = "/project/simmons_hts/kxu/hest/eval/ST_pred_results"
DATA_ROOT = "/project/simmons_hts/kxu/hest/eval/data/"


def load_best_model_inference(run, root_dir=RUN_ROOT):
    """
    For a given run, find the best model (highest Pearson mean in dataset_results.json),
    then load inference_dump.pkl files from all splits under the best model folder.

    Args:
        run (str): Run folder name (e.g. 'run_25-09-05-16-12-17').
        root_dir (str): Root directory containing run folders.

    Returns:
        tuple: (best_model_info, dataset_name, inference_dumps, gene_list_name)
            best_model_info: dict with encoder_name, pearson_mean, pearson_std
            inference_dumps: dict mapping split names (e.g., 'split1') to loaded pickle objects
            gene_list_name: name of the gene list used for this run
    """
    import os, json, pickle

    run_path = os.path.join(root_dir, run)
    dataset_results_path = os.path.join(run_path, "dataset_results.json")

    if not os.path.isfile(dataset_results_path):
        raise FileNotFoundError(f"dataset_results.json not found for run {run}")

    # Step 1: Load dataset_results.json
    with open(dataset_results_path, "r") as f:
        dataset_results = json.load(f)

    dataset_entry = dataset_results.get("results", [])[0]
    dataset_name = dataset_entry.get("dataset_name", "Unknown")
    all_models = dataset_entry.get("results", [])

    if not all_models:
        raise ValueError(f"No model results found in {dataset_results_path}")

    # Step 2: Find best model
    best_model = max(all_models, key=lambda x: x["pearson_mean"])
    best_model_name = best_model["encoder_name"]

    print(f"Best model for dataset {dataset_name}:")
    print(f"encoder_name: {best_model_name}")
    print(f"pearson_mean: {best_model['pearson_mean']}")
    print(f"pearson_std: {best_model['pearson_std']}")

    best_model_info = {
        "encoder_name": best_model_name,
        "pearson_mean": best_model["pearson_mean"],
        "pearson_std": best_model["pearson_std"]
    }

    # Step 2a: Get gene_list_name from config.json
    gene_list_name = ""
    for dirpath, _, filenames in os.walk(run_path):
        if "config.json" in filenames:
            config_path = os.path.join(dirpath, "config.json")
            with open(config_path, 'r') as cf:
                config = json.load(cf)
                gene_list_name = config.get("gene_list", "")
                print(f"Genes predicted from list: {gene_list_name}")
            break

    # Step 3: Path to best model folder
    model_folder_path = os.path.join(run_path, dataset_name, best_model_name)
    if not os.path.isdir(model_folder_path):
        raise FileNotFoundError(f"Best model folder not found at {model_folder_path}")

    # Step 4: Load inference_dump.pkl from each split folder
    inference_dumps = {}
    for split_name in os.listdir(model_folder_path):
        split_path = os.path.join(model_folder_path, split_name)
        if os.path.isdir(split_path):
            inference_file = os.path.join(split_path, "inference_dump.pkl")
            if os.path.isfile(inference_file):
                with open(inference_file, "rb") as f:
                    inference_dumps[split_name] = pickle.load(f)
                print(f"Loaded inference_dump.pkl from {split_name}")
            else:
                print(f"No inference_dump.pkl found in {split_name}")

    return best_model_info, dataset_name, inference_dumps, gene_list_name

def load_gene_list(dataset_name, gene_list_name, eval_data_dir=DATA_ROOT):
    """
    Load the gene list from a specified JSON file for a given dataset.

    Args:
        dataset_name (str): Dataset name (folder under eval_data_dir).
        gene_list_name (str): Name of the JSON file inside the dataset folder (e.g. "full_panel_genes.json").
        eval_data_dir (str or Path): Root directory containing dataset folders.

    Returns:
        gene_list: list of genes.
    """
    dataset_json_path = Path(eval_data_dir) / dataset_name / gene_list_name

    if not dataset_json_path.exists():
        raise FileNotFoundError(f"{gene_list_name} not found at {dataset_json_path}")

    with open(dataset_json_path, "r") as f:
        data = json.load(f)['genes']

    print(f"Loaded {gene_list_name} for dataset '{dataset_name}'")
    return data

def format_inference_with_genes(inference_dumps, gene_list):
    """
    Attach gene names as column labels for predictions and targets.

    Args:
        inference_dumps (dict): Mapping split_name -> inference data (with 'preds_all' and 'targets_all').
        gene_list (list): List of gene names.

    Returns:
        dict: Mapping split_name -> dict with DataFrames:
              {'preds': DataFrame, 'targets': DataFrame}
    """
    formatted_inference = {}

    for split_name, dump in inference_dumps.items():
        preds = dump["preds_all"]
        targets = dump["targets_all"]

        if preds.shape[1] != len(gene_list):
            raise ValueError(
                f"Number of genes ({len(gene_list)}) does not match number of prediction columns ({preds.shape[1]})"
            )

        preds_df = pd.DataFrame(preds, columns=gene_list)
        targets_df = pd.DataFrame(targets, columns=gene_list)

        formatted_inference[split_name] = {
            "preds": preds_df,
            "targets": targets_df
        }

        print(f"{split_name}: preds shape {preds_df.shape}, targets shape {targets_df.shape}")

    return formatted_inference

def plot_split_distributions(formatted_inference, plot_type="box", colors=None, figsize=(10,6), max_points=20000, showfliers=False):
    """
    Plot side-by-side distributions (boxplot or violin) of preds vs targets for each split.

    Parameters
    ----------
    formatted : dict
        Dictionary mapping split_name -> {"preds": DataFrame, "targets": DataFrame}
    plot_type : str
        "box" or "violin"
    colors : dict, optional
        Mapping {"preds": color, "targets": color}. Defaults to matplotlib tab colors.
    figsize : tuple, optional
        Figure size. Default (10,6)
    max_points : int, optional
        Maximum points to sample per DataFrame (only used for violin). Default 20k
    showfliers : bool
        Whether to show outliers in boxplots. Default False
    """
    if colors is None:
        colors = {"preds": "tab:blue", "targets": "tab:orange"}
    
    fig, ax = plt.subplots(figsize=figsize)

    xticks = []
    xtick_positions = []

    if plot_type == "box":
        # Precompute stats for boxplots
        def compute_boxplot_stats(series):
            q1 = np.percentile(series, 25)
            median = np.percentile(series, 50)
            q3 = np.percentile(series, 75)
            iqr = q3 - q1
            lower_whisker = series[series >= (q1 - 1.5 * iqr)].min()
            upper_whisker = series[series <= (q3 + 1.5 * iqr)].max()
            return {
                "whislo": lower_whisker,
                "q1": q1,
                "med": median,
                "q3": q3,
                "whishi": upper_whisker,
                "fliers": []
            }
        
        boxplot_data = []
        positions = []
        colors_list = []

        for split_idx, (split_name, data) in enumerate(formatted_inference.items()):
            base_pos = split_idx * 3
            for i, (label, df) in enumerate(data.items()):
                series = df.values.flatten()
                stats = compute_boxplot_stats(series)
                pos = base_pos + i + 1
                boxplot_data.append(stats)
                positions.append(pos)
                colors_list.append(colors[label])
            xticks.append(split_name)
            xtick_positions.append(base_pos + 1.5)

        bp = ax.bxp(boxplot_data, positions=positions, patch_artist=True, showfliers=showfliers)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    elif plot_type == "violin":
        data_to_plot = []
        positions = []
        violin_colors = []

        for split_idx, (split_name, data) in enumerate(formatted_inference.items()):
            base_pos = split_idx * 3
            for i, (label, df) in enumerate(data.items()):
                series = df.values.flatten()
                if len(series) > max_points:
                    series = np.random.choice(series, max_points, replace=False)
                data_to_plot.append(series)
                pos = base_pos + i + 1
                positions.append(pos)
                violin_colors.append(colors[label])
            xticks.append(split_name)
            xtick_positions.append(base_pos + 1.5)

        vp = ax.violinplot(data_to_plot, positions=positions, showmeans=False, showmedians=True)
        for body, color in zip(vp['bodies'], violin_colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        vp['cmedians'].set_color('black')

    else:
        raise ValueError("plot_type must be 'box' or 'violin'")

    # Common styling
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticks, rotation=45, ha='right')
    ax.set_ylabel("Value")
    ax.set_title(f"Predictions vs Targets by Split ({plot_type.capitalize()})")

    handles = [
        plt.Rectangle((0,0),1,1,facecolor=colors["preds"], alpha=0.7),
        plt.Rectangle((0,0),1,1,facecolor=colors["targets"], alpha=0.7)
    ]
    ax.legend(handles, ["preds", "targets"], title="Type")

    plt.tight_layout()
    plt.show()

def _clean_barcode_raw(x):
    """Turn raw stored barcode entry into a clean string."""
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode()
        except Exception:
            return str(x)
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            flat = np.array(x).flatten()
            if flat.size > 0:
                return _clean_barcode_raw(flat[0])
        except Exception:
            pass
        return str(x)
    s = str(x).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    m = re.match(r"^b['\"](.+)['\"]$", s)
    if m:
        return m.group(1)
    m2 = re.match(r"^['\"](.+)['\"]$", s)
    if m2:
        return m2.group(1)
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    if s.startswith('b"') and s.endswith('"'):
        return s[2:-1]
    return s.strip().strip("'").strip('"')


def read_patch_meta_from_h5(h5_path, barcode_ds='barcode', coords_ds='coords', verbose=True):
    """Read barcodes + coords from one patches.h5 file and return a cleaned DataFrame."""
    with h5py.File(h5_path, "r") as f:
        raw_barcodes = f[barcode_ds][:]
        coords = f[coords_ds][:]
    if coords.shape[0] != len(raw_barcodes):
        raise ValueError(f"coords and barcode length mismatch in {h5_path}")
    cleaned = [_clean_barcode_raw(x) for x in raw_barcodes]
    cleaned = [re.sub(r"^b['\"](.+)['\"]$", r"\1", s) for s in cleaned]
    df = pd.DataFrame({
        "barcode": cleaned,
        "coord_x": coords[:, 0].astype(int),
        "coord_y": coords[:, 1].astype(int),
    })
    if verbose:
        print(f"[read] {os.path.basename(h5_path)}: {len(df)} barcodes")
    return df

import os
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def _count_patches_in_tile_h5(tile_h5_path: str, barcode_ds: str = 'barcode') -> int:
    """Return number of barcode entries (patches) in a tile h5 file."""
    if not os.path.isfile(tile_h5_path):
        raise FileNotFoundError(tile_h5_path)
    with h5py.File(tile_h5_path, 'r') as f:
        if barcode_ds not in f:
            raise KeyError(f"Dataset '{barcode_ds}' not found in {tile_h5_path}. Keys: {list(f.keys())}")
        return int(len(f[barcode_ds]))

def expand_split_keys_to_samples(
    formatted_inference: Dict[str, Dict],
    df_test_splits: pd.DataFrame,
    dataset_name: str,
    base_dir: str = "/project/simmons_hts/kxu/hest/eval/data",
    patches_subdir: str = "patches",
    barcode_ds: str = 'barcode',
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Replace keys in formatted_inference that look like 'splitN' with per-sample keys.
    For each splitN:
      - get list of test samples from df_test_splits[df_test_splits['split']==N]['test_sample']
      - read patch h5 for each sample to determine #patches
      - split formatted_inference['splitN']['preds'] and ['targets'] row-wise in the same order
        using those counts and assign new keys formatted_inference[sample_name] = {...}
    Returns a new dict (does not mutate input in-place).
    """
    out = {}
    # make quick mapping from split -> list of samples (preserve df order)
    split_to_samples = {}
    for s, group in df_test_splits.groupby('split'):
        split_to_samples[int(s)] = list(group['test_sample'].astype(str).tolist())

    patch_dir = os.path.join(base_dir, dataset_name, patches_subdir)
    if verbose:
        print(f"[info] using patch_dir = {patch_dir}")

    for key, entry in formatted_inference.items():
        # keep non-split keys intact
        if not (isinstance(key, str) and key.lower().startswith('split')):
            out[key] = entry
            continue

        # parse split number
        try:
            split_num = int(key.replace('split', '').strip())
        except Exception:
            if verbose:
                print(f"[warn] cannot parse split number from key '{key}', keeping as-is")
            out[key] = entry
            continue

        samples = split_to_samples.get(split_num, [])
        if len(samples) <= 1:
            # no splitting necessary; rename to sample if exactly 1
            if len(samples) == 1:
                new_key = samples[0]
                if verbose:
                    print(f"[info] renaming {key} -> {new_key}")
                out[new_key] = entry
            else:
                # no mapping found for split -> keep key as-is
                if verbose:
                    print(f"[warn] no test samples found for split {split_num}; keeping key '{key}' unchanged")
                out[key] = entry
            continue

        # multi-sample split: fetch per-sample patch counts
        sample_patch_counts = []
        missing = []
        for sname in samples:
            tile_path = os.path.join(patch_dir, f"{sname}.h5")
            if not os.path.isfile(tile_path):
                missing.append(sname)
                sample_patch_counts.append(None)
            else:
                try:
                    cnt = _count_patches_in_tile_h5(tile_path, barcode_ds=barcode_ds)
                    sample_patch_counts.append(int(cnt))
                except Exception as e:
                    if verbose:
                        print(f"[warn] failed to read {tile_path}: {e}")
                    sample_patch_counts.append(None)
                    missing.append(sname)

        preds_df = entry.get('preds')
        targets_df = entry.get('targets')
        if preds_df is None or targets_df is None:
            if verbose:
                print(f"[warn] split {split_num} missing preds/targets; copying as-is under key '{key}'")
            out[key] = entry
            continue

        total_rows = preds_df.shape[0]
        if verbose:
            print(f"[info] splitting {key} ({total_rows} rows) into samples: {samples}")

        # If all patch counts known and sum matches total_rows, do exact slicing
        if all(c is not None for c in sample_patch_counts):
            ssum = sum(sample_patch_counts)
            if ssum == total_rows:
                # exact match -> slice in order
                idx = 0
                for sname, cnt in zip(samples, sample_patch_counts):
                    end = idx + cnt
                    out[sname] = {
                        'preds': preds_df.iloc[idx:end].copy().reset_index(drop=False).set_index(pd.Index(preds_df.index[idx:end])),
                        'targets': targets_df.iloc[idx:end].copy().reset_index(drop=False).set_index(pd.Index(targets_df.index[idx:end]))
                    }
                    # Actually preserve original index (we'll overwrite later when we attach barcodes)
                    out[sname]['preds'].index = preds_df.index[idx:end]
                    out[sname]['targets'].index = targets_df.index[idx:end]
                    if verbose:
                        print(f"  -> {sname}: rows {idx}:{end} ({cnt})")
                    idx = end
                continue
            else:
                if verbose:
                    print(f"[warn] sum of patch counts {ssum} != preds rows {total_rows}. Will attempt fallback split.")
        else:
            if verbose:
                print(f"[warn] some patch counts missing for split {split_num}: missing {missing}. Will attempt fallback split.")

        # Fallback: try to split by available counts where possible and then proportionally split remainder
        known = [(s, c) for s, c in zip(samples, sample_patch_counts) if c is not None]
        unknown = [s for s, c in zip(samples, sample_patch_counts) if c is None]

        # start by assigning known counts
        slices = {}
        idx = 0
        for s, c in known:
            slices[s] = (idx, idx + c)
            idx += c

        remaining_rows = total_rows - idx
        if remaining_rows < 0:
            # counts overshoot, fallback: proportional slicing across all samples based on known counts
            if verbose:
                print(f"[warn] known counts overshoot total_rows. Doing proportional split across all samples by available counts.")
            # compute proportions from known counts but include unknown as equal share
            base_counts = [c if c is not None else 1 for c in sample_patch_counts]
            base_sum = sum(base_counts)
            assigned = 0
            idx = 0
            for i_s, s in enumerate(samples):
                alloc = int(round(total_rows * base_counts[i_s] / float(base_sum)))
                end = min(total_rows, idx + alloc)
                slices[s] = (idx, end)
                assigned += (end - idx)
                idx = end
            # ensure we consumed all rows (adjust last)
            last_s = samples[-1]
            start, _ = slices[last_s]
            slices[last_s] = (start, total_rows)
        else:
            # distribute remaining_rows among unknown samples evenly (or proportional to known)
            n_unknown = len(unknown)
            if n_unknown == 0:
                # all known but sum < total_rows; append remainder to last sample
                last_s = samples[-1]
                s0, e0 = slices.get(last_s, (None, None))
                if s0 is None:
                    slices[last_s] = (0, total_rows)
                else:
                    slices[last_s] = (s0, total_rows)
            else:
                # simple even split among unknowns
                per = remaining_rows // n_unknown
                extra = remaining_rows % n_unknown
                for i, s in enumerate(unknown):
                    start = idx
                    add = per + (1 if i < extra else 0)
                    end = start + add
                    slices[s] = (start, end)
                    idx = end
                # if any remaining rows (numerical drift), append to last unknown
                if idx < total_rows:
                    last_u = unknown[-1]
                    s0, e0 = slices[last_u]
                    slices[last_u] = (s0, total_rows)

        # now build out entries using slices
        for s in samples:
            start, end = slices[s]
            # clamp
            start = max(0, int(start))
            end = max(start, int(min(total_rows, end)))
            out[s] = {}
            out[s]['preds'] = preds_df.iloc[start:end].copy()
            out[s]['targets'] = targets_df.iloc[start:end].copy()
            # keep their index for now (we will set index to barcodes when we attach later)
            if verbose:
                print(f"  fallback -> {s}: rows {start}:{end} (len {end-start})")

    if verbose:
        print(f"[done] expanded splits: produced {len(out)} sample entries")
    return out


def attach_barcodes_to_formatted_inference_auto(
    formatted_inference,
    dataset_name,
    base_dir="/project/simmons_hts/kxu/hest/eval/data",
    subdir="patches",
    barcode_ds='barcode',
    coords_ds='coords',
    verbose=True,
):
    """
    Automatically attach barcodes to each formatted_inference entry.

    Looks for patch files under:
        {base_dir}/{dataset_name}/{subdir}/
    Expects files like:
        {sample_id}.h5  (matching keys in formatted_inference)

    For each sample key found, reads barcodes+coords, cleans them,
    and sets them as the index of preds/targets.
    Returns dict sample_key -> patch_meta_df
    """
    patch_dir = os.path.join(base_dir, dataset_name, subdir)
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")

    patch_meta_map = {}

    for sample_key, entry in formatted_inference.items():
        # expected file path
        patch_path = os.path.join(patch_dir, f"{sample_key}.h5")
        if not os.path.isfile(patch_path):
            if verbose:
                print(f"[skip] {sample_key}: no patch file {patch_path}")
            continue

        # read patch metadata
        patch_meta_df = read_patch_meta_from_h5(patch_path, barcode_ds, coords_ds, verbose=verbose)
        patch_meta_map[sample_key] = patch_meta_df

        preds_df = entry.get("preds")
        targets_df = entry.get("targets")
        if preds_df is None or targets_df is None:
            if verbose:
                print(f"[warn] {sample_key}: missing preds/targets")
            continue

        n_b = len(patch_meta_df)
        n_p, n_t = preds_df.shape[0], targets_df.shape[0]
        if n_p != n_b or n_t != n_b:
            if verbose:
                print(f"[warn] {sample_key}: mismatch (pred={n_p}, target={n_t}, barcodes={n_b})")
        min_len = min(n_b, n_p, n_t)
        used_barcodes = patch_meta_df["barcode"].iloc[:min_len].astype(str).tolist()

        formatted_inference[sample_key]["preds"] = preds_df.iloc[:min_len].copy()
        formatted_inference[sample_key]["targets"] = targets_df.iloc[:min_len].copy()
        formatted_inference[sample_key]["preds"].index = used_barcodes
        formatted_inference[sample_key]["targets"].index = used_barcodes

        if verbose:
            print(f"[ok] {sample_key}: attached {min_len} barcodes")

    if verbose:
        print(f"[done] processed {len(patch_meta_map)} samples from {patch_dir}")

    return patch_meta_map


def add_formatted_inference_to_adata(
    adata_list,
    formatted_inference,
    pred_layer_name='pred',
    target_layer_name='target',
    fill_missing_with=np.nan,
    overwrite_layers=True,
    verbose=True
):
    """
    Write barcode-indexed formatted_inference DataFrames into adata.layers matched by barcode.

    This version **sanitises each matched AnnData in-place first** so the resulting adata_list
    is safe to write to .h5ad (no pandas StringDtype / StringArray / weird categorical categories).
    Returns summary dict per sample.
    """

    def _norm_name(s):
        if s is None:
            return None
        return re.sub(r'[-_.\s]', '', str(s)).lower()

    def infer_sample_name_from_adata(adata):
        if 'sample_id' in adata.obs.columns:
            return str(adata.obs['sample_id'].astype(str).iat[0])
        if 'sample_id' in getattr(adata, 'uns', {}):
            return str(adata.uns['sample_id'])
        spatial = getattr(adata, 'uns', {}).get('spatial', {})
        if isinstance(spatial, dict):
            if 'name' in spatial:
                return str(spatial['name'])
            if 'images' in spatial and isinstance(spatial['images'], dict):
                keys = list(spatial['images'].keys())
                if len(keys) == 1:
                    return str(keys[0])
        return None

    # ---------------- sanitiser (in-place) ----------------
    def _sanitize_anndata_for_h5ad_inplace(adata, verbose_local=False):
        """
        Convert pandas StringDtype / StringArray and categorical categories
        in adata.obs and adata.var to plain Python strings (object dtype).
        Mutates adata in-place.
        """
        def _sanitize_df_inplace(df, df_name):
            for col in list(df.columns):
                try:
                    ser = df[col]
                except Exception:
                    continue
                try:
                    dtype = ser.dtype
                except Exception:
                    dtype = None

                # pandas string dtype (StringDtype / StringArray) -> object
                if pd.api.types.is_string_dtype(dtype):
                    df[col] = ser.astype(object)

                # categorical -> ensure categories are plain Python strings or convert to object
                elif pd.api.types.is_categorical_dtype(dtype):
                    try:
                        cats = ser.cat.categories
                        # Coerce categories to plain strings
                        new_cats = [str(x) for x in cats]
                        df[col] = pd.Categorical(ser.astype(str), categories=new_cats, ordered=ser.cat.ordered)
                    except Exception:
                        # fallback: cast entire series to object strings
                        df[col] = ser.astype(object)

            # sanitize index if StringDtype-like
            try:
                if pd.api.types.is_string_dtype(df.index.dtype):
                    df.index = df.index.astype(object)
            except Exception:
                try:
                    df.index = df.index.astype(str)
                except Exception:
                    pass

        # sanitize obs and var in-place
        if hasattr(adata, 'obs') and isinstance(adata.obs, pd.DataFrame):
            _sanitize_df_inplace(adata.obs, 'obs')
        if hasattr(adata, 'var') and isinstance(adata.var, pd.DataFrame):
            _sanitize_df_inplace(adata.var, 'var')

        # ensure obs.index and var.index are plain str (not pandas extension dtypes)
        try:
            if pd.api.types.is_string_dtype(adata.obs.index.dtype):
                adata.obs.index = adata.obs.index.astype(object)
        except Exception:
            adata.obs.index = adata.obs.index.astype(str)
        try:
            if pd.api.types.is_string_dtype(adata.var.index.dtype):
                adata.var.index = adata.var.index.astype(object)
        except Exception:
            adata.var.index = adata.var.index.astype(str)

        return adata

    # ---------------- build sample lookup ----------------
    sample_to_adata_idx = {}
    for i, ad in enumerate(adata_list):
        inferred = infer_sample_name_from_adata(ad)
        if inferred is not None:
            sample_to_adata_idx[inferred] = i
            sample_to_adata_idx[_norm_name(inferred)] = i
        # also add sample_id (literal) if present
        if 'sample_id' in ad.obs.columns:
            s = str(ad.obs['sample_id'].iat[0])
            sample_to_adata_idx[s] = i
            sample_to_adata_idx[_norm_name(s)] = i

    summary = {}
    used_adata_indices = set()

    # ---------------- main loop ----------------
    for sample_key, entry in formatted_inference.items():
        preds_df = entry.get('preds')
        targets_df = entry.get('targets')
        if preds_df is None or targets_df is None:
            if verbose:
                print(f"[WARN] sample {sample_key} missing 'preds' or 'targets' -> skipping")
            summary[sample_key] = {'matched': False, 'reason': 'missing preds/targets'}
            continue

        # normalize preds/targets frames
        preds_df = preds_df.copy()
        targets_df = targets_df.copy()
        preds_df.index = preds_df.index.astype(str)
        targets_df.index = targets_df.index.astype(str)
        preds_df.columns = preds_df.columns.astype(str)
        targets_df.columns = targets_df.columns.astype(str)

        # try to find matching adata index
        ad_idx = None
        if sample_key in sample_to_adata_idx:
            ad_idx = sample_to_adata_idx[sample_key]
        else:
            nk = _norm_name(sample_key)
            if nk in sample_to_adata_idx:
                ad_idx = sample_to_adata_idx[nk]
            else:
                for k, idx in sample_to_adata_idx.items():
                    if _norm_name(k) == nk:
                        ad_idx = idx
                        break

        if ad_idx is None:
            # fuzzy match by barcode intersection
            best_idx = None
            best_inter = 0
            preds_index_set = set(preds_df.index)
            for i, ad in enumerate(adata_list):
                inter = len(preds_index_set.intersection(set(map(str, ad.obs_names))))
                if inter > best_inter:
                    best_inter = inter
                    best_idx = i
            if best_idx is not None and best_inter > 0:
                ad_idx = best_idx
                if verbose:
                    print(f"[INFO] fuzzy matched sample_key '{sample_key}' -> adata index {ad_idx} by barcode intersection ({best_inter} matches)")
            else:
                if verbose:
                    print(f"[WARN] Could not find matching AnnData for sample_key '{sample_key}'. Skipping.")
                summary[sample_key] = {'matched': False, 'reason': 'no matching adata found'}
                continue

        # sanitize the matched AnnData in-place before doing anything else
        ad = adata_list[ad_idx]
        used_adata_indices.add(ad_idx)
        if verbose:
            print(f"[sanitize] ensuring adata_list[{ad_idx}] is h5ad-safe")
        #_sanitize_anndata_for_h5ad_inplace(ad, verbose_local=verbose)

        # canonical indexes and genes (after sanitization)
        ad_obs_names = list(map(str, ad.obs_names))
        ad_var_names = list(map(str, ad.var_names))

        # determine matched barcodes
        matched_barcodes = sorted(list(set(preds_df.index).intersection(ad_obs_names)))
        n_matched_barcodes = len(matched_barcodes)
        n_total_barcodes = preds_df.shape[0]

        # genes in common
        common_genes = [g for g in preds_df.columns if g in ad_var_names]
        n_common_genes = len(common_genes)
        if n_common_genes == 0:
            if verbose:
                print(f"[WARN] No common gene names between formatted_inference['{sample_key}'] and adata[{ad_idx}]. Skipping sample.")
            summary[sample_key] = {'matched': False, 'reason': 'no common genes'}
            continue

        # build DataFrames aligned to adata obs/var (filled with fill_missing_with)
        preds_on_ad = pd.DataFrame(fill_missing_with, index=ad_obs_names, columns=ad_var_names, dtype=float)
        targets_on_ad = pd.DataFrame(fill_missing_with, index=ad_obs_names, columns=ad_var_names, dtype=float)

        # place matched rows / genes
        if n_matched_barcodes > 0:
            preds_on_ad.loc[matched_barcodes, common_genes] = preds_df.loc[matched_barcodes, common_genes].values
            targets_on_ad.loc[matched_barcodes, common_genes] = targets_df.loc[matched_barcodes, common_genes].values
            method = 'match_by_obs_names'
        else:
            # try barcode-like column mapping
            barcode_col = None
            for c in ['barcode', 'barcodes', 'cell_barcode', 'cell_id', 'patch_barcode']:
                if c in ad.obs.columns:
                    barcode_col = c
                    break
            if barcode_col is not None:
                mapping = {str(v): obs for v, obs in zip(ad.obs[barcode_col].astype(str).values, ad_obs_names)}
                matched_by_col = [b for b in preds_df.index if b in mapping]
                for b in matched_by_col:
                    obsname = mapping[b]
                    preds_on_ad.loc[obsname, common_genes] = preds_df.loc[b, common_genes].values
                    targets_on_ad.loc[obsname, common_genes] = targets_df.loc[b, common_genes].values
                n_matched_barcodes = len(matched_by_col)
                method = f"match_by_obs['{barcode_col}']"
            else:
                if preds_df.shape[0] == ad.n_obs:
                    preds_on_ad.loc[ad_obs_names, common_genes] = preds_df.loc[:, common_genes].values
                    targets_on_ad.loc[ad_obs_names, common_genes] = targets_df.loc[:, common_genes].values
                    n_matched_barcodes = ad.n_obs
                    method = 'assign_by_order'
                else:
                    nplace = min(preds_df.shape[0], ad.n_obs)
                    obs_slice = ad_obs_names[:nplace]
                    preds_on_ad.loc[obs_slice, common_genes] = preds_df.iloc[:nplace][common_genes].values
                    targets_on_ad.loc[obs_slice, common_genes] = targets_df.iloc[:nplace][common_genes].values
                    n_matched_barcodes = nplace
                    method = 'best_effort_slice'

        # convert to numpy float32 arrays and assign as layers
        pred_arr = preds_on_ad.values.astype(np.float32)
        target_arr = targets_on_ad.values.astype(np.float32)

        # compute per-adata spot counts & fraction with predictions
        n_spots = ad.n_obs
        try:
            spots_with_pred_mask = np.any(np.isfinite(pred_arr), axis=1)
            n_spots_with_pred = int(np.sum(spots_with_pred_mask))
            frac_spots_with_pred = float(n_spots_with_pred) / float(n_spots) if n_spots > 0 else 0.0
        except Exception:
            n_spots_with_pred = 0
            frac_spots_with_pred = 0.0

        # ensure layers exist
        if not hasattr(ad, 'layers') or ad.layers is None:
            ad.layers = {}
        if overwrite_layers or pred_layer_name not in ad.layers:
            ad.layers[pred_layer_name] = pred_arr
        else:
            if verbose:
                print(f"[INFO] skipping overwrite of existing layer {pred_layer_name} in adata[{ad_idx}]")
        if overwrite_layers or target_layer_name not in ad.layers:
            ad.layers[target_layer_name] = target_arr
        else:
            if verbose:
                print(f"[INFO] skipping overwrite of existing layer {target_layer_name} in adata[{ad_idx}]")

        # fill summary
        summary[sample_key] = {
            'matched': True,
            'adata_index': int(ad_idx),
            'n_barcodes_in_df': int(n_total_barcodes),
            'n_matched_barcodes': int(n_matched_barcodes),
            'n_common_genes': int(n_common_genes),
            'pred_layer_shape': pred_arr.shape,
            'target_layer_shape': target_arr.shape,
            'method': method,
            'n_spots': int(n_spots),
            'n_spots_with_pred': int(n_spots_with_pred),
            'frac_spots_with_pred': float(frac_spots_with_pred)
        }

        if verbose:
            print(f"[OK] sample '{sample_key}' -> adata[{ad_idx}] | matched_barcodes={n_matched_barcodes}/{n_total_barcodes} | "
                  f"common_genes={n_common_genes} | method={method} | spots_with_pred={n_spots_with_pred}/{n_spots} ({frac_spots_with_pred:.2%})")

    # report unused adatas
    unused = [i for i in range(len(adata_list)) if i not in used_adata_indices]
    if verbose:
        print("Done. Samples processed:", len(summary), "Unused adata indices:", unused)

    return summary


def save_adata_from_list(adata_list, RUN_ROOT, RUN):
    """
    Save all AnnData objects in adata_list to the run directory.
    Each file saved as <sample_name>.h5ad under os.path.join(RUN_ROOT, RUN).

    Parameters
    ----------
    adata_list : list[AnnData]
        list of AnnData objects
    RUN_ROOT : str or Path
        base directory for runs
    RUN : str
        subfolder name for this run
    filename_prefix : str or None
        optional string to prefix all filenames (e.g. 'inferred_')
    """
    save_dir = os.path.join(RUN_ROOT, 'ST_pred_results', RUN, 'pred')
    os.makedirs(save_dir, exist_ok=True)

    for i, ad in enumerate(adata_list):
        # try to infer sample name
        if 'sample_id' in ad.obs.columns:
            sample_name = str(ad.obs['sample_id'].iat[0])
        else:
            sample_name = f"{i}"

        out_path = os.path.join(save_dir, f"{sample_name}.h5ad")

        ad.write(out_path)
        print(f"[OK] Saved {sample_name} → {out_path}")

    print(f"All AnnData objects saved under {save_dir}")

def _safe_sample_name(adata, idx):
    # prefer obs sample_id, then obs sample, then uns sample_id, then fallback
    if 'sample_id' in adata.obs.columns:
        s = str(adata.obs['sample_id'].iat[0])
    elif 'sample_id' in getattr(adata, 'uns', {}):
        s = str(adata.uns['sample_id'])
    else:
        s = f"adata_{idx}"
    # sanitize for filesystem
    s = "".join(c if c.isalnum() or c in ('-', '_') else "_" for c in s)
    return s

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import scanpy as sc
from scipy.sparse import issparse

# pearsonr fallback
try:
    from scipy.stats import pearsonr
except Exception:
    def pearsonr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return (np.nan, np.nan)
        r = np.corrcoef(a[mask], b[mask])[0,1]
        return (float(r), np.nan)

def save_spatial_pred_target_pdfs_for_adata_list(
    adata_list,
    genes,
    RUN_ROOT,
    RUN,
    folder_name="cell_specific_genes",
    img_key="downscaled_fullres",
    pred_layer="pred",
    target_layer="target",
    genes_per_page=4,
    size=1.5,
    dpi=150,
    cmap=None,
    vmin=None,
    vmax=None,
    verbose=True,
):
    """
    For each AnnData in adata_list, create a PDF of spatial pred|target|raw plots for `genes`
    and save in RUN_ROOT/RUN/plots/pred/<folder_name>/<sample_name>.pdf

    Returns:
      results: dict mapping sample_name -> {"pdf_path": path_or_None, "summary": [per-gene dicts]}
    """
    outdir = os.path.join(RUN_ROOT, 'ST_pred_results',RUN, "plots", "pred", folder_name)
    os.makedirs(outdir, exist_ok=True)

    genes_list = list(genes)
    results = {}

    for i, ad in enumerate(adata_list):
        # sample name helper (same as earlier _safe_sample_name)
        if 'sample_id' in ad.obs.columns:
            sample_name = str(ad.obs['sample_id'].iat[0])
        elif 'sample' in ad.obs.columns:
            sample_name = str(ad.obs['sample'].iat[0])
        elif 'sample_id' in getattr(ad, 'uns', {}):
            sample_name = str(ad.uns['sample_id'])
        else:
            sample_name = f"adata_{i}"
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else "_" for c in sample_name)

        out_pdf = os.path.join(outdir, f"{safe_name}.pdf")

        # quick checks: pred/target layers exist?
        if pred_layer not in getattr(ad, "layers", {}):
            if verbose:
                print(f"[SKIP] adata_list[{i}] ({sample_name}): pred layer '{pred_layer}' not present.")
            results[sample_name] = {"pdf_path": None, "summary": None, "reason": f"missing layer {pred_layer}"}
            continue
        if target_layer not in getattr(ad, "layers", {}):
            if verbose:
                print(f"[SKIP] adata_list[{i}] ({sample_name}): target layer '{target_layer}' not present.")
            results[sample_name] = {"pdf_path": None, "summary": None, "reason": f"missing layer {target_layer}"}
            continue

        n_genes = len(genes_list)
        pages = int(math.ceil(n_genes / float(genes_per_page)))
        summary = []

        if verbose:
            print(f"[START] Writing PDF for adata_list[{i}] ({sample_name}) -> {out_pdf} (genes: {n_genes})")

        try:
            with PdfPages(out_pdf) as pp:
                gi = 0
                for page in range(pages):
                    # number of genes on this page
                    nrow = min(genes_per_page, n_genes - gi)

                    # grid: genes_per_page rows x 3 columns (pred | target | raw)
                    fig, axes = plt.subplots(genes_per_page, 3, figsize=(9, 12))  
                    if genes_per_page == 1:
                        axes = np.expand_dims(axes, 0)
                    fig.subplots_adjust(hspace=0.35, wspace=0.2)

                    for row in range(nrow):
                        gene = genes_list[gi]
                        gi += 1

                        pred_col = f"pred_{gene}"
                        target_col = f"target_{gene}"
                        raw_col = f"raw_{gene}"

                        ax_pred = axes[row, 0]
                        ax_target = axes[row, 1]
                        ax_raw = axes[row, 2]

                        if str(gene) not in list(map(str, ad.var_names)):
                            # gene missing: mark all three panels
                            ax_pred.text(0.5, 0.5, f"{gene} not in var_names", ha='center', va='center')
                            ax_pred.axis('off'); ax_target.axis('off'); ax_raw.axis('off')
                            summary.append({"gene": gene, "r": None, "n": 0})
                            continue

                        # compute gene index for ad.var
                        gi_idx = list(map(str, ad.var_names)).index(str(gene))

                        # fetch pred and target arrays (from layers)
                        pred_vals = np.array(ad.layers[pred_layer][:, gi_idx], dtype=float)
                        target_vals = np.array(ad.layers[target_layer][:, gi_idx], dtype=float)

                        # fetch raw expression: prefer ad.raw if available and contains gene, else ad.X
                        raw_vals = None
                        use_raw_name = None
                        if getattr(ad, 'raw', None) is not None:
                            # check var_names in raw
                            raw_var_names = list(map(str, ad.raw.var_names))
                            if str(gene) in raw_var_names:
                                gi_idx_raw = raw_var_names.index(str(gene))
                                raw_X = ad.raw.X
                                if issparse(raw_X):
                                    raw_vals = np.array(raw_X[:, gi_idx_raw].toarray()).ravel()
                                else:
                                    raw_vals = np.array(raw_X[:, gi_idx_raw]).ravel()
                                use_raw_name = 'raw (adata.raw.X)'
                        if raw_vals is None:
                            # fallback to ad.X
                            X = ad.X
                            if issparse(X):
                                raw_vals = np.array(X[:, gi_idx].toarray()).ravel()
                            else:
                                raw_vals = np.array(X[:, gi_idx]).ravel()
                            use_raw_name = 'X (adata.X)'

                        # attach temporary obs columns for scanpy plotting
                        # create copies of arrays as pandas-friendly
                        ad.obs[pred_col] = pred_vals
                        ad.obs[target_col] = target_vals
                        ad.obs[raw_col] = raw_vals

                        # compute stats between pred and target (as before)
                        mask = np.isfinite(pred_vals) & np.isfinite(target_vals)
                        n_valid = int(mask.sum())
                        if n_valid >= 2:
                            r_val, _ = pearsonr(pred_vals[mask], target_vals[mask])
                        else:
                            r_val = np.nan

                        # plot pred
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            try:
                                sc.pl.spatial(
                                    ad,
                                    img_key=img_key,
                                    color=[pred_col],
                                    ax=ax_pred,
                                    show=False,
                                    cmap=cmap,
                                    size=size
                                )
                            except Exception:
                                # fallback copy-fig approach
                                sc.pl.spatial(ad, img_key=img_key, color=[pred_col], show=False,
                                              cmap=cmap, size=size)
                                fig_src = plt.gcf()
                                ax_src = fig_src.axes[0]
                                for artist in ax_src.get_children():
                                    try:
                                        ax_pred.add_artist(artist)
                                    except Exception:
                                        pass
                                plt.close(fig_src)

                        # plot target
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            try:
                                sc.pl.spatial(
                                    ad,
                                    img_key=img_key,
                                    color=[target_col],
                                    ax=ax_target,
                                    show=False,
                                    cmap=cmap,
                                    size=size,
                                )
                            except Exception:
                                sc.pl.spatial(ad, img_key=img_key, color=[target_col], show=False,
                                              cmap=cmap, size=size)
                                fig_src = plt.gcf()
                                ax_src = fig_src.axes[0]
                                for artist in ax_src.get_children():
                                    try:
                                        ax_target.add_artist(artist)
                                    except Exception:
                                        pass
                                plt.close(fig_src)

                        # plot raw
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            try:
                                sc.pl.spatial(
                                    ad,
                                    img_key=img_key,
                                    color=[raw_col],
                                    ax=ax_raw,
                                    show=False,
                                    cmap=cmap,
                                    size=size,
                                )
                            except Exception:
                                sc.pl.spatial(ad, img_key=img_key, color=[raw_col], show=False,
                                              cmap=cmap, size=size)
                                fig_src = plt.gcf()
                                ax_src = fig_src.axes[0]
                                for artist in ax_src.get_children():
                                    try:
                                        ax_raw.add_artist(artist)
                                    except Exception:
                                        pass
                                plt.close(fig_src)

                        # annotate pred only
                        ann = f"r={r_val:.3f}  n={n_valid}" if np.isfinite(r_val) else f"r=NaN  n={n_valid}"
                        ax_pred.text(0.02, 0.95, ann, transform=ax_pred.transAxes,
                                     fontsize=9, color='white', va='top',
                                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

                        ax_pred.set_title(f"pred: {gene}", fontsize=9)
                        ax_target.set_title(f"target: {gene}", fontsize=9)
                        ax_raw.set_title(f"raw: {gene}", fontsize=9)

                        ax_pred.axis('off')
                        ax_target.axis('off')
                        ax_raw.axis('off')

                        # cleanup temporary obs cols
                        for c in (pred_col, target_col, raw_col):
                            if c in ad.obs.columns:
                                del ad.obs[c]

                        summary.append({"gene": gene, "r": float(r_val) if np.isfinite(r_val) else None, "n": n_valid})

                    # hide unused rows if any (axes shape: genes_per_page x 3)
                    total_rows = axes.shape[0]
                    if nrow < total_rows:
                        for rr in range(nrow, total_rows):
                            for cc in range(3):
                                axes[rr, cc].axis('off')

                    # save this page
                    pp.savefig(fig, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)

            if verbose:
                print(f"[OK] Wrote PDF: {out_pdf}")

            results[sample_name] = {"pdf_path": out_pdf, "summary": summary, "reason": None}

        except Exception as e:
            # fail-safe: if anything goes wrong clean up partial file and continue
            try:
                if os.path.exists(out_pdf):
                    os.remove(out_pdf)
            except Exception:
                pass
            if verbose:
                print(f"[ERROR] Failed to create PDF for {sample_name}: {e}")
            results[sample_name] = {"pdf_path": None, "summary": None, "reason": str(e)}

    if verbose:
        print(f"[DONE] saved PDFs to {outdir}")
    return results

# wrapper
def add_inference_to_adata_and_plot(
    RUN,
    RUN_ROOT,
    dataset_name=None,
    gene_lists_to_plot=None,        # dict name-> (dataset, gene_list_name) or list of gene lists
    base_data_dir="/project/simmons_hts/kxu/hest/eval/data",
    patches_subdir="patches",
    adata_subdir_template="{}" ,    # for data dir if you build differently; default expects base_dir/dataset_name/adata
    plot_folder_root='plots/pred',
    pred_layer_name='pred',
    target_layer_name='target',
    genes_per_page=4,
    img_key="downscaled_fullres",
    size=1.2,
    dpi=100,
    cmap=None,
    verbose=True
):
    """
    High-level wrapper that:
      - extracts best model + inference for RUN
      - formats inference with gene list
      - reads dataset adata files into adata_list and attaches sample_id
      - attaches barcodes to formatted_inference (reads patch files under base_data_dir/dataset_name/patches)
      - writes pred/target arrays into each AnnData.layers (sanitises AnnData)
      - saves adata_list to disk under RUN_ROOT/RUN/<save_adata_subfolder>/
      - generates PDFs into RUN_ROOT/RUN/<plot_folder_root>/<gene_list_folder>/
    Returns a dict with summary, saved files, plot results.
    """

    # --------------- 1) extract best model + load inference ----------------
    if verbose: print("[step] extract best model corrs")
    best_model, dataset, df_corr = extract_best_model_gene_corrs(RUN)

    if verbose: print("[step] load best model inference and metadata")
    best_model_info, dataset_name_from_inference, inference_dumps, gene_list_name = load_best_model_inference(RUN)

    # If dataset_name specified as arg, prefer that; else use loaded dataset_name
    dataset_name = dataset_name or dataset_name_from_inference
    if verbose: print(f"[info] using dataset_name = {dataset_name}")

    if verbose: print("[step] load gene list used to format inference")
    gene_list = load_gene_list(dataset_name, gene_list_name)

    if verbose: print("[step] format inference with gene list")
    formatted_inference = format_inference_with_genes(inference_dumps, gene_list)

    if verbose: print("[step] remap split keys -> sample names using test splits")
    df_test_splits = get_test_splits(RUN)

    # --------------- 2) load adata_list for dataset ----------------
    data_dir = os.path.join(base_data_dir, dataset_name, 'adata')
    if verbose: print(f"[step] loading adata files from {data_dir}")
    adata_list = []
    sample_names = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".h5ad"):
            continue
        sample = os.path.splitext(fname)[0]
        fpath = os.path.join(data_dir, fname)
        if verbose: print(f"  loading {fpath}")
        adata = sc.read_h5ad(fpath)
        # store sample name in .obs for later saving and matching
        adata.obs['sample_id'] = sample
        adata_list.append(adata)
        sample_names.append(sample)


    # --------------- 3) attach patch barcodes to formatted_inference automatically -------------
    # extra sampple matching for broad splits (leave-one-patient-out CV)
    if verbose: print("[step] expand split keys into per-sample formatted_inference entries (handles LOO CV)")
    formatted_inference = expand_split_keys_to_samples(
        formatted_inference,
        df_test_splits,
        dataset_name=dataset_name,
        base_dir=base_data_dir,
        patches_subdir=patches_subdir,
        barcode_ds='barcode',
        verbose=verbose
    )

    if verbose: print("[step] attach barcodes to formatted_inference using patches directory")
    patch_meta_map = attach_barcodes_to_formatted_inference_auto(
        formatted_inference,
        dataset_name=dataset_name,
        base_dir=base_data_dir,
        subdir=patches_subdir,
        verbose=verbose
    )

    # --------------- 4) add formatted_inference to adata layers ----------------
    if verbose: print("[step] writing preds/targets into adata.layers (with sanitisation)")
    summary = add_formatted_inference_to_adata(
        adata_list,
        formatted_inference,
        pred_layer_name=pred_layer_name,
        target_layer_name=target_layer_name,
        fill_missing_with=np.nan,
        overwrite_layers=True,
        verbose=verbose
    )

    # --------------- 5) save the adata list to RUN folder ----------------
    if verbose: print("[step] saving adata_list to disk")
    saved_paths = save_adata_from_list(adata_list, RUN_ROOT, RUN)

    # --------------- 6) optionally plot gene lists ----------------
    plot_results = {}
    if gene_lists_to_plot:
        # gene_lists_to_plot can be:
        #  - list of (dataset,gene_list_name) tuples
        #  - dict name-> (dataset, gene_list_name)
        #  - list of lists of gene names
        if isinstance(gene_lists_to_plot, dict):
            iter_items = gene_lists_to_plot.items()
        elif isinstance(gene_lists_to_plot, list) and len(gene_lists_to_plot) and isinstance(gene_lists_to_plot[0], (list,tuple)):
            # interpret as list of (name, (dataset, gene_list_name)) or list of name->list
            iter_items = []
            for item in gene_lists_to_plot:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (list,tuple)):
                    iter_items.append(item)
                else:
                    # fallback: treat as anonymous list of gene names and create default name
                    iter_items.append((f"gene_list_{len(iter_items)}", item))
        else:
            # single list of gene names
            iter_items = [("genes", gene_lists_to_plot)]

        for list_name, gene_spec in iter_items:
            # if gene_spec is tuple pointing to (dataset, gene_list_name) call load_gene_list
            if isinstance(gene_spec, (list,tuple)) and len(gene_spec) == 2 and isinstance(gene_spec[0], str) and isinstance(gene_spec[1], str):
                ds, glname = gene_spec
                genes_to_plot = load_gene_list(ds, glname)
            elif isinstance(gene_spec, (list,tuple)):
                genes_to_plot = list(gene_spec)
            elif isinstance(gene_spec, str):
                # assume this is a gene_list_name for dataset_name
                genes_to_plot = load_gene_list(dataset_name, gene_spec)
            else:
                raise ValueError("Unsupported gene_lists_to_plot format")

            if verbose: print(f"[plot] creating pdfs for gene list '{list_name}' with {len(genes_to_plot)} genes")
            out = save_spatial_pred_target_pdfs_for_adata_list(
                adata_list,
                genes_to_plot,
                RUN_ROOT=RUN_ROOT,
                RUN=RUN,
                folder_name=list_name,
                img_key=img_key,
                pred_layer=pred_layer_name,
                target_layer=target_layer_name,
                genes_per_page=genes_per_page,
                size=size,
                dpi=dpi,
                cmap=cmap,
                vmin=None,
                vmax=None,
                verbose=verbose
            )
            plot_results[list_name] = out

    # final assembled information
    result = {
        'RUN': RUN,
        'dataset_name': dataset_name,
        'formatted_inference_keys': list(formatted_inference.keys()),
        'add_inference_summary': summary,
        'saved_adata_paths': saved_paths,
        'patch_meta_map_keys': list(patch_meta_map.keys()),
        'plot_results': plot_results
    }

    if verbose: print("[DONE] pipeline finished")
    return result
