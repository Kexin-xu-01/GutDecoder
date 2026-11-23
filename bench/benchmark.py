from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from operator import itemgetter
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from argparse import Namespace


import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import yaml
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import joblib

from gutdecoder.bench.fusion import fuse_embeddings, read_slide_embedding
from gutdecoder.bench.cpath_model_zoo.inference_models import (
    CustomInferenceEncoder, InferenceEncoder, inf_encoder_factory)


from hest.bench.st_dataset import H5PatchDataset
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download

from gutdecoder.bench.st_dataset import load_adata
from gutdecoder.bench.trainer import train_test_reg
from gutdecoder.utils.workflow_config import coerce_bool
from hest.bench.utils.file_utils import (read_assets_from_h5, save_hdf5,
                                         save_pkl)
from hest.bench.utils.utils import merge_dict, get_current_time

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Configurations for linear probing')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--overwrite', action='store_true', default=None)
    parser.add_argument('--bench_data_root', type=str)
    parser.add_argument('--embed_dataroot', type=str)
    parser.add_argument('--weights_root', type=str)
    parser.add_argument('--private_weights_root', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--exp_code', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--gene_list', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--kfold', action='store_true', default=None)
    parser.add_argument('--benchmark_encoders', action='store_true', default=None)
    # --normalize is kept as an alias; the stored setting is still called `normalize`.
    parser.add_argument('--log_normalize', '--normalize', dest='normalize', type=coerce_bool,
                        metavar='{true,false}')
    parser.add_argument('--library_size_normalize', type=coerce_bool, metavar='{true,false}')
    parser.add_argument('--dimreduce', type=str)
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument('--encoders', nargs='+')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--config', type=str)
    parser.add_argument('--slide_emb_root', type=str)
    parser.add_argument('--fusion', type=str)
    return parser.parse_args(argv)
            
@dataclass
class BenchmarkConfig:
    """
    Dataclass containing default arguments for benchmarking. Note that arguments are overwritten in ``benchmark`` either through: 
    - CLI arguments
    - Function kwargs
    - Config file (which paths needs to be specified in the CLI or kwargs)
    """
    seed: int = 1
    overwrite: bool = False

    bench_data_root: Optional[str] = 'eval/bench_data'
    # Benchmark data will automatically be downloaded to this path

    embed_dataroot: Optional[str] = 'eval/ST_data_emb'
    # Embeddings generated during benchmarking will be saved to this path

    weights_root: Optional[str] = 'eval/fm_v1'
    # Path to patch encoder weights

    results_dir: Optional[str] = 'eval/ST_pred_results'
    # Path to benchmark results

    batch_size: int = 128
    # Batch size used during embedding extraction

    num_workers: int = 1
    # Number of workers used during embedding extraction

    private_weights_root: Optional[str] = None
    exp_code: Optional[str] = None
    gene_list: str = 'var_50genes.json'
    method: str = 'ridge'
    alpha: Optional[float] = None
    kfold: bool = False
    benchmark_encoders: bool = False
    normalize: bool = True
    library_size_normalize: bool = True
    dimreduce: Optional[str] = "PCA"
    latent_dim: int = 256
    encoders: list = field(default_factory=lambda: ['resnet50'])
    datasets: list = field(default_factory=lambda: ['IDC'])
    config: Optional[str] = None
    slide_emb_root: Optional[str] = None
    fusion: Optional[str] = "concat"

def get_path(path):
    src = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../'))
    if path.startswith('./'):
        return os.path.join(src, path)
    return path


def benchmark_grid(args, device, model_names, datasets: List[str], save_dir, slide_emb_root, fusion, custom_encoder=None) -> Tuple[list, dict]:
    """ Execute predict_folds for each encoders and datasets and dump the results in a nested directory structure """
    
    dataset_perfs = []
    for dataset in datasets:
        bench_data_root = os.path.join(get_path(args.bench_data_root), dataset)
        enc_perfs = []
        for model_name in model_names:
            logger.info(f'HESTBench task: {dataset}, Encoder: {model_name}')
            exp_save_dir = os.path.join(save_dir, dataset, model_name)
            os.makedirs(exp_save_dir, exist_ok=True)
            enc_results = predict_folds(args, exp_save_dir, model_name, dataset, device, bench_data_root, slide_emb_root, fusion, custom_encoder)
            
            enc_perfs.append({
                'encoder_name': model_name,
                'pearson_mean': enc_results['pearson_mean'], 
                'pearson_std': enc_results['pearson_std'], 
            })
            
        with open(os.path.join(save_dir, dataset, 'enc_results.json'), 'w') as f:
            enc_perfs = sorted(enc_perfs, key=itemgetter('pearson_mean'), reverse=True)
            json.dump({'results': enc_perfs}, f, sort_keys=True, indent=4)
            

        dataset_perfs.append({
            'dataset_name': dataset,
            'results': enc_perfs
        })
        
    perf_per_enc = {}
    for dataset_perf in dataset_perfs:
        for enc_perf in dataset_perf['results']:
            perf_per_enc[enc_perf['encoder_name']] = perf_per_enc.get(enc_perf['encoder_name'], []) + [enc_perf['pearson_mean']]
          
    row_dicts = []
    for dataset_perf in dataset_perfs:
        row_dict = {}
        row_dict['dataset'] = dataset_perf['dataset_name']
        for result in dataset_perf['results']:
            enc_name = result['encoder_name']
            row_dict[f'{enc_name}_mean'] = result['pearson_mean']
            row_dict[f'{enc_name}_std'] = result['pearson_std']
        row_dicts.append(row_dict)
    df = pd.DataFrame(row_dicts)
    
    df.to_csv(os.path.join(save_dir, 'dataset_results.csv'))
            
    for key, val in perf_per_enc.items():
        perf_per_enc[key] = np.mean(val)
    perf_per_enc = dict(sorted(perf_per_enc.items(), key=lambda item: item[1], reverse=True))
        
    with open(os.path.join(save_dir, 'dataset_results.json'), 'w') as f:
        json.dump({'results': dataset_perfs, 'average': perf_per_enc}, f, sort_keys=True, indent=4)
    
    return dataset_perfs, perf_per_enc
        

def post_collate_fn(batch):
    """
    Post collate function to clean up batch
    """
    if batch["imgs"].dim() == 5:
        assert batch["imgs"].size(0) == 1
        batch["imgs"] = batch["imgs"].squeeze(0)
    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)
    return batch


def embed_tiles(
    dataloader: DataLoader,
    model: torch.nn.Module,
    embedding_save_path: str,
    device: str,
    precision
):
    """ Extract embeddings from tiles using `encoder` and save to an h5 file (TODO move to hestcore) """
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch['imgs'].to(device).float()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            embeddings = model(imgs)
        if batch_idx == 0:
            mode = 'w'
        else:
            mode = 'a'
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        for key, val in batch.items():
            if key == 'imgs':
                continue
            arr = np.array(val)
            if arr.dtype.kind == 'U':  # unicode strings — h5py can't save these; encode to bytes
                arr = np.array([v.encode() for v in val])
            asset_dict[key] = arr
        save_hdf5(embedding_save_path,
                  asset_dict=asset_dict,
                  mode=mode)
    return embedding_save_path


def get_bench_weights(weights_root, name):
    local_ckpt_registry = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_ckpts.json')
    with open(local_ckpt_registry, 'r') as f:
        ckpt_registry = json.load(f)
    if name in ckpt_registry:
        path = ckpt_registry[name]
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(weights_root, path)
    else:
        raise ValueError(f"Please specify the weights path to {name} in {local_ckpt_registry}")


def _fuse_tile_and_slide_assets(
    tile_assets,
    slide_h5_path,
    sample_id=None,
    fusion="concat",
):
    """
    Given tile-level assets from read_assets_from_h5(embed_path),
    read the slide-level embedding and fuse it with every tile embedding.

    tile_assets must contain:
        tile_assets["embeddings"] shape (n_tiles, d_tile)
        tile_assets["barcodes"] shape (n_tiles, 1) or (n_tiles,)
    """
    if "embeddings" not in tile_assets:
        raise KeyError(
            f"Tile assets for {sample_id} do not contain 'embeddings'. "
            f"Available keys: {list(tile_assets.keys())}"
        )

    tile_emb = np.asarray(tile_assets["embeddings"], dtype=np.float32)
    slide_emb = read_slide_embedding(slide_h5_path)
    fused_emb = fuse_embeddings(tile_emb, slide_emb, fusion)

    tile_assets["tile_embeddings"] = tile_emb
    tile_assets["slide_embeddings"] = fused_emb[:, tile_emb.shape[1]:]
    tile_assets["embeddings"] = fused_emb

    return tile_assets


def predict_single_split_tile_slide_fusion(
    train_split, test_split, args, save_dir, dataset_name, model_name, device,
    bench_data_root, custom_encoder, extract_tiles, slide_emb_root, fusion="concat",
):
    """Compatibility entry point for tile + slide feature concatenation."""
    return predict_single_split(
        train_split, test_split, args, save_dir, dataset_name, model_name, device,
        bench_data_root, custom_encoder, extract_tiles, slide_emb_root, fusion,
        _legacy_fusion_schema=True,
    )


def merge_fold_results(arr):
    aggr_dict = {}
    for dict in arr:
        for item in dict['pearson_corrs']:
            gene_name = item['name']
            correlation = item['pearson_corr']
            aggr_dict[gene_name] = aggr_dict.get(gene_name, []) + [correlation]
    
    aggr_results = []
    all_corrs = []
    for key, value in aggr_dict.items():
        aggr_results.append({
            "name": key,
            "pearson_corrs": value,
            "mean": np.mean(value),
            "std": np.std(value)
        })
        all_corrs += value
        
    mean_per_split = [d['pearson_mean'] for d in arr]    
        
    return {"pearson_corrs": aggr_results, "pearson_mean": np.mean(mean_per_split), "pearson_std": np.std(mean_per_split), "mean_per_split": mean_per_split}

def _load_split_assets(train_df, test_df, args, bench_data_root, embedding_dir,
                       genes, slide_emb_root, fusion):
    """Load expression and embeddings in the legacy sample/barcode order."""
    use_slide_embedding = slide_emb_root is not None
    all_split_assets = {}
    test_sample_ids, test_sample_barcodes = [], []
    for split_key, split in zip(["train", "test"], [train_df, test_df]):
        split_assets = {}

        for i in tqdm(range(len(split))):
            sample_id = split.iloc[i]["sample_id"]

            embed_path = os.path.join(embedding_dir, f"{sample_id}.h5")
            expr_path = os.path.join(bench_data_root, split.iloc[i]["expr_path"])

            if not os.path.isfile(embed_path):
                raise FileNotFoundError(f"Missing tile embedding H5: {embed_path}")

            # Tile-level H5 assets
            assets, _ = read_assets_from_h5(embed_path)

            # Optional slide-level fusion
            if use_slide_embedding:
                slide_h5_path = os.path.join(
                    slide_emb_root,
                    f"{sample_id}.h5",
                )

                if not os.path.isfile(slide_h5_path):
                    raise FileNotFoundError(
                        f"Missing slide embedding H5 for {sample_id}: {slide_h5_path}"
                    )

                assets = _fuse_tile_and_slide_assets(
                    tile_assets=assets,
                    slide_h5_path=slide_h5_path,
                    sample_id=sample_id,
                    fusion=fusion,
                )
            else:
                # Keep this for consistent downstream debugging.
                # The actual model still uses assets["embeddings"] only.
                pass  # Alias assigned after concatenation to avoid a second copy.

            _raw_bc = assets.get("barcodes")
            barcodes = _raw_bc.flatten().astype(str).tolist() if _raw_bc is not None else None

            adata = load_adata(
                expr_path,
                genes=genes,
                barcodes=barcodes,
                normalize=args.normalize,
                library_size_normalize=args.library_size_normalize,
            )

            assets["adata"] = adata.values
            if use_slide_embedding:
                # Diagnostic views can be reconstructed from fused features after
                # concatenation, avoiding two additional full-size allocations.
                tile_width = assets.pop("tile_embeddings").shape[1]
                assets.pop("slide_embeddings")

            split_assets = merge_dict(split_assets, assets)

            if split_key == "test":
                test_sample_ids.append(sample_id)
                test_sample_barcodes.append(barcodes)

        for key, val in split_assets.items():
            split_assets[key] = np.concatenate(val, axis=0)

        if use_slide_embedding:
            split_assets["tile_embeddings"] = split_assets["embeddings"][:, :tile_width]
            split_assets["slide_embeddings"] = split_assets["embeddings"][:, tile_width:]
        else:
            split_assets["tile_embeddings"] = split_assets["embeddings"]
        all_split_assets[split_key] = split_assets

        logger.info(
            f"Loaded {split_key} split with "
            f"{len(split_assets['embeddings'])} samples: "
            f"{split_assets['embeddings'].shape}"
        )

        if use_slide_embedding:
            logger.info(
                f"{split_key} tile embeddings: "
                f"{split_assets['tile_embeddings'].shape}; "
                f"slide embeddings repeated: "
                f"{split_assets['slide_embeddings'].shape}; "
                f"fused embeddings: "
                f"{split_assets['embeddings'].shape}"
            )
        else:
            logger.info(
                f"{split_key} tile-only embeddings: "
                f"{split_assets['embeddings'].shape}"
            )

    return all_split_assets, test_sample_ids, test_sample_barcodes


def _extract_sample_embeddings(tile_h5_path, embed_path, encoder, args, device):
    """Extract one sample using the existing transforms, order, and precision."""
    dataset = H5PatchDataset(tile_h5_path, img_transform=encoder.eval_transforms)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    return embed_tiles(loader, encoder, embed_path, device, encoder.precision)


def _fit_split(X_train, X_test, y_train, y_test, args, genes):
    """Fit preprocessing on training rows only, then fit the existing regressor."""
    model_bundle = {}

    # Optional PCA
    if args.dimreduce == "PCA":
        from sklearn.decomposition import PCA

        logger.info("perform PCA dim reduction")

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "PCA",
                    PCA(
                        n_components=args.latent_dim,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

        X_train = torch.Tensor(pipe.fit_transform(X_train))
        X_test = torch.Tensor(pipe.transform(X_test))

        model_bundle["pca_pipeline"] = pipe

    # Regression
    probe_results, linprobe_dump, reg = train_test_reg(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=args.seed,
        genes=genes,
        alpha=args.alpha,
        method=args.method,
    )

    model_bundle["regression_model"] = reg
    return probe_results, linprobe_dump, model_bundle, X_train, X_test


def predict_single_split(
    train_split,
    test_split,
    args,
    save_dir,
    dataset_name,
    model_name,
    device,
    bench_data_root,
    custom_encoder,
    extract_tiles,
    slide_emb_root=None,
    fusion="concat",
    *,
    _legacy_fusion_schema=False,
):
    """
    Predict a single split.

    If slide_emb_root is None:
        - behaves like the original tile-only version
        - uses only tile-level embeddings

    If slide_emb_root is provided:
        - reads slide-level H5 from:
            slide_emb_root/{sample_id}.h5
        - repeats the slide embedding for every barcode/tile
        - fuses tile + slide embeddings by concatenation

    Parameters
    ----------
    slide_emb_root : str or None
        Optional path to slide-level H5 embeddings.

        Example:
            /project/gutdecoder/kxu/xenium/he/xenium/trident_processed/hest/20x_512px_0px_overlap/slide_features_titan

        If None, no slide-level embeddings are used.

    fusion : str
        Currently only "concat".
    """

    use_slide_embedding = slide_emb_root is not None

    # Resolve split paths
    if not os.path.isfile(train_split):
        train_split = os.path.join(bench_data_root, "splits", train_split)
    if not os.path.isfile(test_split):
        test_split = os.path.join(bench_data_root, "splits", test_split)

    train_df = pd.read_csv(train_split)
    test_df = pd.read_csv(test_split)

    # Tile embedding directory
    embedding_dir = os.path.join(args.embed_dataroot, dataset_name, model_name)
    os.makedirs(embedding_dir, exist_ok=True)

    # Embed patches if requested
    logger.info(f"Embedding tiles for {dataset_name} using {model_name} encoder")

    # A custom encoder can't be identified by name, so cached embeddings may be stale.
    overwrite = args.overwrite or model_name == "custom_encoder"
    encoder = None
    for split in (train_df, test_df):
        for row in split.itertuples(index=False):
            tile_h5_path = os.path.join(bench_data_root, row.patches_path)
            assert os.path.isfile(tile_h5_path), f"Missing tile H5: {tile_h5_path}"
            embed_path = os.path.join(embedding_dir, f"{row.sample_id}.h5")
            if not extract_tiles:
                continue
            if os.path.isfile(embed_path) and not overwrite:
                logger.info(f"Skipping {row.sample_id} as it already exists")
                continue
            if encoder is None:
                encoder = (custom_encoder if model_name == "custom_encoder" else
                           inf_encoder_factory(model_name)(get_bench_weights(args.weights_root, model_name)))
                encoder.eval()
                encoder.to(device)
            _extract_sample_embeddings(tile_h5_path, embed_path, encoder, args, device)

    # Save config
    os.makedirs(save_dir, exist_ok=True)

    config_dict = vars(args).copy()
    config_dict["slide_emb_root"] = slide_emb_root
    if not _legacy_fusion_schema:
        config_dict["use_slide_embedding"] = use_slide_embedding
    config_dict["fusion"] = fusion if use_slide_embedding else None

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    # Load gene list
    gene_list = args.gene_list
    logger.info(f"using gene_list {gene_list}")

    with open(os.path.join(bench_data_root, gene_list), "r") as f:
        genes = json.load(f)["genes"]

    # Load embeddings and expression
    all_split_assets, test_sample_ids, test_sample_barcodes = _load_split_assets(
        train_df, test_df, args, bench_data_root, embedding_dir, genes, slide_emb_root, fusion,
    )

    # Train/test arrays
    X_train = all_split_assets["train"]["embeddings"]
    y_train = all_split_assets["train"]["adata"]

    X_test = all_split_assets["test"]["embeddings"]
    y_test = all_split_assets["test"]["adata"]

    probe_results, linprobe_dump, model_bundle, X_train, X_test = _fit_split(
        X_train, X_test, y_train, y_test, args, genes,
    )

    # Keep metadata for visualization alignment
    linprobe_dump["test_sample_ids"] = test_sample_ids
    linprobe_dump["test_sample_barcodes"] = test_sample_barcodes

    # Record whether slide fusion was used
    if not _legacy_fusion_schema:
        linprobe_dump["use_slide_embedding"] = use_slide_embedding
    linprobe_dump["slide_emb_root"] = slide_emb_root
    linprobe_dump["fusion"] = fusion if use_slide_embedding else None

    if not _legacy_fusion_schema:
        model_bundle["use_slide_embedding"] = use_slide_embedding
    model_bundle["slide_emb_root"] = slide_emb_root
    model_bundle["fusion"] = fusion if use_slide_embedding else None

    model_path = os.path.join(save_dir, "model.pkl")
    joblib.dump(model_bundle, model_path)

    logger.info(f"Model saved in '{model_path}'")

    if use_slide_embedding:
        logger.info(f"Fused train shape: {X_train.shape}")
        logger.info(f"Fused test shape: {X_test.shape}")
    else:
        logger.info(f"Tile-only train shape: {X_train.shape}")
        logger.info(f"Tile-only test shape: {X_test.shape}")

    # Save outputs
    probe_summary = {}
    probe_summary.update({"n_train": len(y_train), "n_test": len(y_test)})
    probe_summary.update({key: val for key, val in probe_results.items()})

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(probe_results, f, sort_keys=True, indent=4)

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(probe_summary, f, sort_keys=True, indent=4)

    save_pkl(os.path.join(save_dir, "inference_dump.pkl"), linprobe_dump)
    from gutdecoder.utils.provenance import write_manifest
    write_manifest(Path(save_dir) / "benchmark_manifest.json", config_dict,
                   [train_split, test_split, Path(bench_data_root) / gene_list])

    return probe_results

        
def predict_folds(args, exp_save_dir, model_name, dataset_name, device, bench_data_root, slide_emb_root,fusion, custom_encoder):
    """ Predict all folds for a given model """
    split_dir = os.path.join(bench_data_root, 'splits')
    #if not os.path.exists(split_dir):
    #    raise FileNotFoundError(f"{split_dir} doesn't exist, make sure that you specified the ")
    splits = os.listdir(split_dir)
    n_splits = len(splits) // 2
    

    libprobe_results_arr = []
    for i in range(n_splits):
        train_split = os.path.join(split_dir, f'train_{i}.csv')
        test_split = os.path.join(split_dir, f'test_{i}.csv')
        kfold_save_dir = os.path.join(exp_save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        extract_tiles = True if i == 0 else False
        linprobe_results = predict_single_split(train_split, test_split, args, kfold_save_dir, dataset_name, model_name, device=device, bench_data_root=bench_data_root, custom_encoder=custom_encoder, extract_tiles=extract_tiles,
                                                slide_emb_root = slide_emb_root, fusion=fusion)
        libprobe_results_arr.append(linprobe_results)
        
        
    kfold_results = merge_fold_results(libprobe_results_arr)
    with open(os.path.join(exp_save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)
        
    return kfold_results


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


_BOOL_SETTINGS = ("normalize", "library_size_normalize")


def _canonical_config(config, path):
    """Accept `log_normalize` as the config-file name of `normalize`, and parse boolean settings."""
    config = dict(config)
    if "log_normalize" in config:
        value = config.pop("log_normalize")
        if "normalize" in config and coerce_bool(config["normalize"]) != coerce_bool(value):
            raise ValueError(
                f"Config file {path} sets both log_normalize={value!r} and normalize={config['normalize']!r}; "
                "keep only log_normalize."
            )
        config["normalize"] = value
    for key in _BOOL_SETTINGS:
        if key in config:
            config[key] = coerce_bool(config[key])
    return config


def _resolve_args(cli_args, kwargs):
    """Merge settings in order, each overriding the previous: defaults < CLI args < kwargs < config file."""
    args = Namespace(**asdict(BenchmarkConfig()))

    explicit = {}
    sources = [("CLI", vars(cli_args) if cli_args is not None else {}), ("kwargs", kwargs)]
    for source, values in sources:
        for k, v in values.items():
            if v is None:
                continue
            logger.info(f"Updating {k} with {v} ({source})")
            setattr(args, k, v)
            explicit[k] = (v, source)

    if args.config is not None:
        with open(args.config) as stream:
            config = _canonical_config(yaml.safe_load(stream), args.config)
        for key, value in config.items():
            if key not in args:
                logger.warning(f"Ignoring unknown key {key!r} in config file {args.config}")
                continue
            if key in explicit and explicit[key][0] != value:
                given, source = explicit[key]
                logger.warning(
                    f"{key}={given!r} was passed via {source}, but config file {args.config} "
                    f"sets {key}={value!r}. Following the config file."
                )
            setattr(args, key, value)
    return args


def benchmark(encoder: torch.nn.Module, enc_transf: Callable, precision: torch.dtype, cli_args: dict=None, **kwargs) -> Tuple[list, dict]:
    """ Benchmark a patch encoder on HEST-bench

    Args:
        encoder (torch.nn.Module): patch encoder to benchmark
        enc_transf (Callable): transformation applied to `encoder` during inference
        precision (torch.dtype): precision used by torch.cuda.amp.autocast() during inference for `encoder`
        cli_args (dict): cli_arguments. Defaults to None.
        **kwargs: lookup `BenchmarkConfig` for additional parameters

    """  

    args = _resolve_args(cli_args, kwargs)
    set_seed(args.seed)
    # Many DataLoader workers passing tensors can exhaust file descriptors with the default strategy.
    torch.multiprocessing.set_sharing_strategy('file_system')

    logger.info(f'Saving models to {args.weights_root}...')
    snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=args.weights_root, allow_patterns=['fm_v1/*'])
    
    logger.info(f'Fetch the bench data...')
    snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=args.bench_data_root, ignore_patterns=['fm_v1/*'])
    
    
    logger.info(f'Benchmarking on the following datasets {args.datasets}')
    
    logger.info(f'run parameters {args}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    datasets = args.datasets
    if len(datasets) >= 1 and datasets[0] == '*':
        datasets = os.listdir(get_path(args.bench_data_root))
    
    #### Setup Save Directory ####
    save_dir = get_path(args.results_dir)
    if args.exp_code is None:
        exp_code = f"run_{get_current_time()}"
    else:
        exp_code = args.exp_code + f"::{get_current_time()}"
    save_dir = os.path.join(save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)
    
    encoders = []
    weights_root = args.weights_root
    if encoder is not None:
        encoders.append('custom_encoder')
        custom_encoder = CustomInferenceEncoder(None, 'custom_encoder', encoder, enc_transf, precision)
    else:
        custom_encoder = None
        
    encoders += args.encoders

    slide_emb_root = args.slide_emb_root
    fusion = args.fusion
    
    dataset_perfs, perf_per_enc = benchmark_grid(args, device, encoders, datasets, save_dir=save_dir,  slide_emb_root=slide_emb_root, fusion=fusion, custom_encoder=custom_encoder)
    
    return dataset_perfs, perf_per_enc
    
    
if __name__ == '__main__':
    benchmark(None, None, None, _parse_args())
    
