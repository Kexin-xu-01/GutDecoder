"""
Tests for the HEST 1.1.2 / TRIDENT migration.

These tests verify imports, utility functions, and the encoder factory without
downloading any model weights (no network calls beyond import-time checks).
Run with: pytest tests/test_migration.py -v
"""
import os
import numpy as np
import pytest
import h5py
import torch


# ---------------------------------------------------------------------------
# 1. Import smoke tests
# ---------------------------------------------------------------------------

def test_import_inference_models():
    from gutdecoder.bench.cpath_model_zoo.inference_models import (
        inf_encoder_factory,
        InferenceEncoder,
        _get_eval_transforms,
        _IMAGENET_MEAN,
        _IMAGENET_STD,
        _make_trident_encoder_class,
        _TRIDENT_NEW_MODELS,
    )


def test_import_benchmark():
    from gutdecoder.bench.benchmark import (
        get_path,
        get_bench_weights,
        BenchmarkConfig,
        benchmark_grid,
        embed_tiles,
    )


def test_import_h5patchdataset():
    from hest.bench.st_dataset import H5PatchDataset


def test_import_hest_utils():
    from hest.bench.utils.file_utils import read_assets_from_h5, save_hdf5, save_pkl
    from hest.bench.utils.utils import merge_dict, get_current_time


def test_no_hestcore_import():
    """hestcore is no longer a dependency — confirm it's not imported by our modules."""
    import sys
    import importlib
    # Force fresh import
    mods_to_check = [
        'gutdecoder.bench.benchmark',
        'gutdecoder.bench.cpath_model_zoo.inference_models',
        'gutdecoder.wrappers.segment_wrapper',
    ]
    for mod in mods_to_check:
        if mod in sys.modules:
            del sys.modules[mod]

    import gutdecoder.bench.benchmark
    import gutdecoder.bench.cpath_model_zoo.inference_models
    assert 'hestcore' not in sys.modules, "hestcore should not be imported"


# ---------------------------------------------------------------------------
# 2. _get_eval_transforms
# ---------------------------------------------------------------------------

def test_get_eval_transforms_returns_compose():
    from torchvision import transforms
    from gutdecoder.bench.cpath_model_zoo.inference_models import _get_eval_transforms
    t = _get_eval_transforms((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert isinstance(t, transforms.Compose)
    assert len(t.transforms) == 4


def test_get_eval_transforms_custom_size():
    from torchvision.transforms import Resize
    from gutdecoder.bench.cpath_model_zoo.inference_models import _get_eval_transforms
    t = _get_eval_transforms((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), target_img_size=448)
    resize_step = t.transforms[0]
    assert isinstance(resize_step, Resize)
    assert resize_step.size == 448


def test_get_eval_transforms_produces_tensor():
    from PIL import Image
    from gutdecoder.bench.cpath_model_zoo.inference_models import (
        _get_eval_transforms, _IMAGENET_MEAN, _IMAGENET_STD,
    )
    t = _get_eval_transforms(_IMAGENET_MEAN, _IMAGENET_STD)
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    out = t(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# 3. inf_encoder_factory — all names resolve to a class, no weight downloads
# ---------------------------------------------------------------------------

LEGACY_MODELS = [
    'conch_v1', 'conch_v1_5', 'uni_v1', 'uni_v2',
    'phikon', 'phikon_v2', 'h0_mini',
    'hibou_large', 'kaiko_base_8',
    'resnet50', 'gigapath', 'virchow', 'virchow2',
    'hoptimus0', 'hoptimus1',
]

TRIDENT_DELEGATED = [
    'ctranspath', 'remedis',
]

TRIDENT_NEW = [
    'kaiko-vitb16', 'kaiko-vits8', 'kaiko-vits16', 'kaiko-vitl14',
    'lunit-vits8', 'gpfm', 'musk', 'midnight12k', 'openmidnight', 'genbio-pathfm',
]

ALIASES = [
    'hibou_l', 'hibou-l', 'kaiko-vitb8',
]

ALL_MODELS = LEGACY_MODELS + TRIDENT_DELEGATED + TRIDENT_NEW + ALIASES


@pytest.mark.parametrize("enc_name", ALL_MODELS)
def test_inf_encoder_factory_returns_class(enc_name):
    from gutdecoder.bench.cpath_model_zoo.inference_models import inf_encoder_factory, InferenceEncoder
    cls = inf_encoder_factory(enc_name)
    assert cls is not None
    assert isinstance(cls, type), f"{enc_name} should return a class, got {type(cls)}"
    assert issubclass(cls, (InferenceEncoder, torch.nn.Module)), \
        f"{enc_name} class should be a torch.nn.Module subclass"


def test_inf_encoder_factory_unknown_raises():
    from gutdecoder.bench.cpath_model_zoo.inference_models import inf_encoder_factory
    with pytest.raises(ValueError, match="Unknown encoder name"):
        inf_encoder_factory('not_a_real_model_xyz')


def test_aliases_point_to_same_class_as_originals():
    from gutdecoder.bench.cpath_model_zoo.inference_models import inf_encoder_factory
    assert inf_encoder_factory('hibou_l') is inf_encoder_factory('hibou_large')
    assert inf_encoder_factory('hibou-l') is inf_encoder_factory('hibou_large')
    assert inf_encoder_factory('kaiko-vitb8') is inf_encoder_factory('kaiko_base_8')


# ---------------------------------------------------------------------------
# 4. _make_trident_encoder_class
# ---------------------------------------------------------------------------

def test_make_trident_encoder_class_is_module_subclass():
    from gutdecoder.bench.cpath_model_zoo.inference_models import (
        _make_trident_encoder_class, InferenceEncoder,
    )
    cls = _make_trident_encoder_class('gpfm')
    assert issubclass(cls, InferenceEncoder)
    assert issubclass(cls, torch.nn.Module)


def test_make_trident_encoder_class_different_per_name():
    from gutdecoder.bench.cpath_model_zoo.inference_models import _make_trident_encoder_class
    cls_gpfm = _make_trident_encoder_class('gpfm')
    cls_musk = _make_trident_encoder_class('musk')
    assert cls_gpfm is not cls_musk


def test_trident_new_models_use_adapter():
    from gutdecoder.bench.cpath_model_zoo.inference_models import (
        inf_encoder_factory, InferenceEncoder,
        _TRIDENT_NEW_MODELS,
    )
    for name in _TRIDENT_NEW_MODELS:
        cls = inf_encoder_factory(name)
        assert issubclass(cls, InferenceEncoder), f"{name} should be InferenceEncoder subclass"
        assert 'Trident' in cls.__name__, f"{name} class name should indicate TRIDENT adapter"


# ---------------------------------------------------------------------------
# 5. get_path and get_bench_weights
# ---------------------------------------------------------------------------

def test_get_path_absolute():
    from gutdecoder.bench.benchmark import get_path
    result = get_path('/absolute/path/to/data')
    assert result == '/absolute/path/to/data'


def test_get_path_relative_resolves():
    from gutdecoder.bench.benchmark import get_path
    result = get_path('./some/relative/path')
    assert not result.startswith('./')
    assert result.endswith('/some/relative/path')
    assert os.path.isabs(result)


def test_get_bench_weights_hf_model(tmp_path):
    from gutdecoder.bench.benchmark import get_bench_weights
    # HF models have empty string in local_ckpts.json — should not raise
    result = get_bench_weights(str(tmp_path), 'resnet50')
    assert result is not None


def test_get_bench_weights_unknown_raises(tmp_path):
    from gutdecoder.bench.benchmark import get_bench_weights
    with pytest.raises(ValueError):
        get_bench_weights(str(tmp_path), 'definitely_not_a_real_model_abc123')


def test_get_bench_weights_all_registered_models(tmp_path):
    from gutdecoder.bench.benchmark import get_bench_weights
    for name in ALL_MODELS:
        if name in ('hibou-l',):
            continue  # only 'hibou_l' is in local_ckpts.json, not 'hibou-l'
        try:
            get_bench_weights(str(tmp_path), name)
        except ValueError as e:
            pytest.fail(f"get_bench_weights raised for registered model '{name}': {e}")


# ---------------------------------------------------------------------------
# 6. H5PatchDataset
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_h5_file(tmp_path):
    h5_path = tmp_path / "patches.h5"
    n = 12
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('img', data=np.random.randint(0, 255, (n, 224, 224, 3), dtype=np.uint8))
        f.create_dataset('coords', data=np.random.randint(0, 10000, (n, 2)))
        barcodes = np.array([f'AAACCTGAGCTTCGCG-{i}'.encode() for i in range(n)])
        f.create_dataset('barcodes', data=barcodes)
    return str(h5_path)


def test_h5patchdataset_length(mock_h5_file):
    from hest.bench.st_dataset import H5PatchDataset
    ds = H5PatchDataset(mock_h5_file)
    assert len(ds) == 12


def test_h5patchdataset_item_keys(mock_h5_file):
    from hest.bench.st_dataset import H5PatchDataset
    ds = H5PatchDataset(mock_h5_file)
    item = ds[0]
    assert set(item.keys()) == {'imgs', 'coords', 'barcodes'}


def test_h5patchdataset_no_transform_returns_array(mock_h5_file):
    from hest.bench.st_dataset import H5PatchDataset
    ds = H5PatchDataset(mock_h5_file)
    item = ds[0]
    assert isinstance(item['imgs'], np.ndarray)
    assert item['imgs'].shape == (224, 224, 3)


def test_h5patchdataset_with_transform_returns_tensor(mock_h5_file):
    from hest.bench.st_dataset import H5PatchDataset
    from gutdecoder.bench.cpath_model_zoo.inference_models import (
        _get_eval_transforms, _IMAGENET_MEAN, _IMAGENET_STD,
    )
    t = _get_eval_transforms(_IMAGENET_MEAN, _IMAGENET_STD)
    ds = H5PatchDataset(mock_h5_file, img_transform=t)
    item = ds[0]
    assert isinstance(item['imgs'], torch.Tensor)
    assert item['imgs'].shape == (3, 224, 224)


def test_h5patchdataset_dataloader_batch(mock_h5_file):
    from torch.utils.data import DataLoader
    from hest.bench.st_dataset import H5PatchDataset
    from gutdecoder.bench.cpath_model_zoo.inference_models import (
        _get_eval_transforms, _IMAGENET_MEAN, _IMAGENET_STD,
    )
    t = _get_eval_transforms(_IMAGENET_MEAN, _IMAGENET_STD)
    ds = H5PatchDataset(mock_h5_file, img_transform=t)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch['imgs'].shape == (4, 3, 224, 224)
    assert batch['coords'].shape == (4, 2)
    assert len(batch['barcodes']) == 4


def test_h5patchdataset_barcode_key_fallback(tmp_path):
    """H5PatchDataset handles files that use 'barcode' (singular) key."""
    from hest.bench.st_dataset import H5PatchDataset
    h5_path = str(tmp_path / "patches_singular.h5")
    n = 5
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('img', data=np.random.randint(0, 255, (n, 64, 64, 3), dtype=np.uint8))
        f.create_dataset('coords', data=np.zeros((n, 2), dtype=int))
        f.create_dataset('barcode', data=np.array([b'BC001'] * n))
    ds = H5PatchDataset(h5_path)
    item = ds[0]
    assert 'barcodes' in item
