import huggingface_hub
import pytest
import torch
from huggingface_hub.utils import EntryNotFoundError

from gutdecoder.bench.cpath_model_zoo.inference_models import (
    _cls_or_cls_mean_patch,
    _download_hf_timm_weights,
    inf_encoder_factory,
)


def test_download_falls_back_to_bin_when_safetensors_absent(monkeypatch):
    def fake_download(repo_id, filename):
        if filename == "model.safetensors":
            raise EntryNotFoundError("no safetensors")
        return f"/cache/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert _download_hf_timm_weights("org/model") == "/cache/pytorch_model.bin"


def test_download_surfaces_access_errors(monkeypatch):
    def fake_download(repo_id, filename):
        raise PermissionError("401: gated repo, request access")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    # Used to be swallowed and reported as "could not download".
    with pytest.raises(PermissionError, match="gated repo"):
        _download_hf_timm_weights("org/model")


def test_cls_or_cls_mean_patch_skips_register_tokens():
    # batch 1, CLS + 4 registers + 2 patches, dim 2
    tokens = torch.tensor([[[1.0, 1.0], [9, 9], [9, 9], [9, 9], [9, 9], [2, 4], [4, 8]]])

    assert _cls_or_cls_mean_patch(tokens, return_cls=True, n_prefix_tokens=5).tolist() == [[1.0, 1.0]]
    assert _cls_or_cls_mean_patch(tokens, return_cls=False, n_prefix_tokens=5).tolist() == [[1.0, 1.0, 3.0, 6.0]]


def test_ctranspath_and_remedis_use_the_trident_adapter():
    for name in ("ctranspath", "remedis"):
        cls = inf_encoder_factory(name)
        assert cls is inf_encoder_factory(name)  # one class, not a new one per call
        assert cls.__name__ == f"TridentEncoder_{name}"
