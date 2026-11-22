import huggingface_hub
import pytest
from huggingface_hub.utils import EntryNotFoundError

from gutdecoder.bench.cpath_model_zoo.inference_models import _download_hf_timm_weights


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
