"""
GPU-accelerated H&E stain normalisation via torchstain.

These wrappers expose the same fit()/transform() interface as the NumPy
normalizers so they are interchangeable in the pipeline.

Available method names (passed to get_normalizer):
    macenko_gpu   — Macenko in OD space, torchstain backend
    vahadane_gpu  — Vahadane sparse NMF, torchstain backend
    reinhard_gpu  — Reinhard LAB transfer, torchstain backend

Install dependency:
    pip install torchstain
"""

from __future__ import annotations

import numpy as np
import torch


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32 [0, 255] tensor on device."""
    return (
        torch.from_numpy(image.astype(np.float32))
        .permute(2, 0, 1)
        .to(device)
    )


def _from_tensor(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor → (H, W, 3) uint8 numpy."""
    return t.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()


class _TorchStainNormalizer:
    """fit()/transform() wrapper around a torchstain normaliser named by `_torchstain_name`."""

    _torchstain_name: str

    def __init__(self, device: str | torch.device = "auto") -> None:
        self._device = _resolve_device(device)
        self._norm = None

    def fit(self, reference: np.ndarray):
        try:
            import torchstain.normalizers
        except ImportError as e:
            raise ImportError(
                "torchstain is required for GPU normalisation. "
                "Install with: pip install torchstain"
            ) from e
        self._norm = getattr(torchstain.normalizers, self._torchstain_name)(backend="torch")
        self._norm.fit(_to_tensor(reference, self._device))
        return self

    def fit_source(self, sample: np.ndarray):
        # torchstain re-estimates source parameters inside every normalize() call and
        # offers no way to fix them, so tiles of one slide are normalised independently.
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self._norm is None:
            raise RuntimeError("Call fit() before transform().")
        result = self._norm.normalize(I=_to_tensor(image, self._device))
        norm = result[0] if isinstance(result, (tuple, list)) else result
        return _from_tensor(norm)


class TorchMacenkoNormalizer(_TorchStainNormalizer):
    _torchstain_name = "MacenkoNormalizer"


class TorchVahadaneNormalizer(_TorchStainNormalizer):
    _torchstain_name = "VahadaneNormalizer"


class TorchReinhardNormalizer(_TorchStainNormalizer):
    _torchstain_name = "ReinhardNormalizer"
