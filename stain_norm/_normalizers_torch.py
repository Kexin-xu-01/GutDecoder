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


# ---------------------------------------------------------------------------
# Macenko (torchstain)
# ---------------------------------------------------------------------------

class TorchMacenkoNormalizer:
    def __init__(self, device: str | torch.device = "auto") -> None:
        self._device = _resolve_device(device)
        self._norm = None

    def fit(self, reference: np.ndarray) -> "TorchMacenkoNormalizer":
        try:
            from torchstain.normalizers import MacenkoNormalizer
        except ImportError as e:
            raise ImportError(
                "torchstain is required for GPU normalisation. "
                "Install with: pip install torchstain"
            ) from e
        self._norm = MacenkoNormalizer(backend="torch")
        self._norm.fit(_to_tensor(reference, self._device))
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        norm, _, _ = self._norm.normalize(
            I=_to_tensor(image, self._device), stains=True
        )
        return _from_tensor(norm)


# ---------------------------------------------------------------------------
# Vahadane (torchstain)
# ---------------------------------------------------------------------------

class TorchVahadaneNormalizer:
    def __init__(self, device: str | torch.device = "auto") -> None:
        self._device = _resolve_device(device)
        self._norm = None

    def fit(self, reference: np.ndarray) -> "TorchVahadaneNormalizer":
        try:
            from torchstain.normalizers import VahadaneNormalizer
        except ImportError as e:
            raise ImportError(
                "torchstain is required for GPU normalisation. "
                "Install with: pip install torchstain"
            ) from e
        self._norm = VahadaneNormalizer(backend="torch")
        self._norm.fit(_to_tensor(reference, self._device))
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        norm, _, _ = self._norm.normalize(
            I=_to_tensor(image, self._device), stains=True
        )
        return _from_tensor(norm)


# ---------------------------------------------------------------------------
# Reinhard (torchstain)
# ---------------------------------------------------------------------------

class TorchReinhardNormalizer:
    def __init__(self, device: str | torch.device = "auto") -> None:
        self._device = _resolve_device(device)
        self._norm = None

    def fit(self, reference: np.ndarray) -> "TorchReinhardNormalizer":
        try:
            from torchstain.normalizers import ReinhardNormalizer
        except ImportError as e:
            raise ImportError(
                "torchstain is required for GPU normalisation. "
                "Install with: pip install torchstain"
            ) from e
        self._norm = ReinhardNormalizer(backend="torch")
        self._norm.fit(_to_tensor(reference, self._device))
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        result = self._norm.normalize(I=_to_tensor(image, self._device))
        norm = result[0] if isinstance(result, (tuple, list)) else result
        return _from_tensor(norm)


