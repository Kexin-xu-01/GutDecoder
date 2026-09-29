"""
GPU-accelerated H&E stain normalisation via torchstain (tested with v1.3.0).

Import path: torchstain.base.normalizers
Tensor convention: input [C, H, W] uint8, moved to device before calling torchstain.

Whole-slide strategy:
  fit()       – reference is resized to _FIT_SIDE × _FIT_SIDE before fitting;
                stain matrix estimation does not need all pixels.
  transform() – slide is processed in _CHUNK × _CHUNK spatial tiles so that
                the per-tile lstsq stays within cuSOLVER matrix-size limits.

Available method names:
    macenko_gpu   — Macenko OD-space, torchstain torch backend
    reinhard_gpu  — Reinhard LAB transfer, torchstain torch backend

Install: python3.11 -m pip install --user torchstain
"""

from __future__ import annotations

import numpy as np
import torch

# Reference is downsampled to this side length before stain matrix fitting.
# cuSOLVER lstsq fails on very large N (whole-slide millions of pixels).
_FIT_SIDE = 1024   # 1024×1024 = ~1M pixels – sufficient for Macenko PCA

# Tile size for chunked normalize().  Each tile is at most _CHUNK×_CHUNK pixels.
_CHUNK = 2048


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 numpy → [C, H, W] uint8 tensor on device."""
    return torch.from_numpy(image).permute(2, 0, 1).to(device)


def _resize_for_fit(image: np.ndarray, side: int = _FIT_SIDE) -> np.ndarray:
    """Resize image so its longest side equals `side` (preserving aspect ratio)."""
    H, W = image.shape[:2]
    if max(H, W) <= side:
        return image
    from PIL import Image as PILImage
    scale = side / max(H, W)
    new_w, new_h = max(1, int(W * scale)), max(1, int(H * scale))
    return np.array(PILImage.fromarray(image).resize((new_w, new_h), PILImage.LANCZOS))


# ---------------------------------------------------------------------------
# Macenko (torchstain torch backend)
# fit  → torchstain.TorchMacenkoNormalizer.fit()   expects [C,H,W] uint8
# norm → returns (Inorm, H, E) where Inorm is (H,W,C) int tensor
# ---------------------------------------------------------------------------

class TorchMacenkoNormalizer:
    def __init__(self, device: str | torch.device = "auto") -> None:
        self._device = _resolve_device(device)
        self._norm = None

    def fit(self, reference: np.ndarray) -> "TorchMacenkoNormalizer":
        try:
            from torchstain.base.normalizers import MacenkoNormalizer
        except ImportError as e:
            raise ImportError(
                "torchstain is required for GPU normalisation. "
                "Install with: python3.11 -m pip install --user torchstain"
            ) from e

        self._norm = MacenkoNormalizer(backend="torch")
        # Downsample: stain matrix estimation needs representative pixels,
        # not every pixel; this also keeps cuSOLVER lstsq within its size limits.
        ref_small = _resize_for_fit(reference, _FIT_SIDE)
        self._norm.fit(_to_tensor(ref_small, self._device))
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        H, W, _ = image.shape
        # Process in tiles to keep per-tile N within cuSOLVER limits.
        if H <= _CHUNK and W <= _CHUNK:
            return self._normalize_tile(image)

        result = np.empty_like(image)
        for y0 in range(0, H, _CHUNK):
            for x0 in range(0, W, _CHUNK):
                tile = image[y0:y0 + _CHUNK, x0:x0 + _CHUNK]
                result[y0:y0 + _CHUNK, x0:x0 + _CHUNK] = self._normalize_tile(tile)
        return result

    def _normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        inorm, _, _ = self._norm.normalize(
            I=_to_tensor(tile, self._device), stains=False
        )
        # torchstain returns (H, W, C) int tensor
        return inorm.clamp(0, 255).byte().cpu().numpy()


# ---------------------------------------------------------------------------
# Reinhard (torchstain torch backend)
# fit  → TorchReinhardNormalizer.fit()    expects [C,H,W] uint8
# norm → returns [C,H,W] uint8 tensor (bare, no tuple)
# ---------------------------------------------------------------------------

class TorchReinhardNormalizer:
    def __init__(self, device: str | torch.device = "auto") -> None:
        self._device = _resolve_device(device)
        self._norm = None

    def fit(self, reference: np.ndarray) -> "TorchReinhardNormalizer":
        try:
            from torchstain.base.normalizers import ReinhardNormalizer
        except ImportError as e:
            raise ImportError(
                "torchstain is required for GPU normalisation. "
                "Install with: python3.11 -m pip install --user torchstain"
            ) from e

        self._norm = ReinhardNormalizer(backend="torch")
        # Reinhard only needs per-channel mean/std — tiny summary statistics.
        # Downsampling is cheap and avoids any potential size issues.
        ref_small = _resize_for_fit(reference, _FIT_SIDE)
        self._norm.fit(_to_tensor(ref_small, self._device))
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        H, W, _ = image.shape
        if H <= _CHUNK and W <= _CHUNK:
            return self._normalize_tile(image)

        result = np.empty_like(image)
        for y0 in range(0, H, _CHUNK):
            for x0 in range(0, W, _CHUNK):
                tile = image[y0:y0 + _CHUNK, x0:x0 + _CHUNK]
                result[y0:y0 + _CHUNK, x0:x0 + _CHUNK] = self._normalize_tile(tile)
        return result

    def _normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        # torchstain returns [C, H, W] uint8 — permute to (H, W, C)
        out = self._norm.normalize(I=_to_tensor(tile, self._device))
        return out.permute(1, 2, 0).cpu().numpy()
