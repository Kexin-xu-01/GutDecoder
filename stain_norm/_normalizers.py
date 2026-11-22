"""
Classical H&E stain normalisation algorithms.

Three methods are provided:

    Reinhard — colour statistics transfer in LAB space (fastest, ~0.5 s/slide).
    Macenko  — SVD-based stain matrix estimation in OD space (good balance).
    Vahadane — NMF-based sparse stain separation (highest quality, ~30 s/slide).

All classes expose the same interface:
    normalizer.fit(reference_image)    # reference: (H,W,3) uint8 RGB
    normalizer.fit_source(sample)      # pixels representative of the slide to normalise
    normalizer.transform(image)        # image: (H,W,3) uint8 RGB → (H,W,3) uint8 RGB

transform() applies fixed parameters pixel by pixel, so normalising a slide whole
or tile by tile gives the same result.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab, lab2rgb


# ---------------------------------------------------------------------------
# Reinhard (2001)
# ---------------------------------------------------------------------------

def _lab_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LAB mean and std of tissue pixels, excluding white background."""
    lab = rgb2lab(rgb)
    mask = np.mean(rgb, axis=-1) < 235
    # A sample that is almost all background has no tissue to measure; use every pixel.
    v = lab[mask] if mask.sum() >= 100 else lab.reshape(-1, 3)
    return v.mean(axis=0), v.std(axis=0).clip(1e-4)


class ReinhardNormalizer:
    """
    Colour statistics transfer in CIE LAB space.

    Reference:
        Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001.
    """

    def __init__(self) -> None:
        self._target = None
        self._source = None

    def fit(self, reference: np.ndarray) -> "ReinhardNormalizer":
        self._target = _lab_stats(reference)
        return self

    def fit_source(self, sample: np.ndarray) -> "ReinhardNormalizer":
        self._source = _lab_stats(sample)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self._target is None or self._source is None:
            raise RuntimeError("Call fit() and fit_source() before transform().")
        (mu_s, sd_s), (mu_t, sd_t) = self._source, self._target
        lab = (rgb2lab(image) - mu_s) * (sd_t / sd_s) + mu_t
        lab[..., 0] = lab[..., 0].clip(0, 100)
        out = (lab2rgb(lab).clip(0, 1) * 255).astype(np.uint8)
        # Keep near-white background exactly to avoid a tinted canvas.
        background = image.mean(axis=-1) >= 245
        out[background] = image[background]
        return out


# ---------------------------------------------------------------------------
# Macenko (2009) and Vahadane (2016): stain separation in optical density space
# ---------------------------------------------------------------------------

def _rgb2od(image: np.ndarray) -> np.ndarray:
    return -np.log((image.astype(np.float64) + 1) / 256.0)


class _StainNormalizer:
    """Shared fit/transform for stain-matrix methods; subclasses define _get_stain_matrix."""

    def __init__(self, beta: float = 0.15) -> None:
        self.beta = beta
        self._target = None
        self._source = None

    def _tissue_od(self, image: np.ndarray) -> np.ndarray:
        """OD of tissue pixels: above beta (not background) and below 3.0 (not pure-black masks)."""
        OD_flat = _rgb2od(image).reshape(-1, 3)
        OD_hat = OD_flat[np.all(OD_flat > self.beta, axis=1) & np.all(OD_flat < 3.0, axis=1)]
        # Faintly stained samples: relax the threshold before giving up on the mask.
        if OD_hat.shape[0] < 10:
            OD_hat = OD_flat[np.any(OD_flat > 0.05, axis=1) & np.all(OD_flat < 3.0, axis=1)]
        if OD_hat.shape[0] < 10:
            OD_hat = OD_flat[np.any(OD_flat > 0.05, axis=1)]
        return OD_hat

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _get_concentrations(image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        OD_flat = np.clip(_rgb2od(image).reshape(-1, 3), 0, 3.0)  # cap masked-black pixels
        # Solve stain_matrix.T @ c = OD for each pixel
        conc, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD_flat.T, rcond=None)
        return conc.T  # (N, 2)

    def _stain_params(self, image: np.ndarray):
        stain_matrix = self._get_stain_matrix(image)
        maxC = np.percentile(self._get_concentrations(image, stain_matrix), 99, axis=0)
        return stain_matrix, maxC

    def fit(self, reference: np.ndarray):
        self._target = self._stain_params(reference)
        return self

    def fit_source(self, sample: np.ndarray):
        self._source = self._stain_params(sample)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self._target is None or self._source is None:
            raise RuntimeError("Call fit() and fit_source() before transform().")
        (stain_src, maxC_src), (stain_tgt, maxC_tgt) = self._source, self._target
        H, W, _ = image.shape
        conc = self._get_concentrations(image, stain_src)
        conc_norm = conc / (maxC_src + 1e-6) * maxC_tgt
        I_norm = 255.0 * np.exp(-(conc_norm @ stain_tgt))
        return I_norm.clip(0, 255).astype(np.uint8).reshape(H, W, 3)


class MacenkoNormalizer(_StainNormalizer):
    """
    SVD-based stain matrix estimation in optical density space.

    Reference:
        Macenko et al., "A Method for Normalizing Histology Slides for
        Quantitative Analysis", ISBI 2009.
    """

    def __init__(self, beta: float = 0.15, alpha: float = 1.0) -> None:
        super().__init__(beta)
        self.alpha = alpha

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        OD_hat = self._tissue_od(image)

        # PCA: two largest eigenvectors of the covariance matrix
        _, V = np.linalg.eigh(OD_hat.T @ OD_hat)
        V = V[:, -2:]  # 3 x 2  (columns = eigenvectors)

        # Project pixels onto the plane spanned by V and find the angular extremes
        That = OD_hat @ V
        phi = np.arctan2(That[:, 1], That[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100.0 - self.alpha)

        v1 = V @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = V @ np.array([np.cos(max_phi), np.sin(max_phi)])

        # H stain absorbs more red light → higher OD in channel 0
        return np.array([v1, v2]) if v1[0] >= v2[0] else np.array([v2, v1])


class VahadaneNormalizer(_StainNormalizer):
    """
    Structure-preserving stain normalisation via sparse NMF.

    Reference:
        Vahadane et al., "Structure-Preserving Color Normalization and Sparse
        Stain Separation for Histological Images", TMI 2016.

    Uses scikit-learn NMF as a widely-available substitute for SPAMS.
    """

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import NMF

        model = NMF(n_components=2, init="nndsvd", max_iter=500, random_state=42)
        model.fit(self._tissue_od(image))
        W = model.components_  # (2, 3)
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-6)

        # H stain first (highest OD in the red channel)
        return W[[1, 0]] if W[0, 0] < W[1, 0] else W


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_METHODS: dict[str, type] = {
    "reinhard": ReinhardNormalizer,
    "macenko": MacenkoNormalizer,
    "vahadane": VahadaneNormalizer,
}

# GPU methods — loaded lazily from _normalizers_torch to avoid hard torch import
_GPU_METHODS = {"macenko_gpu", "vahadane_gpu", "reinhard_gpu"}

#: All supported normalisation methods (CPU + GPU).
AVAILABLE_METHODS = sorted(_METHODS) + sorted(_GPU_METHODS)


def get_normalizer(method: str, device: str = "auto"):
    """
    Instantiate a normaliser by name.

    Args:
        method: Normalisation algorithm. CPU options: 'reinhard', 'macenko',
                'vahadane'. GPU options (requires torchstain + PyTorch):
                'macenko_gpu', 'vahadane_gpu', 'reinhard_gpu'.
        device: PyTorch device for GPU methods — 'auto' (default, uses CUDA if
                available), 'cpu', 'cuda', 'cuda:0', etc.
                Ignored for CPU numpy methods.

    Returns:
        An unfitted normaliser with fit(reference) / transform(image) methods.
    """
    key = method.lower()

    if key in _GPU_METHODS:
        from gutdecoder.stain_norm._normalizers_torch import (
            TorchMacenkoNormalizer,
            TorchVahadaneNormalizer,
            TorchReinhardNormalizer,
        )
        _torch_map = {
            "macenko_gpu": TorchMacenkoNormalizer,
            "vahadane_gpu": TorchVahadaneNormalizer,
            "reinhard_gpu": TorchReinhardNormalizer,
        }
        return _torch_map[key](device=device)

    if key not in _METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available: {AVAILABLE_METHODS}"
        )
    return _METHODS[key]()
