"""
Classical H&E stain normalisation: Reinhard, Macenko, Vahadane.

Follows the approach of:
  multi-lab-stain-normalization-benchmarking (Zenodo 12344369)

Key utilities (mirrors stain_utils.py from the original):
  standardize_brightness  — scale so 90th-percentile pixel = 255
  RGB_to_OD / OD_to_RGB   — optical density conversion
  normalize_rows           — L2-normalise rows of a matrix
  notwhite_mask            — LAB L-channel tissue mask (Vahadane)
  get_concentrations       — non-negative lstsq (replaces SPAMS lasso)

All normalisers expose:   fit(reference: np.ndarray)
                          transform(image: np.ndarray) -> np.ndarray
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Utility functions (from stain_utils.py)
# ---------------------------------------------------------------------------

def standardize_brightness(I: np.ndarray) -> np.ndarray:
    """Scale image so the 90th-percentile pixel value equals 255."""
    p = np.percentile(I, 90)
    if p == 0:
        return I
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def _remove_zeros(I: np.ndarray) -> np.ndarray:
    """Replace zero pixels with 1 to avoid log(0)."""
    mask = (I == 0)
    I = I.copy()
    I[mask] = 1
    return I


def RGB_to_OD(I: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 → optical density.  OD = -log(I/255)."""
    I = _remove_zeros(I)
    return -np.log(I.astype(np.float64) / 255.0)


def OD_to_RGB(OD: np.ndarray) -> np.ndarray:
    """Convert optical density → RGB uint8."""
    return (255 * np.exp(-OD)).astype(np.uint8)


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """L2-normalise each row of A."""
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def notwhite_mask(I: np.ndarray, thresh: float = 0.8) -> np.ndarray:
    """
    Boolean mask: True where tissue is present (not white background).
    Uses LAB L-channel: tissue pixels have L < thresh.
    """
    import cv2 as cv
    I_lab = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_lab[:, :, 0] / 255.0
    return L < thresh


def get_concentrations(
    I: np.ndarray,
    stain_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel stain concentrations (N_pixels × 2).

    Replaces SPAMS lasso (original) with non-negative least squares:
      min  ||stain_matrix.T @ c - OD||   s.t.  c >= 0
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    conc, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)
    return np.clip(conc.T, 0, None)   # (N, 2), non-negative


# ---------------------------------------------------------------------------
# Reinhard (2001) — LAB colour statistics transfer
# ---------------------------------------------------------------------------

def _lab_split(I: np.ndarray):
    """RGB uint8 → split LAB channels with original paper's scaling."""
    import cv2 as cv
    I = cv.cvtColor(I, cv.COLOR_RGB2LAB).astype(np.float64)
    I1, I2, I3 = cv.split(I)
    I1 /= 2.55       # [0, 100]
    I2 -= 128.0      # [−128, 127]
    I3 -= 128.0
    return I1, I2, I3


def _merge_back(I1, I2, I3) -> np.ndarray:
    """Scaled LAB channels → RGB uint8."""
    import cv2 as cv
    I1 *= 2.55
    I2 += 128.0
    I3 += 128.0
    I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    return cv.cvtColor(I, cv.COLOR_LAB2RGB)


def _get_mean_std(I: np.ndarray):
    import cv2 as cv
    I1, I2, I3 = _lab_split(I)
    means = [cv.meanStdDev(ch)[0] for ch in (I1, I2, I3)]
    stds  = [cv.meanStdDev(ch)[1] for ch in (I1, I2, I3)]
    return means, stds


class ReinhardNormalizer:
    """
    Colour statistics transfer in CIE LAB space.

    Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001.
    Follows original stainnorm_reinhard.py: cv2 LAB, L/2.55, AB−128 scaling.
    """

    def __init__(self) -> None:
        self.target_means = None
        self.target_stds = None

    def fit(self, reference: np.ndarray) -> "ReinhardNormalizer":
        reference = standardize_brightness(reference)
        self.target_means, self.target_stds = _get_mean_std(reference)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        image = standardize_brightness(image)
        I1, I2, I3 = _lab_split(image)
        means, stds = _get_mean_std(image)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return _merge_back(norm1, norm2, norm3)


# ---------------------------------------------------------------------------
# Macenko (2009) — SVD stain matrix in OD space
# ---------------------------------------------------------------------------

def _get_stain_matrix_macenko(I: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
    """
    Estimate 2×3 H&E stain matrix via Macenko PCA.
    Follows original stainnorm_macenko.py exactly.
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    # Keep pixels where ANY channel has OD > beta (tissue; not bright background)
    OD = OD[(OD > beta).any(axis=1)]

    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]           # two largest eigenvectors (eigh returns ascending)
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    That = OD @ V
    phi = np.arctan2(That[:, 1], That[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v1 = V @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = V @ np.array([np.cos(max_phi), np.sin(max_phi)])

    HE = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
    return normalize_rows(HE)


class MacenkoNormalizer:
    """
    SVD-based stain matrix estimation in optical density space.

    Macenko et al., "A Method for Normalizing Histology Slides for
    Quantitative Analysis", ISBI 2009.
    Follows original stainnorm_macenko.py.
    """

    def __init__(self, beta: float = 0.15, alpha: float = 1.0) -> None:
        self.beta = beta
        self.alpha = alpha
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, reference: np.ndarray) -> "MacenkoNormalizer":
        reference = standardize_brightness(reference)
        self.stain_matrix_target = _get_stain_matrix_macenko(reference, self.beta, self.alpha)
        self.target_concentrations = get_concentrations(reference, self.stain_matrix_target)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        image = standardize_brightness(image)
        stain_matrix_source = _get_stain_matrix_macenko(image, self.beta, self.alpha)
        source_concentrations = get_concentrations(image, stain_matrix_source)

        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (maxC_target / maxC_source)

        return (255 * np.exp(
            -source_concentrations @ self.stain_matrix_target
        ).reshape(image.shape)).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Vahadane (2016) — NMF stain matrix (replaces SPAMS trainDL)
# ---------------------------------------------------------------------------

def _get_stain_matrix_vahadane(I: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Estimate 2×3 H&E stain matrix via NMF on tissue pixels.
    Follows original stainnorm_vahadane.py; sklearn NMF replaces spams.trainDL.
    Tissue selection uses notwhite_mask (LAB L<thresh), matching the original.
    """
    from sklearn.decomposition import NMF

    mask = notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]

    model = NMF(n_components=2, init="nndsvd", max_iter=1000, random_state=42)
    model.fit(OD)
    dictionary = model.components_   # (2, 3)

    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    return normalize_rows(dictionary)


class VahadaneNormalizer:
    """
    Structure-preserving stain normalisation via sparse NMF.

    Vahadane et al., "Structure-Preserving Color Normalization and Sparse
    Stain Separation for Histological Images", TMI 2016.
    Follows original stainnorm_vahadane.py; sklearn NMF replaces spams.trainDL.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.stain_matrix_target = None

    def fit(self, reference: np.ndarray) -> "VahadaneNormalizer":
        reference = standardize_brightness(reference)
        self.stain_matrix_target = _get_stain_matrix_vahadane(reference, self.threshold)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        image = standardize_brightness(image)
        stain_matrix_source = _get_stain_matrix_vahadane(image, self.threshold)
        source_concentrations = get_concentrations(image, stain_matrix_source)

        return (255 * np.exp(
            -source_concentrations @ self.stain_matrix_target
        ).reshape(image.shape)).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_METHODS: dict[str, type] = {
    "reinhard": ReinhardNormalizer,
    "macenko":  MacenkoNormalizer,
    "vahadane": VahadaneNormalizer,
}

# GPU methods — loaded lazily from _normalizers_torch to avoid hard torch import
_GPU_METHODS = {"macenko_gpu", "reinhard_gpu"}

#: All supported normalisation methods (CPU + GPU).
AVAILABLE_METHODS = sorted(_METHODS) + sorted(_GPU_METHODS)


def get_normalizer(method: str, device: str = "auto"):
    """
    Instantiate a normaliser by name.

    Args:
        method: 'reinhard', 'macenko', 'vahadane' (CPU numpy),
                or 'reinhard_gpu', 'macenko_gpu' (GPU via torchstain).
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
            TorchReinhardNormalizer,
        )
        _torch_map = {
            "macenko_gpu":  TorchMacenkoNormalizer,
            "reinhard_gpu": TorchReinhardNormalizer,
        }
        return _torch_map[key](device=device)

    if key not in _METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available: {AVAILABLE_METHODS}"
        )
    return _METHODS[key]()
