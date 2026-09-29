"""
Classical H&E stain normalisation algorithms.

Three methods are provided:

    Reinhard — colour statistics transfer in LAB space (fastest, ~0.5 s/slide).
    Macenko  — SVD-based stain matrix estimation in OD space (good balance).
    Vahadane — NMF-based sparse stain separation (highest quality, ~30 s/slide).

All classes expose the same interface:
    normalizer.fit(reference_image)    # reference: (H,W,3) uint8 RGB
    normalizer.transform(image)        # image: (H,W,3) uint8 RGB → (H,W,3) uint8 RGB
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab, lab2rgb


# ---------------------------------------------------------------------------
# Reinhard (2001)
# ---------------------------------------------------------------------------

class ReinhardNormalizer:
    """
    Colour statistics transfer in CIE LAB space.

    Reference:
        Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001.
    """

    def __init__(self) -> None:
        self._target_mean: np.ndarray | None = None
        self._target_std: np.ndarray | None = None

    def fit(self, reference: np.ndarray) -> "ReinhardNormalizer":
        """Compute LAB statistics from the reference image."""
        lab = rgb2lab(reference)
        self._target_mean = lab.mean(axis=(0, 1))
        self._target_std = lab.std(axis=(0, 1))
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transfer reference colour statistics onto image."""
        if self._target_mean is None:
            raise RuntimeError("Call fit() before transform().")
        lab = rgb2lab(image)
        src_mean = lab.mean(axis=(0, 1))
        src_std = lab.std(axis=(0, 1))

        lab_norm = (lab - src_mean) / (src_std + 1e-6) * self._target_std + self._target_mean

        # Clamp to valid LAB range
        lab_norm[:, :, 0] = np.clip(lab_norm[:, :, 0], 0, 100)
        lab_norm[:, :, 1:] = np.clip(lab_norm[:, :, 1:], -128, 127)

        rgb = lab2rgb(lab_norm)
        return (rgb * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Macenko (2009)
# ---------------------------------------------------------------------------

class MacenkoNormalizer:
    """
    SVD-based stain matrix estimation in optical density space.

    Reference:
        Macenko et al., "A Method for Normalizing Histology Slides for
        Quantitative Analysis", ISBI 2009.
    """

    def __init__(self, beta: float = 0.15, alpha: float = 1.0) -> None:
        self.beta = beta
        self.alpha = alpha
        self._stain_matrix_target: np.ndarray | None = None
        self._maxC_target: np.ndarray | None = None

    @staticmethod
    def _rgb2od(image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        return -np.log((img + 1) / 256.0)

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        OD = self._rgb2od(image)
        OD_flat = OD.reshape(-1, 3)

        # Keep tissue pixels: above beta (not bright background) and below 3.0
        # (not pure-black mask pixels, which have OD ≈ 5.5)
        mask = np.all(OD_flat > self.beta, axis=1) & np.all(OD_flat < 3.0, axis=1)
        OD_hat = OD_flat[mask]
        if OD_hat.shape[0] < 10:
            OD_hat = OD_flat[np.any(OD_flat > 0.05, axis=1) & np.all(OD_flat < 3.0, axis=1)]
        if OD_hat.shape[0] < 10:
            OD_hat = OD_flat[np.any(OD_flat > 0.05, axis=1)]

        # PCA: two largest eigenvectors of the covariance matrix
        cov = OD_hat.T @ OD_hat
        _, V = np.linalg.eigh(cov)
        V = V[:, -2:]  # 3 x 2  (columns = eigenvectors)

        # Project pixels onto the plane spanned by V
        That = OD_hat @ V  # N x 2

        # Find the angular extremes (alpha-percentile)
        phi = np.arctan2(That[:, 1], That[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100.0 - self.alpha)

        v1 = V @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = V @ np.array([np.cos(max_phi), np.sin(max_phi)])

        # H stain absorbs more red light → higher OD in channel 0
        stain_matrix = np.array([v1, v2]) if v1[0] >= v2[0] else np.array([v2, v1])
        return stain_matrix  # (2, 3)

    def _get_concentrations(
        self, image: np.ndarray, stain_matrix: np.ndarray
    ) -> np.ndarray:
        OD = self._rgb2od(image)
        OD_flat = np.clip(OD.reshape(-1, 3), 0, 3.0)  # cap masked-black pixels
        # Solve stain_matrix.T @ c = OD for each pixel
        conc, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD_flat.T, rcond=None)
        return conc.T  # (N, 2)

    def fit(self, reference: np.ndarray) -> "MacenkoNormalizer":
        self._stain_matrix_target = self._get_stain_matrix(reference)
        conc = self._get_concentrations(reference, self._stain_matrix_target)
        self._maxC_target = np.percentile(conc, 99, axis=0)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self._stain_matrix_target is None:
            raise RuntimeError("Call fit() before transform().")
        H, W, _ = image.shape
        stain_src = self._get_stain_matrix(image)
        conc_src = self._get_concentrations(image, stain_src)

        maxC_src = np.percentile(conc_src, 99, axis=0)
        conc_norm = conc_src / (maxC_src + 1e-6) * self._maxC_target

        OD_norm = conc_norm @ self._stain_matrix_target
        I_norm = 255.0 * np.exp(-OD_norm)
        return I_norm.clip(0, 255).astype(np.uint8).reshape(H, W, 3)


# ---------------------------------------------------------------------------
# Vahadane (2016)
# ---------------------------------------------------------------------------

class VahadaneNormalizer:
    """
    Structure-preserving stain normalisation via sparse NMF.

    Reference:
        Vahadane et al., "Structure-Preserving Color Normalization and Sparse
        Stain Separation for Histological Images", TMI 2016.

    Uses scikit-learn NMF as a widely-available substitute for SPAMS.
    """

    def __init__(self, beta: float = 0.15) -> None:
        self.beta = beta
        self._stain_matrix_target: np.ndarray | None = None
        self._maxC_target: np.ndarray | None = None

    @staticmethod
    def _rgb2od(image: np.ndarray) -> np.ndarray:
        return -np.log((image.astype(np.float64) + 1) / 256.0)

    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import NMF

        OD = self._rgb2od(image)
        OD_flat = OD.reshape(-1, 3)

        mask = np.all(OD_flat > self.beta, axis=1) & np.all(OD_flat < 3.0, axis=1)
        OD_hat = OD_flat[mask]
        if OD_hat.shape[0] < 10:
            OD_hat = OD_flat[np.any(OD_flat > 0.05, axis=1) & np.all(OD_flat < 3.0, axis=1)]
        if OD_hat.shape[0] < 10:
            OD_hat = OD_flat[np.any(OD_flat > 0.05, axis=1)]

        model = NMF(n_components=2, init="nndsvd", max_iter=500, random_state=42)
        model.fit(OD_hat)
        W = model.components_  # (2, 3)

        # Unit-normalise stain vectors
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / (norms + 1e-6)

        # H stain first (highest OD in the red channel)
        if W[0, 0] < W[1, 0]:
            W = W[[1, 0]]

        return W  # (2, 3)

    def _get_concentrations(
        self, image: np.ndarray, stain_matrix: np.ndarray
    ) -> np.ndarray:
        OD = self._rgb2od(image)
        OD_flat = np.clip(OD.reshape(-1, 3), 0, 3.0)  # cap masked-black pixels
        conc, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD_flat.T, rcond=None)
        return conc.T  # (N, 2)

    def fit(self, reference: np.ndarray) -> "VahadaneNormalizer":
        self._stain_matrix_target = self._get_stain_matrix(reference)
        conc = self._get_concentrations(reference, self._stain_matrix_target)
        self._maxC_target = np.percentile(conc, 99, axis=0)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self._stain_matrix_target is None:
            raise RuntimeError("Call fit() before transform().")
        H, W, _ = image.shape
        stain_src = self._get_stain_matrix(image)
        conc_src = self._get_concentrations(image, stain_src)

        maxC_src = np.percentile(conc_src, 99, axis=0)
        conc_norm = conc_src / (maxC_src + 1e-6) * self._maxC_target

        OD_norm = conc_norm @ self._stain_matrix_target
        I_norm = 255.0 * np.exp(-OD_norm)
        return I_norm.clip(0, 255).astype(np.uint8).reshape(H, W, 3)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_METHODS: dict[str, type] = {
    "reinhard": ReinhardNormalizer,
    "macenko": MacenkoNormalizer,
    "vahadane": VahadaneNormalizer,
}


def get_normalizer(method: str) -> ReinhardNormalizer | MacenkoNormalizer | VahadaneNormalizer:
    """
    Instantiate a normaliser by name.

    Args:
        method: One of 'reinhard', 'macenko', 'vahadane'.

    Returns:
        An unfitted normaliser with fit() / transform() methods.
    """
    key = method.lower()
    if key not in _METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(_METHODS)}"
        )
    return _METHODS[key]()
