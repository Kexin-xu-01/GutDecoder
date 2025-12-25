"""
Monkey patch for scanpy.pp.calculate_qc_metrics
to make it safe for datasets with few features.
"""

import scanpy as sc
import logging

logger = logging.getLogger(__name__)


def apply_qc_patch(
    default_percent_top=(10, 50, 100, 200, 500),
    fallback=None,
):
    """
    Apply a monkey patch to scanpy.pp.calculate_qc_metrics.

    Args:
        default_percent_top (tuple[int]): Default percent_top values to try.
        fallback:
            - None: remove percent_top entirely if invalid
            - int: force percent_top=[min(fallback, n_vars)]
    """

    # Avoid double patching
    if getattr(sc.pp, "_qc_patch_applied", False):
        return

    original_fn = sc.pp.calculate_qc_metrics

    def patched_calculate_qc_metrics(*args, **kwargs):
        # Extract adata
        adata = args[0] if args else kwargs.get("adata", None)
        if adata is None:
            return original_fn(*args, **kwargs)

        # Determine number of features
        n_vars = getattr(adata, "n_vars", None)
        if n_vars is None:
            try:
                n_vars = adata.shape[1]
            except Exception:
                n_vars = 0

        # Get percent_top from kwargs or defaults
        if "percent_top" in kwargs:
            percent_top = kwargs["percent_top"]
        else:
            percent_top = default_percent_top

        if percent_top is not None:
            try:
                if isinstance(percent_top, int):
                    percent_top_list = [percent_top]
                else:
                    percent_top_list = list(percent_top)

                percent_top_filtered = [p for p in percent_top_list if p <= n_vars]
            except Exception:
                percent_top_filtered = []

            if not percent_top_filtered:
                if fallback is None:
                    kwargs.pop("percent_top", None)
                    logger.info(
                        "QC patch: n_vars=%d too small; disabling percent_top.",
                        n_vars,
                    )
                else:
                    kwargs["percent_top"] = [min(fallback, n_vars)]
                    logger.info(
                        "QC patch: n_vars=%d too small; forcing percent_top=%s.",
                        n_vars,
                        kwargs["percent_top"],
                    )
            else:
                kwargs["percent_top"] = percent_top_filtered

        return original_fn(*args, **kwargs)

    # Apply patch
    sc.pp.calculate_qc_metrics = patched_calculate_qc_metrics
    sc.pp._qc_patch_applied = True
