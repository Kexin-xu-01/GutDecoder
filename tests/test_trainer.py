import numpy as np
import torch

import pytest

from gutdecoder.bench.trainer import _compute_metrics, _train_mlp_torch


def test_mlp_torch_learns_linear_signal():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    w = rng.normal(size=(16, 3))
    X_train = rng.normal(size=(512, 16)).astype(np.float32)
    X_test = rng.normal(size=(128, 16)).astype(np.float32)
    y_train, y_test = X_train @ w, X_test @ w

    preds, _ = _train_mlp_torch(X_train, X_test, y_train)

    for j in range(y_test.shape[1]):
        assert np.corrcoef(preds[:, j], y_test[:, j])[0, 1] > 0.8


def test_metrics_give_nan_for_constant_gene_without_hiding_errors():
    y_test = np.column_stack([np.arange(5.0), np.ones(5)])  # second gene is constant
    preds = np.column_stack([np.arange(5.0) * 2, np.arange(5.0)])

    results, _ = _compute_metrics(preds, y_test, genes=["g1", "g2"])

    assert results["pearson_corrs"][0]["pearson_corr"] == pytest.approx(1.0)
    assert np.isnan(results["pearson_corrs"][1]["pearson_corr"])
    assert results["pearson_mean"] == pytest.approx(1.0)
