import numpy as np
import torch

from gutdecoder.bench.trainer import _train_mlp_torch


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
