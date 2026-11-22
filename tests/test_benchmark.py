from types import SimpleNamespace

import numpy as np

from gutdecoder.bench.benchmark import _fit_split


def test_fit_split_passes_alpha_to_ridge():
    rng = np.random.default_rng(0)
    X_train, X_test = rng.normal(size=(40, 8)), rng.normal(size=(10, 8))
    y_train, y_test = rng.normal(size=(40, 2)), rng.normal(size=(10, 2))
    args = SimpleNamespace(dimreduce=None, seed=0, method="ridge", alpha=7.0)

    _, _, bundle, _, _ = _fit_split(X_train, X_test, y_train, y_test, args, genes=["a", "b"])

    assert bundle["regression_model"].alpha == 7.0
