import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
import xgboost as xgb
from tqdm import tqdm
import torch
import os
import warnings
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from loguru import logger

os.environ["TQDM_DISABLE"] = "1"

try:
    import cupy as cp
except Exception:
    cp = None


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(1024, 512), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def can_use_cuml():
    if bool(int(os.environ.get("FORCE_CPU", "0"))):
        return False
    try:
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        import cuml
        from cuml.ensemble import RandomForestRegressor  # noqa
    except Exception as e:
        warnings.warn(f"cuML import failed: {e}")
        return False
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        warnings.warn(f"CUDA not usable for cuML (driver/runtime issue): {e}")
        return False
    return True


def to_cuml_compatible_array(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        x_det = x.detach()
        if x_det.is_cuda:
            if cp is None:
                raise RuntimeError("cupy is required to pass CUDA tensors to cuML.")
            x_det = x_det.contiguous()
            dlpack = torch.utils.dlpack.to_dlpack(x_det)
            arr = cp.from_dlpack(dlpack)
            if arr.dtype != cp.dtype(dtype):
                arr = arr.astype(dtype, copy=False)
            return arr
        else:
            return np.asarray(x_det.cpu().numpy(), dtype=dtype, order='C')
    if isinstance(x, np.ndarray):
        arr = x.astype(dtype, copy=False) if x.dtype != dtype else x
        return np.ascontiguousarray(arr) if not arr.flags['C_CONTIGUOUS'] else arr
    if cp is not None and isinstance(x, cp.ndarray):
        return x.astype(dtype, copy=False) if x.dtype != cp.dtype(dtype) else x
    arr = np.asarray(x, dtype=dtype)
    return np.ascontiguousarray(arr) if not arr.flags['C_CONTIGUOUS'] else arr


# ---------------------------------------------------------------------------
# Per-method trainers — each returns (preds_all: np.ndarray, reg: estimator)
# ---------------------------------------------------------------------------

def _train_ridge(X_train, X_test, y_train, alpha, random_state, max_iter):
    if alpha is None:
        alpha = 100 / (X_train.shape[1] * y_train.shape[1])
    logger.info(f"Using alpha: {alpha}")
    reg = Ridge(
        solver='lsqr', alpha=alpha, random_state=random_state,
        fit_intercept=True, max_iter=max_iter,
    )
    reg.fit(X_train, y_train)
    return reg.predict(X_test), reg


def _train_rf(X_train, X_test, y_train, random_state):
    if can_use_cuml():
        import cuml
        from cuml.ensemble import RandomForestRegressor as cuRF

        X_tr = to_cuml_compatible_array(X_train, dtype=np.float32)
        X_te = to_cuml_compatible_array(X_test,  dtype=np.float32)
        y_tr = to_cuml_compatible_array(y_train, dtype=np.float32)

        preds_per_target, regs = [], []
        for i in tqdm(range(int(y_tr.shape[1]))):
            logger.debug(f"Fitting cuML RF target {i}")
            reg_i = cuRF(n_estimators=70, random_state=random_state)
            y1 = y_tr[:, i].ravel()
            reg_i.fit(X_tr, y1)
            res = reg_i.predict(X_te)
            try:
                if cp is not None and isinstance(res, cp.ndarray):
                    res = cp.asnumpy(res)
            except Exception:
                pass
            preds_per_target.append(np.asarray(res).ravel())
            regs.append(reg_i)
        return np.column_stack(preds_per_target), regs
    else:
        from sklearn.ensemble import RandomForestRegressor as skRF

        n_targets, n_test = y_train.shape[1], X_test.shape[0]
        preds_list, regs = [], []
        for i in range(n_targets):
            logger.debug(f"Fitting sklearn RF target {i+1}/{n_targets}")
            reg_i = skRF(n_estimators=70, random_state=random_state, n_jobs=-1)
            reg_i.fit(X_train, y_train[:, i])
            preds_list.append(np.asarray(reg_i.predict(X_test)).reshape(n_test,))
            regs.append(reg_i)
        return np.column_stack(preds_list), regs


def _train_xgboost(X_train, X_test, y_train, random_state):
    reg = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        gamma=0, reg_alpha=0, reg_lambda=1, random_state=random_state,
    )
    reg.fit(X_train, y_train)
    return reg.predict(X_test), reg


def _train_mlp_torch(X_train, X_test, y_train):
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = np.asarray(X_train, dtype=np.float32)
    X_te = np.asarray(X_test,  dtype=np.float32)
    y_scaler = StandardScaler()
    y_tr = y_scaler.fit_transform(y_train.reshape(y_train.shape[0], -1)).astype(np.float32)

    reg = MLP(in_dim=X_tr.shape[1], out_dim=y_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(reg.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=128, shuffle=True,
    )
    reg.train()
    for epoch in range(40):
        total_loss = sum(
            (optimizer.zero_grad() or True) and
            (loss := criterion(reg(xb.to(device)), yb.to(device))) and
            loss.backward() or loss.item()
            for xb, yb in loader
        )
        logger.debug(f"Epoch {epoch+1}, Loss {total_loss:.4f}")

    reg.eval()
    with torch.no_grad():
        preds = reg(torch.tensor(X_te).to(device)).cpu().numpy()
    return y_scaler.inverse_transform(preds), reg


def _train_mlp_sklearn(X_train, X_test, y_train, random_state=0):
    from sklearn.neural_network import MLPRegressor

    y_scaler = StandardScaler()
    y_tr = y_scaler.fit_transform(np.asarray(y_train))
    reg = MLPRegressor(
        hidden_layer_sizes=(1024, 512), activation='relu', solver='adam',
        alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
        max_iter=200, early_stopping=True, n_iter_no_change=10,
        verbose=False, random_state=random_state,
    )
    reg.fit(np.asarray(X_train), y_tr)
    preds = y_scaler.inverse_transform(reg.predict(np.asarray(X_test)))
    return preds, reg


_METHODS = {
    'ridge':        _train_ridge,
    'random-forest': _train_rf,
    'rf':           _train_rf,
    'xgboost':      _train_xgboost,
    'mlp_torch':    _train_mlp_torch,
    'mlp_sklearn':  _train_mlp_sklearn,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(preds_all, y_test, genes=None):
    """Compute per-gene metrics and return (results dict, dump dict)."""
    errors, r2_scores, pearson_corrs, pearson_genes = [], [], [], []
    spearman_corrs, spearman_genes = [], []

    for i in range(y_test.shape[1]):
        pred = preds_all[:, i]
        true = y_test[:, i]

        l2_error = float(np.mean((pred - true) ** 2))

        denom = np.sum((true - np.mean(true)) ** 2)
        r2 = float('nan') if denom == 0 else float(1 - np.sum((true - pred) ** 2) / denom)

        try:
            pc, _ = pearsonr(true, pred)
        except Exception:
            pc = float('nan')
        if np.isnan(pc):
            logger.warning(f"NaN pearson for target {i}")

        try:
            sc, _ = spearmanr(true, pred)
        except Exception:
            sc = float('nan')
        if np.isnan(sc):
            logger.warning(f"NaN spearman for target {i}")

        gene_name = genes[i] if genes is not None and i < len(genes) else f"target_{i}"
        errors.append(l2_error)
        r2_scores.append(r2)
        pearson_corrs.append(pc)
        spearman_corrs.append(sc)
        pearson_genes.append({'name': gene_name, 'pearson_corr': pc})
        spearman_genes.append({'name': gene_name, 'spearman_corr': sc})

    valid_r2 = [v for v in r2_scores if not np.isnan(v)]
    results = {
        'l2_errors':      list(errors),
        'r2_scores':      list(r2_scores),
        'pearson_corrs':  pearson_genes,
        'spearman_corrs': spearman_genes,
        'pearson_mean':   float(np.nanmean(pearson_corrs)) if pearson_corrs else float('nan'),
        'pearson_std':    float(np.nanstd(pearson_corrs))  if pearson_corrs else float('nan'),
        'spearman_mean':  float(np.nanmean(spearman_corrs)) if spearman_corrs else float('nan'),
        'spearman_std':   float(np.nanstd(spearman_corrs))  if spearman_corrs else float('nan'),
        'l2_error_q1':    float(np.percentile(errors, 25)) if errors else float('nan'),
        'l2_error_q2':    float(np.median(errors))         if errors else float('nan'),
        'l2_error_q3':    float(np.percentile(errors, 75)) if errors else float('nan'),
        'r2_score_q1':    float(np.percentile(valid_r2, 25)) if valid_r2 else float('nan'),
        'r2_score_q2':    float(np.median(valid_r2))         if valid_r2 else float('nan'),
        'r2_score_q3':    float(np.percentile(valid_r2, 75)) if valid_r2 else float('nan'),
    }
    dump = {'preds_all': preds_all, 'targets_all': y_test}
    return results, dump


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def train_test_reg(X_train, X_test, y_train, y_test,
                   max_iter=1000, random_state=0, genes=None, alpha=None, method='ridge'):
    """Train a regression model and evaluate it. Returns (results, dump, reg)."""
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(_METHODS)}")

    fn = _METHODS[method]
    # ridge needs alpha/random_state/max_iter; rf/xgboost need random_state; mlp methods need none
    if method == 'ridge':
        preds_all, reg = fn(X_train, X_test, y_train, alpha, random_state, max_iter)
    elif method in ('random-forest', 'rf'):
        preds_all, reg = fn(X_train, X_test, y_train, random_state)
    elif method == 'xgboost':
        preds_all, reg = fn(X_train, X_test, y_train, random_state)
    elif method == 'mlp_sklearn':
        preds_all, reg = fn(X_train, X_test, y_train, random_state)
    else:  # mlp_torch
        preds_all, reg = fn(X_train, X_test, y_train)

    results, dump = _compute_metrics(preds_all, y_test, genes)
    return results, dump, reg
