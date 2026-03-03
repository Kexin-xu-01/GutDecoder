import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import Ridge
import xgboost as xgb
from tqdm import tqdm
import torch
import os
import warnings
import torch.nn as nn

os.environ["TQDM_DISABLE"] = "1" # silence training loop

# helper function for cuml for random forest 
# optional: cupy may not be present in CPU-only environments
try:
    import cupy as cp
except Exception:
    cp = None

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(1024,512), dropout=0.2):
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
    """
    Return True only if:
      - GPU is present
      - cuML imports
      - CUDA context can be initialized (no driver/runtime mismatch)
    """
    # Optional hard override
    if bool(int(os.environ.get("FORCE_CPU", "0"))):
        return False

    # 1) Check GPU visibility
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        # torch not installed or broken → don't try GPU
        return False

    # 2) Try cuML import
    try:
        import cuml
        from cuml.ensemble import RandomForestRegressor  # noqa
    except Exception as e:
        warnings.warn(f"cuML import failed: {e}")
        return False

    # 3) Try initializing CUDA via cupy / rmm (this catches driver mismatch)
    try:
        import cupy as cp
        # this line actually touches the CUDA driver
        _ = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        warnings.warn(f"CUDA not usable for cuML (driver/runtime issue): {e}")
        return False

    return True

def to_cuml_compatible_array(x, dtype=np.float32):
    """
    Minimal conversion helper:
      - torch.cuda.Tensor -> cupy.ndarray via DLPack (zero-copy)
      - torch.cpu.Tensor -> numpy.ndarray
      - numpy.ndarray -> ensure dtype & C-contiguous
      - cupy.ndarray -> ensure dtype
    Returns the converted array.
    """
    # torch tensor handling
    if isinstance(x, torch.Tensor):
        x_det = x.detach()
        if x_det.is_cuda:
            if cp is None:
                raise RuntimeError("cupy is required to pass CUDA tensors to cuML. Install a matching cupy/cuML build.")
            # ensure contiguous
            x_det = x_det.contiguous()
            dlpack = torch.utils.dlpack.to_dlpack(x_det)
            arr = cp.from_dlpack(dlpack)
            if arr.dtype != cp.dtype(dtype):
                arr = arr.astype(dtype, copy=False)
            return arr
        else:
            arr = np.asarray(x_det.cpu().numpy(), dtype=dtype, order='C')
            return arr

    # numpy array
    if isinstance(x, np.ndarray):
        if x.dtype != dtype:
            arr = x.astype(dtype, copy=False)
        else:
            arr = x
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return arr

    # cupy array
    if cp is not None and isinstance(x, cp.ndarray):
        if x.dtype != cp.dtype(dtype):
            arr = x.astype(dtype, copy=False)
        else:
            arr = x
        return arr

    # fallback: convert to numpy
    arr = np.asarray(x, dtype=dtype)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr



def train_test_reg(X_train, X_test, y_train, y_test, 
                   max_iter=1000, random_state=0, genes=None, alpha=None, method='ridge'):
    """
    Train and test a regression model.

    Returns:
        results (dict), dump (dict), reg (estimator or list of estimators)
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr
    from tqdm import tqdm

    reg = None  # ensure variable exists

    if method == 'ridge':
        # default alpha if not provided
        if alpha is None:
            alpha = 100 / (X_train.shape[1] * y_train.shape[1])

        print(f"Using alpha: {alpha}")
        reg = Ridge(solver='lsqr',
                    alpha=alpha, 
                    random_state=random_state, 
                    fit_intercept=True, # if no intercept, ridge is forced to go through zero
                    max_iter=max_iter)
        reg.fit(X_train, y_train)

        preds_all = reg.predict(X_test)

    elif method in ('random-forest', 'rf'):
        # Try to use RAPIDS cuML if available and X is on GPU / RAPIDS environment,
        # otherwise fall back to sklearn's RandomForestRegressor.
        import warnings
        import os

        use_cuml = can_use_cuml()

        if use_cuml:
            import cuml
            from cuml.ensemble import RandomForestRegressor as cuRF

            # Convert train/test and targets once (cupy via DLPack if CUDA torch.Tensor)
            X_train_conv = to_cuml_compatible_array(X_train, dtype=np.float32)
            X_test_conv  = to_cuml_compatible_array(X_test,  dtype=np.float32)
            y_train_conv = to_cuml_compatible_array(y_train, dtype=np.float32)
            # Note: we keep y_test as-is for metric calculation later (we'll convert if needed)

            preds_per_target = []
            regs = []
            n_genes = int(y_train_conv.shape[1])
            for i in tqdm(range(n_genes)):
                print('fitting model ', i)
                reg_i = cuRF(n_estimators=70, random_state=random_state)
                # ensure 1D y
                y1 = y_train_conv[:, i].ravel() if hasattr(y_train_conv[:, i], 'ravel') else y_train_conv[:, i]
                reg_i.fit(X_train_conv, y1)
                # predict on X_test_conv
                res = reg_i.predict(X_test_conv)
                # convert cupy results to numpy so downstream code (metrics) works as before
                try:
                    import cupy as cp
                    if cp is not None and isinstance(res, cp.ndarray):
                        res = cp.asnumpy(res)
                except Exception:
                    # if cupy not available or conversion not needed, pass
                    pass

                preds_per_target.append(np.asarray(res).ravel())
                regs.append(reg_i)

            # assemble numpy preds_all
            preds_all = np.column_stack(preds_per_target)
            reg = regs  # list of per-target cuML regressors

        else:
            # fallback to sklearn
            from sklearn.ensemble import RandomForestRegressor as skRF

            n_targets = y_train.shape[1]
            n_test = X_test.shape[0]
            preds_list = []
            regs = []

            for i in range(n_targets):
                print(f"Fitting RandomForest target {i+1}/{n_targets}")
                y_col = y_train[:, i]

                reg_i = skRF(n_estimators=70, random_state=random_state, n_jobs=-1)
                reg_i.fit(X_train, y_col)
                pred_i = reg_i.predict(X_test)

                pred_i = np.asarray(pred_i).reshape(n_test,)
                preds_list.append(pred_i)
                regs.append(reg_i)

            preds_all = np.column_stack(preds_list)
            reg = regs  # list of per-target sklearn regressors

    elif method == 'xgboost':
        import xgboost as xgb
        reg = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            random_state=random_state
        )
        reg.fit(X_train, y_train)
        preds_all = reg.predict(X_test)


    elif method == 'mlp_torch':

        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train_np = np.asarray(X_train, dtype=np.float32)
        y_train_np = np.asarray(y_train, dtype=np.float32)
        X_test_np  = np.asarray(X_test, dtype=np.float32)
    
        y_scaler = StandardScaler()
            # ensure 2D (n_samples, n_targets)
        y_train_2d = y_train.reshape(y_train.shape[0], -1)
        y_train_np = y_scaler.fit_transform(y_train_2d).astype(np.float32)

        in_dim = X_train_np.shape[1]
        out_dim = y_train_np.shape[1]

        # ---- Instantiate your top-level MLP ----
        reg = MLP(in_dim=in_dim, out_dim=out_dim).to(device)

        optimizer = torch.optim.AdamW(reg.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_np),
                torch.tensor(y_train_np)
            ),
            batch_size=128,
            shuffle=True
        )

        reg.train()
        for epoch in range(40):
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                preds = reg(xb)
                loss = criterion(preds, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss {total_loss:.4f}")

        # ---- Predict ----
        reg.eval()
        with torch.no_grad():
            preds = reg(torch.tensor(X_test_np).to(device)).cpu().numpy()
            preds = y_scaler.inverse_transform(preds)

        preds_all = preds

    elif method == 'mlp_sklearn':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        X_train_np = np.asarray(X_train)
        X_test_np  = np.asarray(X_test)
        y_train_np = np.asarray(y_train)
        y_test_np  = np.asarray(y_test)

        # ---- Optional Y scaling (recommended) ----
        scale_y = True
        if scale_y:
            y_scaler = StandardScaler()
            y_train_np = y_scaler.fit_transform(y_train_np)
        else:
            y_scaler = None

        reg = MLPRegressor(
            hidden_layer_sizes=(1024, 512),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=True,
            random_state=random_state
        )

        reg.fit(X_train_np, y_train_np)

        preds = reg.predict(X_test_np)

        if scale_y:
            preds = y_scaler.inverse_transform(preds)

        preds_all = preds


    else:
        raise ValueError(f"Unknown method: {method}")

    # compute metrics per target
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    spearman_corrs = []
    spearman_genes = []

    n_targets = y_test.shape[1]
    for i in range(n_targets):
        preds = preds_all[:, i]
        target_vals = y_test[:, i]

        # L2 / MSE
        l2_error = float(np.mean((preds - target_vals) ** 2))

        # R^2 (guard against zero variance)
        denom = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if denom == 0:
            r2_score = float('nan')
        else:
            r2_score = float(1 - np.sum((target_vals - preds) ** 2) / denom)

        # Pearson (handle constant arrays -> returns nan)
        try:
            pearson_corr, _ = pearsonr(target_vals, preds)
        except Exception:
            pearson_corr = float('nan')
        if np.isnan(pearson_corr):
            print(f"Warning: NaN pearson for target {i}")

        # Spearman (rank correlation) (handle constant arrays -> returns nan)
        try:
            spearman_corr, _ = spearmanr(target_vals, preds)
        except Exception:
            spearman_corr = float('nan')
        if np.isnan(spearman_corr):
            print(f"Warning: NaN spearman for target {i}")

        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)

        gene_name = genes[i] if genes is not None and i < len(genes) else f"target_{i}"
        pearson_genes.append({
            'name': gene_name,
            'pearson_corr': pearson_corr,
        })
        spearman_genes.append({
            'name': gene_name,
            'spearman_corr': spearman_corr,
        })

    # build results (robust to all-NaN lists)
    valid_pearson = [v for v in pearson_corrs if not np.isnan(v)]
    valid_spearman = [v for v in spearman_corrs if not np.isnan(v)]
    valid_r2 = [v for v in r2_scores if not np.isnan(v)]

    results = {
        'l2_errors': list(errors),
        'r2_scores': list(r2_scores),
        'pearson_corrs': pearson_genes,
        'spearman_corrs': spearman_genes,
        'pearson_mean': float(np.nanmean(pearson_corrs)) if len(pearson_corrs) > 0 else float('nan'),
        'pearson_std': float(np.nanstd(pearson_corrs)) if len(pearson_corrs) > 0 else float('nan'),
        'spearman_mean': float(np.nanmean(spearman_corrs)) if len(spearman_corrs) > 0 else float('nan'),
        'spearman_std': float(np.nanstd(spearman_corrs)) if len(spearman_corrs) > 0 else float('nan'),
        'l2_error_q1': float(np.percentile(errors, 25)) if len(errors) > 0 else float('nan'),
        'l2_error_q2': float(np.median(errors)) if len(errors) > 0 else float('nan'),
        'l2_error_q3': float(np.percentile(errors, 75)) if len(errors) > 0 else float('nan'),
        'r2_score_q1': float(np.percentile(valid_r2, 25)) if len(valid_r2) > 0 else float('nan'),
        'r2_score_q2': float(np.median(valid_r2)) if len(valid_r2) > 0 else float('nan'),
        'r2_score_q3': float(np.percentile(valid_r2, 75)) if len(valid_r2) > 0 else float('nan'),
    }

    dump = {
        'preds_all': preds_all,
        'targets_all': y_test,
    }

    return results, dump, reg