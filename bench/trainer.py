import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
import xgboost as xgb
from tqdm import tqdm
import torch
import os
import warnings

os.environ["TQDM_DISABLE"] = "1" # silence training loop

# helper function for cuml for random forest 
# optional: cupy may not be present in CPU-only environments
try:
    import cupy as cp
except Exception:
    cp = None


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
                    fit_intercept=False, 
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

    else:
        raise ValueError(f"Unknown method: {method}")

    # compute metrics per target
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    i = 0
    for target in range(y_test.shape[1]):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]
        l2_error = float(np.mean((preds - target_vals)**2))
        # compute r2 score (guard against zero-variance target)
        denom = np.sum((target_vals - np.mean(target_vals))**2)
        if denom == 0:
            r2_score = float('nan')
        else:
            r2_score = float(1 - np.sum((target_vals - preds)**2) / denom)

        pearson_corr, _ = pearsonr(target_vals, preds)
        if np.isnan(pearson_corr):
            print(f"Warning: NaN pearson for target {target}")
            # optionally print debug arrays
            # print(target_vals)
            # print(preds)
        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        score_dict = {
            'name': genes[i] if genes is not None and i < len(genes) else f"target_{i}",
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
        i += 1

    results = {'l2_errors': list(errors), 
               'r2_scores': list(r2_scores),
               'pearson_corrs': pearson_genes,
               'pearson_mean': float(np.nanmean(pearson_corrs)),
               'pearson_std': float(np.nanstd(pearson_corrs)),
               'l2_error_q1': float(np.percentile(errors, 25)),
               'l2_error_q2': float(np.median(errors)),
               'l2_error_q3': float(np.percentile(errors, 75)),
               'r2_score_q1': float(np.percentile([v for v in r2_scores if not np.isnan(v)], 25)) if any(not np.isnan(v) for v in r2_scores) else float('nan'),
               'r2_score_q2': float(np.median([v for v in r2_scores if not np.isnan(v)])) if any(not np.isnan(v) for v in r2_scores) else float('nan'),
               'r2_score_q3': float(np.percentile([v for v in r2_scores if not np.isnan(v)], 75)) if any(not np.isnan(v) for v in r2_scores) else float('nan'),
               }
    dump = {
        'preds_all': preds_all,
        'targets_all': y_test,
    }

    return results, dump, reg