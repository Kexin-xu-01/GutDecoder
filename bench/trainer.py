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
    
    
    if method == 'ridge':
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
        import numpy as np
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
            y_test_conv  = to_cuml_compatible_array(y_test,  dtype=np.float32)

            def train_regressor(X_conv, y_column, i):
                print('fitting model ', i)
                regressor = cuRF(n_estimators=70, random_state=random_state)
                # ensure 1D y
                if hasattr(y_column, 'ravel'):
                    y1 = y_column.ravel()
                else:
                    y1 = y_column
                regressor.fit(X_conv, y1)
                # predict on X_test_conv
                res = regressor.predict(X_test_conv)
                # convert cupy results to numpy so downstream code (metrics) works as before
                if cp is not None and isinstance(res, cp.ndarray):
                    res = cp.asnumpy(res)
                del regressor
                return res
            
            results = []
            # iterate genes: use shape from converted y_train (works for numpy or cupy)
            n_genes = int(y_train_conv.shape[1])
            for i in tqdm(range(n_genes)):
                # indexing returns the appropriate backend (numpy or cupy)
                y_col = y_train_conv[:, i]
                results.append(train_regressor(X_train_conv, y_col, i))
            
            # assemble numpy preds_all (results elements already converted to numpy in train_regressor)
            preds_all = np.zeros(y_test.shape)
            for i in range(len(results)):
                preds_all[:, i] = results[i]
        

        # fallback to sklearn
        if not use_cuml:
            from sklearn.ensemble import RandomForestRegressor as skRF

            n_targets = y_train.shape[1]
            n_test = X_test.shape[0]
            preds_list = []

            for i in range(n_targets):
                print(f"Fitting RandomForest target {i+1}/{n_targets}")
                y_col = y_train[:, i]

                reg = skRF(n_estimators=70, random_state=random_state, n_jobs=-1)
                reg.fit(X_train, y_col)
                pred_i = reg.predict(X_test)

                pred_i = pred_i.reshape(n_test,)
                preds_list.append(pred_i)
                del reg

            preds_all = np.column_stack(preds_list)

            
    elif method == 'xgboost':
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
            

    
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    i = 0
    for target in range(y_test.shape[1]):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]
        l2_error = float(np.mean((preds - target_vals)**2))
        # compute r2 score
        r2_score = float(1 - np.sum((target_vals - preds)**2) / np.sum((target_vals - np.mean(target_vals))**2))
        pearson_corr, _ = pearsonr(target_vals, preds)
        if np.isnan(pearson_corr):
            print(target_vals)
            print(preds)
        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        score_dict = {
            'name': genes[i],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)
        i += 1
        

    results = {'l2_errors': list(errors), 
               'r2_scores': list(r2_scores),
               'pearson_corrs': pearson_genes,
               'pearson_mean': float(np.mean(pearson_corrs)),
               'pearson_std': float(np.std(pearson_corrs)),
               'l2_error_q1': float(np.percentile(errors, 25)),
               'l2_error_q2': float(np.median(errors)),
               'l2_error_q3': float(np.percentile(errors, 75)),
               'r2_score_q1': float(np.percentile(r2_scores, 25)),
               'r2_score_q2': float(np.median(r2_scores)),
               'r2_score_q3': float(np.percentile(r2_scores, 75)),}
    dump = {
        'preds_all': preds_all,
        'targets_all': y_test,
    }
    
    return results, dump