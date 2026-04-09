# metrics.py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from sklearn.metrics import mutual_info_score, roc_auc_score


# --- small helpers ---
def _safe_pearson(x, y):
    if x.size < 2 or y.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x, y):
    if x.size < 2:
        return np.nan
    rho, _ = stats.spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else np.nan


def _rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _nrmse(x, y, method="range"):
    r = _rmse(x, y)
    if method == "range":
        denom = np.max(y) - np.min(y)
    elif method == "mean":
        denom = np.mean(y)
    elif method == "sd":
        denom = np.std(y)
    else:
        raise ValueError("method must be 'range', 'mean' or 'sd'")
    return float(r / denom) if denom != 0 else np.nan


def _normalized_mi(x, y):
    # rank-binning approach
    n = len(x)
    bins = int(max(10, round(n ** (1 / 3))))
    x_d = pd.qcut(pd.Series(x).rank(method="first"), q=bins, labels=False, duplicates="drop")
    y_d = pd.qcut(pd.Series(y).rank(method="first"), q=bins, labels=False, duplicates="drop")
    if len(np.unique(x_d)) < 2 or len(np.unique(y_d)) < 2:
        return np.nan
    mi = mutual_info_score(x_d, y_d)

    def _entropy(labels):
        counts = np.bincount(labels)
        probs = counts[counts > 0] / counts.sum()
        return stats.entropy(probs, base=2)

    h_x = _entropy(x_d.values)
    h_y = _entropy(y_d.values)
    return float(mi / np.sqrt(h_x * h_y)) if h_x and h_y else np.nan


def _js_divergence(p, q):
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    p = p / p.sum() if p.sum() != 0 else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() != 0 else np.ones_like(q) / len(q)
    js = distance.jensenshannon(p, q, base=2)
    return float(js ** 2)


def _ssim_1d(x, y, num_breaks=256):
    if x.size == 0 or y.size == 0:
        return np.nan
    xn = x / np.max(x) if np.max(x) != 0 else np.zeros_like(x)
    yn = y / np.max(y) if np.max(y) != 0 else np.zeros_like(y)
    xd = pd.cut(xn, bins=num_breaks, labels=False)
    yd = pd.cut(yn, bins=num_breaks, labels=False)
    mux, muy = np.mean(xd), np.mean(yd)
    sigxy = np.mean((xd - mux) * (yd - muy))
    sigx, sigy = np.var(xd), np.var(yd)
    C1 = (0.01 * (num_breaks - 1)) ** 2
    C2 = (0.03 * (num_breaks - 1)) ** 2
    denom = ((mux ** 2 + muy ** 2 + C1) * (sigx + sigy + C2))
    return float(((2 * mux * muy + C1) * (2 * sigxy + C2)) / denom) if denom != 0 else np.nan


def _auc_at_cutoff(pred, actual, cutoff):
    label = (actual > cutoff).astype(int)
    if len(np.unique(label)) < 2:
        return np.nan
    return float(roc_auc_score(label, pred))


# ----------------------------
# per-gene metrics (saved to adata.var)
# ----------------------------
def compute_metrics_per_gene(adata, pred_layer="pred", target_layer="target", auc_cutoffs=(0, 1)):
    preds = np.asarray(adata.layers[pred_layer])
    trues = np.asarray(adata.layers[target_layer])

    if preds.shape != trues.shape:
        raise ValueError("pred and target must have same shape")

    n_obs, n_genes = preds.shape
    genes = list(adata.var_names)
    rows = []

    for gi in range(n_genes):
        p = preds[:, gi]
        t = trues[:, gi]
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() < 2:
            continue
        p_ok = p[mask]
        t_ok = t[mask]
        row = {
            "gene": genes[gi],
            "cor_pearson": _safe_pearson(t_ok, p_ok),
            "cor_spearman": _safe_spearman(t_ok, p_ok),
            "var_exprs": float(np.var(t_ok)),
            "var_pred": float(np.var(p_ok)),
            "mean_exprs": float(np.mean(t_ok)),
            "mean_pred": float(np.mean(p_ok)),
            "rmse": _rmse(p_ok, t_ok),
            "mi": _normalized_mi(p_ok, t_ok),
            "js_div": _js_divergence(t_ok, p_ok),
            "nrmse_range": _nrmse(p_ok, t_ok, "range"),
            "nrmse_sd": _nrmse(p_ok, t_ok, "sd"),
            "ssim": _ssim_1d(p_ok, t_ok),
        }
        for c in auc_cutoffs:
            row[f"auc_{c}"] = _auc_at_cutoff(p_ok, t_ok, c)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("gene") if rows else pd.DataFrame(index=adata.var_names)
    # align and write into adata.var
    df = df.reindex(adata.var_names)
    for col in df.columns:
        adata.var[col] = df[col].values
    return df


# ----------------------------
# per-spot metrics (saved to adata.obs)
# ----------------------------
def compute_metrics_per_spot(adata, pred_layer="pred", target_layer="target", auc_cutoffs=(0, 1, 2, 5, 7), min_genes=2):
    preds = np.asarray(adata.layers[pred_layer])
    trues = np.asarray(adata.layers[target_layer])

    if preds.shape != trues.shape:
        raise ValueError("pred and target must have same shape")

    n_obs, n_genes = preds.shape
    spots = list(adata.obs_names)
    rows = []

    for si in range(n_obs):
        p = preds[si, :]
        t = trues[si, :]
        mask = np.isfinite(p) & np.isfinite(t)
        valid = int(mask.sum())
        if valid < min_genes:
            continue
        p_ok = p[mask]
        t_ok = t[mask]
        row = {
            "spot": spots[si],
            "n_genes_valid": valid,
            "cor_pearson": _safe_pearson(t_ok, p_ok),
            "cor_spearman": _safe_spearman(t_ok, p_ok),
            "var_exprs": float(np.var(t_ok)),
            "var_pred": float(np.var(p_ok)),
            "mean_exprs": float(np.mean(t_ok)),
            "mean_pred": float(np.mean(p_ok)),
            "rmse": _rmse(p_ok, t_ok),
            "mi": _normalized_mi(p_ok, t_ok),
            "js_div": _js_divergence(t_ok, p_ok),
            "nrmse_range": _nrmse(p_ok, t_ok, "range"),
            "nrmse_sd": _nrmse(p_ok, t_ok, "sd"),
            "ssim": _ssim_1d(p_ok, t_ok),
        }
        for c in auc_cutoffs:
            row[f"auc_{c}"] = _auc_at_cutoff(p_ok, t_ok, c)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("spot") if rows else pd.DataFrame(index=adata.obs_names)
    df = df.reindex(adata.obs_names)
    for col in df.columns:
        adata.obs[col] = df[col].values
    return df


# ----------------------------
# convenience wrapper for lists
# ----------------------------
def compute_metrics(adata_list,
                          pred_layer="pred",
                          target_layer="target",
                          auc_cutoffs=(0, 1, 2, 5, 7),
                          min_genes=2,
                          per_spot=True,
                          per_gene=True):
    """
    Compute metrics for each AnnData in adata_list and write results into adata.var and adata.obs.
    Returns a list of tuples (df_genes, df_spots) for each adata.
    """
    outputs = []
    for ad in adata_list:
        dg = compute_metrics_per_gene(ad, pred_layer=pred_layer, target_layer=target_layer, auc_cutoffs=auc_cutoffs) if per_gene else None
        ds = compute_metrics_per_spot(ad, pred_layer=pred_layer, target_layer=target_layer, auc_cutoffs=auc_cutoffs, min_genes=min_genes) if per_spot else None
        outputs.append((dg, ds))
    return outputs