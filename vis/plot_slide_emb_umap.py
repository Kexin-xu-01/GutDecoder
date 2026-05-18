from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_feature_vector(h5_path: Path, key: str = "features") -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"{key!r} not found in {h5_path.name}. Keys: {list(f.keys())}")
        x = f[key][:]

    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    if x.ndim != 2:
        raise ValueError(f"Unexpected feature shape {x.shape} in {h5_path.name}")

    return x


def coalesce_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    out = df[cols[0]].copy()
    for c in cols[1:]:
        out = out.combine_first(df[c])
    return out


def load_features(h5_root: Path, feature_key: str = "features") -> pd.DataFrame:
    h5_files = sorted(h5_root.rglob("*.h5"))
    print(f"Found {len(h5_files)} H5 files")

    records = []
    for p in h5_files:
        try:
            x = load_feature_vector(p, key=feature_key)
            records.append({"sample_id": p.stem, "features": x.squeeze()})
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    feat_df = pd.DataFrame(records)
    feat_df["sample_id"] = feat_df["sample_id"].astype("string")
    print("Loaded features for:", len(feat_df), "samples")
    return feat_df


def load_metadata(
    xenium_meta_path: Path,
    hest_meta_path: Path,
    sample_id_col: str = "Sample_ID",
    hest_id_col: str = "sample_id",
) -> pd.DataFrame:
    xenium_meta = pd.read_excel(xenium_meta_path, engine="openpyxl")
    hest_meta = pd.read_csv(hest_meta_path)

    if sample_id_col not in xenium_meta.columns:
        raise KeyError(f"{sample_id_col!r} not in xenium metadata columns")
    if hest_id_col not in hest_meta.columns:
        raise KeyError(f"{hest_id_col!r} not in hest metadata columns")

    xenium_meta[sample_id_col] = xenium_meta[sample_id_col].astype("string")
    hest_meta[hest_id_col] = hest_meta[hest_id_col].astype("string")

    return xenium_meta, hest_meta

def load_metadata(
    hest_meta_path: Path,
    hest_id_col: str = "sample_id",
) -> pd.DataFrame:
    hest_meta = pd.read_csv(hest_meta_path)

    if hest_id_col not in hest_meta.columns:
        raise KeyError(f"{hest_id_col!r} not in hest metadata columns")

    hest_meta[hest_id_col] = hest_meta[hest_id_col].astype("string")

    return hest_meta


def compute_umap(X: np.ndarray, random_state: int = 0) -> np.ndarray:
    X = StandardScaler().fit_transform(X)

    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    if n_pca >= 2:
        Xr = PCA(n_components=n_pca, random_state=0).fit_transform(X)
    else:
        Xr = X

    embedding = umap.UMAP(
        n_neighbors=min(15, max(2, X.shape[0] - 1)),
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    ).fit_transform(X)

    return embedding



def plot_categorical(ax, emb, values, title):
    vals = pd.Series(values).astype("string").fillna("NA")
    cats = pd.unique(vals)
    n = len(cats)

    cmap_name = "tab20" if n <= 20 else "hsv"
    cmap = mpl.colormaps.get_cmap(cmap_name)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    color_map = dict(zip(cats, colors))

    for cat in cats:
        mask = (vals == cat).fillna(False).to_numpy(dtype=bool)
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=18,
            alpha=0.85,
            color=color_map[cat],
            label=str(cat),
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    ncol = 1 if n <= 20 else 2 if n <= 50 else 3 if n <= 100 else 4
    ax.legend(
        fontsize=6,
        markerscale=1.2,
        frameon=False,
        ncol=ncol,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )


def plot_numeric(ax, emb, values, title):
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    mask = vals.notna().to_numpy()

    sc = ax.scatter(
        emb[mask, 0],
        emb[mask, 1],
        c=vals[mask],
        s=18,
        alpha=0.9,
        cmap="viridis",
        edgecolors="none",
    )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def _plot_page(ax, emb, series, title, is_numeric: bool):
    if is_numeric:
        plot_numeric(ax, emb, series, title)
    else:
        plot_categorical(ax, emb, series, title)


def make_umap_pdf(
    meta: pd.DataFrame,
    out_pdf: Path,
    categorical_cols: list[str],
    numeric_cols: list[str],
    group_by_cols: list[str],
    feature_col: str = "features",
    sample_id_col: str = "sample_id",
    recompute_group_umap: bool = True,
):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    cat_cols = [c for c in categorical_cols if c in meta.columns]
    num_cols = [c for c in numeric_cols if c in meta.columns]
    group_by_cols = [c for c in group_by_cols if c in meta.columns]

    with PdfPages(out_pdf) as pdf:
        # -----------------------
        # Overall pages
        # -----------------------
        global_X = np.stack(meta[feature_col].to_numpy())
        global_emb = compute_umap(global_X)

        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(10, 7))
            _plot_page(
                ax,
                global_emb,
                meta[col],
                f"Overall: {col}",
                is_numeric=False,
            )
            fig.suptitle(f"UMAP colored by {col}", fontsize=15, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for col in num_cols:
            fig, ax = plt.subplots(figsize=(10, 7))
            _plot_page(
                ax,
                global_emb,
                meta[col],
                f"Overall: {col}",
                is_numeric=True,
            )
            fig.suptitle(f"UMAP colored by {col}", fontsize=15, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # -----------------------
        # Grouped pages
        # -----------------------
        for group_col in group_by_cols:
            group_vals = pd.Series(meta[group_col]).astype("string").fillna("NA")
            groups = [g for g in pd.unique(group_vals) if g != "NA"]

            for g in sorted(groups, key=lambda x: str(x)):
                sub = meta[group_vals == g].copy()
                if len(sub) < 2:
                    print(f"Skipping {group_col}={g}: too few samples ({len(sub)})")
                    continue

                if recompute_group_umap:
                    emb = compute_umap(np.stack(sub[feature_col].to_numpy()))
                else:
                    emb = global_emb[group_vals == g]

                for col in cat_cols:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    _plot_page(
                        ax,
                        emb,
                        sub[col],
                        f"{group_col}={g}: {col}",
                        is_numeric=False,
                    )
                    fig.suptitle(f"{group_col} = {g} | colored by {col}", fontsize=15, y=1.02)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                for col in num_cols:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    _plot_page(
                        ax,
                        emb,
                        sub[col],
                        f"{group_col}={g}: {col}",
                        is_numeric=True,
                    )
                    fig.suptitle(f"{group_col} = {g} | colored by {col}", fontsize=15, y=1.02)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

    print(f"Saved PDF to: {out_pdf}")


def make_default_pdf_name(dataset_name: str, out_dir: Path) -> Path:
    return out_dir / f"umap_{dataset_name}.pdf"