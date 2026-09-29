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
import harmonypy as hm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_feature_vector(
    h5_path: Path,
    key: str = "features",
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(
                f"{key!r} not found in {h5_path.name}. "
                f"Keys: {list(f.keys())}"
            )

        x = f[key][:]

    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    if x.ndim != 2:
        raise ValueError(
            f"Unexpected feature shape {x.shape} in {h5_path.name}"
        )

    return x


def load_features(
    h5_root: Path,
    feature_key: str = "features",
    sample_ids=None,
) -> pd.DataFrame:

    h5_root = Path(h5_root)

    if sample_ids is None:
        h5_files = sorted(h5_root.rglob("*.h5"))
    else:
        h5_files = [
            h5_root / f"{sample_id}.h5"
            for sample_id in sample_ids
        ]

    print(f"Found/requested {len(h5_files)} H5 files")

    records = []

    for p in h5_files:
        try:
            x = load_feature_vector(
                p,
                key=feature_key,
            )

            records.append({
                "sample_id": p.stem,
                "features": x.squeeze(),
            })

        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    if not records:
        raise ValueError(
            f"No H5 features successfully loaded from {h5_root}"
        )

    feat_df = pd.DataFrame(records)
    feat_df["sample_id"] = feat_df["sample_id"].astype("string")

    print(f"Loaded features for {len(feat_df)} samples")

    return feat_df


def coalesce_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    out = df[cols[0]].copy()
    for c in cols[1:]:
        out = out.combine_first(df[c])
    return out


def load_metadata(
    metadata_path: Path,
    sample_id_col: str,
) -> pd.DataFrame:

    metadata_path = Path(metadata_path)

    if metadata_path.suffix.lower() == ".csv":
        meta = pd.read_csv(metadata_path)

    elif metadata_path.suffix.lower() in [".xlsx", ".xls"]:
        meta = pd.read_excel(metadata_path)

    else:
        raise ValueError(
            f"Unsupported metadata file: {metadata_path}"
        )

    if sample_id_col not in meta.columns:
        raise KeyError(
            f"{sample_id_col!r} not in metadata columns"
        )

    meta[sample_id_col] = meta[sample_id_col].astype("string")

    return meta

def merge_metadata(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_id_col: str,
    feature_id_col: str = "sample_id",
) -> pd.DataFrame:

    features = features.copy()
    metadata = metadata.copy()

    features[feature_id_col] = features[feature_id_col].astype("string")
    metadata[metadata_id_col] = metadata[metadata_id_col].astype("string")

    merged = features.merge(
        metadata,
        left_on=feature_id_col,
        right_on=metadata_id_col,
        how="left",
        validate="many_to_one",
    )

    return merged


def compute_umap(X: np.ndarray, random_state: int = 0) -> np.ndarray:
    X = StandardScaler().fit_transform(X)

    n_pca = min(50, X.shape[1], X.shape[0] - 1)

    if n_pca >= 2:
        Xr = PCA(
            n_components=n_pca,
            random_state=random_state,
        ).fit_transform(X)
    else:
        Xr = X

    embedding = umap.UMAP(
        n_neighbors=min(15, max(2, Xr.shape[0] - 1)),
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    ).fit_transform(Xr)

    return embedding

def compute_harmony_umap(
    X: np.ndarray,
    batch,
    n_pca: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 0,
):
    """
    Standardize -> PCA -> Harmony -> UMAP.

    Parameters
    ----------
    X
        Slide embedding matrix, shape (n_samples, n_features).

    batch
        Batch/group label for each sample, e.g. CZI vs NDPI.

    Returns
    -------
    embedding
        2D UMAP coordinates.

    X_harmony
        Harmony-corrected PCA representation.
    """

    # -----------------------
    # Standardize
    # -----------------------
    X_scaled = StandardScaler().fit_transform(X)

    # -----------------------
    # PCA
    # -----------------------
    n_components = min(
        n_pca,
        X_scaled.shape[1],
        X_scaled.shape[0] - 1,
    )

    X_pca = PCA(
        n_components=n_components,
        random_state=random_state,
    ).fit_transform(X_scaled)

    print("PCA:", X_pca.shape)

    # -----------------------
    # Harmony
    # -----------------------
    harmony_meta = pd.DataFrame({
        "batch": pd.Series(batch).astype(str).to_numpy()
    })

    harmony = hm.run_harmony(
        X_pca,
        harmony_meta,
        vars_use=["batch"],
        random_state=random_state,
    )

    # harmonypy returns dimensions x samples
    X_harmony = harmony.Z_corr.T

    print("Harmony:", X_harmony.shape)

    # -----------------------
    # UMAP
    # -----------------------
    actual_neighbors = min(
        n_neighbors,
        X_harmony.shape[0] - 1,
    )

    embedding = umap.UMAP(
        n_neighbors=actual_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(X_harmony)

    return embedding, X_harmony

def get_encoder_paths(base_path):
    base_path = Path(base_path)

    encoder_configs = {
        "prism": "20x_224px_0px_overlap",
        "chief": "10x_256px_0px_overlap",
        "madeleine": "10x_256px_0px_overlap",
        "gigapath": "20x_256px_0px_overlap",
        "titan": "20x_512px_0px_overlap",
        "feather": "20x_512px_0px_overlap",
        "prism2": "20x_224px_0px_overlap",
        "care": "20x_512px_0px_overlap",
        "feather_uni_v2": "20x_256px_0px_overlap"
    }

    return {
        encoder: base_path / config / f"slide_features_{encoder}"
        for encoder, config in encoder_configs.items()
    }

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
        fontsize=8,
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
            fig, ax = plt.subplots(figsize=(7, 5))
            _plot_page(
                ax,
                global_emb,
                meta[col],
                f"Overall: {col}",
                is_numeric=False,
            )
            fig.suptitle(f"UMAP colored by {col}", fontsize=15, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for col in num_cols:
            fig, ax = plt.subplots(figsize=(7, 5))
            _plot_page(
                ax,
                global_emb,
                meta[col],
                f"Overall: {col}",
                is_numeric=True,
            )
            fig.suptitle(f"UMAP colored by {col}", fontsize=15, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig)
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
                    fig, ax = plt.subplots(figsize=(7, 5))
                    _plot_page(
                        ax,
                        emb,
                        sub[col],
                        f"{group_col}={g}: {col}",
                        is_numeric=False,
                    )
                    fig.suptitle(f"{group_col} = {g} | colored by {col}", fontsize=15, y=1.02)
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                for col in num_cols:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    _plot_page(
                        ax,
                        emb,
                        sub[col],
                        f"{group_col}={g}: {col}",
                        is_numeric=True,
                    )
                    fig.suptitle(f"{group_col} = {g} | colored by {col}", fontsize=15, y=1.02)
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"Saved PDF to: {out_pdf}")


def make_default_pdf_name(dataset_name: str, out_dir: Path) -> Path:
    return out_dir / f"umap_{dataset_name}.pdf"


def run_slide_encoder_umap(
    slide_encoder,
    h5_root,
    metadata_path,
    metadata_id_col,
    categorical_cols,
    plot_folder,
    sample_ids=None,
    numeric_cols=None,
    group_by_cols=None,
    feature_key="features",
):
    numeric_cols = numeric_cols or []
    group_by_cols = group_by_cols or []

    h5_root = Path(h5_root)
    metadata_path = Path(metadata_path)
    plot_folder = Path(plot_folder)

    out_pdf = plot_folder / f"umap_{slide_encoder}.pdf"

    print(f"\n{'=' * 60}")
    print(f"Slide encoder: {slide_encoder}")
    print(f"Features:      {h5_root}")
    print(f"Output:        {out_pdf}")

    if sample_ids is not None:
        print(f"Selected samples: {len(sample_ids)}")

    print(f"{'=' * 60}")

    # Load all embeddings or only selected samples
    features = load_features(
        h5_root=h5_root,
        feature_key=feature_key,
        sample_ids=sample_ids,
    )

    # Load metadata
    metadata = load_metadata(
        metadata_path=metadata_path,
        sample_id_col=metadata_id_col,
    )

    # Merge metadata
    meta = merge_metadata(
        features=features,
        metadata=metadata,
        metadata_id_col=metadata_id_col,
    )

    # UMAP + PDF
    make_umap_pdf(
        meta=meta,
        out_pdf=out_pdf,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        group_by_cols=group_by_cols,
        feature_col="features",
        recompute_group_umap=True,
    )

    return meta

def plot_matched_samples(
    ax,
    embedding,
    joint,
    matched_col="matched_xenium",
    source_col="source",
):
    # IDs that genuinely have an adjacent + Xenium pair
    matched_ids = joint.loc[
        (joint[source_col] == "Adjacent")
        & (joint["pair_status"] == "Paired"),
        matched_col,
    ].dropna().astype("string").unique()

    n = len(matched_ids)

    cmap_name = "tab20" if n <= 20 else "hsv"
    cmap = mpl.colormaps.get_cmap(cmap_name)

    colors = [
        cmap(i / max(1, n - 1))
        for i in range(n)
    ]

    color_map = dict(zip(matched_ids, colors))

    # ---------------------------------
    # Everything grey first
    # ---------------------------------
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=18,
        alpha=0.25,
        color="lightgrey",
        edgecolors="none",
        zorder=1,
    )

    # ---------------------------------
    # Matched pairs
    # ---------------------------------
    for sample_id in matched_ids:

        # Xenium = circle
        xenium_mask = (
            joint[matched_col].astype("string").eq(sample_id)
            & joint[source_col].eq("Xenium")
        ).fillna(False).to_numpy(dtype=bool)

        ax.scatter(
            embedding[xenium_mask, 0],
            embedding[xenium_mask, 1],
            s=45,
            marker="o",
            color=color_map[sample_id],
            label=str(sample_id),
            edgecolors="none",
            zorder=3,
        )

        # Adjacent = cross
        adjacent_mask = (
            joint[matched_col].astype("string").eq(sample_id)
            & joint[source_col].eq("Adjacent")
        ).fillna(False).to_numpy(dtype=bool)

        ax.scatter(
            embedding[adjacent_mask, 0],
            embedding[adjacent_mask, 1],
            s=55,
            marker="x",
            color=color_map[sample_id],
            linewidths=1.5,
            zorder=4,
        )

    ax.set_title("Matched Xenium–Adjacent samples")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # ---------------------------------
    # Sample ID legend
    # ---------------------------------
    if n > 0:
        ncol = (
            1 if n <= 20
            else 2 if n <= 50
            else 3
        )

        sample_legend = ax.legend(
            fontsize=7,
            markerscale=1.2,
            frameon=False,
            ncol=ncol,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            title="Matched Xenium Sample",
        )

        ax.add_artist(sample_legend)

    # ---------------------------------
    # Shape legend
    # ---------------------------------
    shape_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="black",
            markersize=6,
            label="Xenium",
        ),
        plt.Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="black",
            markersize=7,
            label="Adjacent",
        ),
    ]

    ax.legend(
        handles=shape_handles,
        title="Source",
        frameon=False,
        loc="lower right",
    )

def make_joint_umap_pdf(
    xenium_h5_root: Path,
    adjacent_h5_root: Path,
    xenium_metadata_path: Path,
    adjacent_metadata_path: Path,
    out_pdf: Path,
    xenium_sample_ids=None,
    categorical_cols=None,
    numeric_cols=None,
    feature_key: str = "features",
    xenium_id_col: str = "Sample_ID",
    adjacent_id_col: str = "adjacent_nec",
    adjacent_xenium_col: str = "xenium_sample",
):
    """
    Joint UMAP of Xenium and matched adjacent NEC slide embeddings.

    The resulting PDF contains:
      1. Source: Xenium vs Adjacent
      2. Metadata-colored UMAPs
      3. Matched Xenium sample ID

    For the matched-sample plot, an adjacent sample is assigned the
    Sample_ID of the Xenium sample to which it is matched.
    """

    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []

    # ========================================================
    # Load features
    # ========================================================

    print("\nLoading Xenium embeddings...")

    xenium_features = load_features(
        h5_root=Path(xenium_h5_root),
        feature_key=feature_key,
        sample_ids=xenium_sample_ids,
    )

    print("\nLoading adjacent NEC embeddings...")

    adjacent_features = load_features(
        h5_root=Path(adjacent_h5_root),
        feature_key=feature_key,
    )

    # ========================================================
    # Load metadata
    # ========================================================

    xenium_metadata = load_metadata(
        metadata_path=Path(xenium_metadata_path),
        sample_id_col=xenium_id_col,
    )

    adjacent_metadata = load_metadata(
        metadata_path=Path(adjacent_metadata_path),
        sample_id_col=adjacent_id_col,
    )

    # ========================================================
    # Merge metadata
    # ========================================================

    xenium = merge_metadata(
        features=xenium_features,
        metadata=xenium_metadata,
        metadata_id_col=xenium_id_col,
    )

    adjacent = merge_metadata(
        features=adjacent_features,
        metadata=adjacent_metadata,
        metadata_id_col=adjacent_id_col,
    )

    # ========================================================
    # Add source
    # ========================================================

    xenium["source"] = "Xenium"
    adjacent["source"] = "Adjacent"


    # ========================================================
    # Create common matched Xenium ID
    # ========================================================

    # Xenium samples identify themselves
    xenium["matched_xenium"] = xenium[xenium_id_col].astype("string")

    # Adjacent samples contain the ID of their matched Xenium sample
    adjacent["matched_xenium"] = adjacent[adjacent_xenium_col].astype("string")


    # ========================================================
    # Identify paired / unpaired samples
    # ========================================================

    # Xenium samples actually present in this joint analysis
    available_xenium = set(
        xenium["matched_xenium"]
        .dropna()
        .astype("string")
    )

    # Does each adjacent sample have a Xenium sample present?
    adjacent["pair_status"] = np.where(
        adjacent["matched_xenium"].isin(available_xenium),
        "Paired",
        "Unpaired",
    )

    # For Xenium, determine whether an adjacent counterpart exists
    paired_xenium = set(
        adjacent.loc[
            adjacent["pair_status"] == "Paired",
            "matched_xenium",
        ]
        .dropna()
        .astype("string")
    )

    xenium["pair_status"] = np.where(
        xenium["matched_xenium"].isin(paired_xenium),
        "Paired",
        "Unpaired",
    )


    # ========================================================
    # Combine
    # ========================================================

    joint = pd.concat(
        [
            xenium,
            adjacent,
        ],
        ignore_index=True,
        sort=False,
    )

    print(f"Xenium samples:   {len(xenium)}")
    print(f"Adjacent samples: {len(adjacent)}")
    print(f"Joint samples:    {len(joint)}")

    print("\nPair status:")
    print(
        joint.groupby(["source", "pair_status"])
        .size()
    )

    # ========================================================
    # Make common metadata columns
    # ========================================================

    # If adjacent metadata already contains disease/location
    # inherited from the matched Xenium sample, these columns
    # will line up automatically when concatenated.

    joint = pd.concat(
        [
            xenium,
            adjacent,
        ],
        ignore_index=True,
        sort=False,
    )

    print(f"Joint samples: {len(joint)}")

    # ========================================================
    # Joint UMAP
    # ========================================================

    X = np.stack(
        joint["features"].to_numpy()
    )

    embedding = compute_umap(X)

    # ========================================================
    # Save coordinates in dataframe
    # ========================================================

    joint["UMAP1"] = embedding[:, 0]
    joint["UMAP2"] = embedding[:, 1]

    # ========================================================
    # Columns
    # ========================================================

    cat_cols = [
        c
        for c in categorical_cols
        if c in joint.columns
    ]

    num_cols = [
        c
        for c in numeric_cols
        if c in joint.columns
    ]

    missing = (
        set(categorical_cols)
        | set(numeric_cols)
    ) - set(joint.columns)

    if missing:
        print(
            "Warning: columns not found:",
            sorted(missing),
        )

    # ========================================================
    # PDF
    # ========================================================

    out_pdf = Path(out_pdf)

    out_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PdfPages(out_pdf) as pdf:

        # ----------------------------------------------------
        # 1. Xenium vs Adjacent
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        plot_categorical(
            ax,
            embedding,
            joint["source"],
            "Xenium vs Adjacent NEC",
        )

        fig.tight_layout()

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ----------------------------------------------------
        # 2. Metadata
        # ----------------------------------------------------

        for col in cat_cols:

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            plot_categorical(
                ax,
                embedding,
                joint[col],
                f"UMAP colored by {col}",
            )

            fig.tight_layout()

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            plt.close(fig)

        for col in num_cols:

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            plot_numeric(
                ax,
                embedding,
                joint[col],
                f"UMAP colored by {col}",
            )

            fig.tight_layout()

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            plt.close(fig)

        # ----------------------------------------------------
        # 3. Matched Xenium sample
        # ----------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 7))

        plot_matched_samples(
            ax,
            embedding,
            joint,
            matched_col="matched_xenium",
        )

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved joint UMAP to: {out_pdf}")

    return joint

def make_joint_group_umap_pdf(
    group1_h5_root: Path,
    group2_h5_root: Path,
    group1_metadata_path: Path,
    group2_metadata_path: Path,
    out_pdf: Path,
    group1_label: str,
    group2_label: str,
    group1_id_col: str,
    group2_id_col: str,
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    group1_sample_ids=None,
    group2_sample_ids=None,
    feature_key: str = "features",
    match_col: str | None = None,
    random_state: int = 0,
    use_harmony: bool = False,
    harmony_n_pca: int = 50,
):
    """
    Joint UMAP of any two groups of slide embeddings.

    Shape indicates group:
        group 1 = circle
        group 2 = cross

    If use_harmony=True:
        StandardScaler -> PCA -> Harmony(source) -> UMAP

    Otherwise:
        Uses the existing compute_umap() function.

    Pages:
        1. Group membership
        2. Categorical metadata
        3. Numeric metadata
        4. Matched-sample plot, if match_col is supplied
    """

    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []

    # ========================================================
    # Load features
    # ========================================================

    print(f"\nLoading {group1_label} embeddings...")

    group1_features = load_features(
        h5_root=Path(group1_h5_root),
        feature_key=feature_key,
        sample_ids=group1_sample_ids,
    )

    print(f"\nLoading {group2_label} embeddings...")

    group2_features = load_features(
        h5_root=Path(group2_h5_root),
        feature_key=feature_key,
        sample_ids=group2_sample_ids,
    )

    # ========================================================
    # Load metadata
    # ========================================================

    group1_metadata = load_metadata(
        metadata_path=Path(group1_metadata_path),
        sample_id_col=group1_id_col,
    )

    group2_metadata = load_metadata(
        metadata_path=Path(group2_metadata_path),
        sample_id_col=group2_id_col,
    )

    # ========================================================
    # Merge metadata
    # ========================================================

    group1 = merge_metadata(
        features=group1_features,
        metadata=group1_metadata,
        metadata_id_col=group1_id_col,
    )

    group2 = merge_metadata(
        features=group2_features,
        metadata=group2_metadata,
        metadata_id_col=group2_id_col,
    )

    # ========================================================
    # Add group/source labels
    # ========================================================

    group1["source"] = group1_label
    group2["source"] = group2_label

    print(f"{group1_label}: {len(group1)}")
    print(f"{group2_label}: {len(group2)}")

    # ========================================================
    # Combine
    # ========================================================

    joint = pd.concat(
        [group1, group2],
        ignore_index=True,
        sort=False,
    )

    print(f"Joint: {len(joint)}")

    # ========================================================
    # Feature matrix
    # ========================================================

    X = np.stack(
        joint["features"].to_numpy()
    )

    # ========================================================
    # Dimensionality reduction
    # ========================================================

    if use_harmony:

        print(
            f"\nRunning Harmony integration across "
            f"{group1_label} and {group2_label}"
        )

        # -----------------------
        # Standardize
        # -----------------------
        X_scaled = StandardScaler().fit_transform(X)

        # -----------------------
        # PCA
        # -----------------------
        n_pca = min(
            harmony_n_pca,
            X_scaled.shape[1],
            X_scaled.shape[0] - 1,
        )

        if n_pca < 2:
            raise ValueError(
                "Not enough samples/components to run PCA + Harmony."
            )

        X_pca = PCA(
            n_components=n_pca,
            random_state=random_state,
        ).fit_transform(X_scaled)

        print(f"PCA shape: {X_pca.shape}")

        # -----------------------
        # Harmony
        # -----------------------
        harmony_meta = pd.DataFrame({
            "source": (
                joint["source"]
                .astype("string")
                .astype(str)
                .to_numpy()
            )
        })

        harmony_result = hm.run_harmony(
            X_pca,
            harmony_meta,
            vars_use=["source"],
            random_state=random_state,
        )

        # harmonypy output:
        # dimensions x samples
        X_harmony = harmony_result.Z_corr.T

        print(
            f"Harmony corrected shape: "
            f"{X_harmony.shape}"
        )

        # Save Harmony dimensions in joint dataframe
        for i in range(X_harmony.shape[1]):
            joint[f"Harmony_{i + 1}"] = X_harmony[:, i]

        # -----------------------
        # UMAP on Harmony output
        # -----------------------
        n_neighbors = min(
            15,
            max(2, X_harmony.shape[0] - 1),
        )

        embedding = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(X_harmony)

    else:

        print("\nRunning standard PCA + UMAP")

        embedding = compute_umap(
            X,
            random_state=random_state,
        )

    # ========================================================
    # Save UMAP coordinates
    # ========================================================

    joint["UMAP1"] = embedding[:, 0]
    joint["UMAP2"] = embedding[:, 1]

    # ========================================================
    # Validate plotting columns
    # ========================================================

    cat_cols = [
        c
        for c in categorical_cols
        if c in joint.columns
    ]

    num_cols = [
        c
        for c in numeric_cols
        if c in joint.columns
    ]

    missing_cols = (
        set(categorical_cols)
        | set(numeric_cols)
    ) - set(joint.columns)

    if missing_cols:
        print(
            "Warning: plotting columns not found:",
            sorted(missing_cols),
        )

    # ========================================================
    # Output PDF
    # ========================================================

    out_pdf = Path(out_pdf)

    out_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PdfPages(out_pdf) as pdf:

        # ====================================================
        # Page 1: group membership
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        group_styles = {
            group1_label: "o",
            group2_label: "x",
        }

        for group, marker in group_styles.items():

            mask = (
                joint["source"]
                .eq(group)
                .to_numpy(dtype=bool)
            )

            if marker == "o":

                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=35,
                    alpha=0.8,
                    marker="o",
                    label=group,
                    edgecolors="none",
                )

            else:

                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=45,
                    alpha=0.9,
                    marker="x",
                    label=group,
                    linewidths=1.5,
                )

        if use_harmony:
            ax.set_title(
                f"Harmony UMAP: "
                f"{group1_label} vs {group2_label}"
            )
        else:
            ax.set_title(
                f"{group1_label} vs {group2_label}"
            )

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

        ax.legend(
            title="Group",
            frameon=False,
        )

        fig.tight_layout()

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ====================================================
        # Categorical metadata
        #
        # Color = metadata
        # Shape = group
        # ====================================================

        for col in cat_cols:

            n_categories = (
                joint[col]
                .astype("string")
                .fillna("NA")
                .nunique()
            )

            # Increase page height when legend is long
            fig_height = max(
                6,
                1.2 + n_categories * 0.28,
            )

            fig = plt.figure(
                figsize=(12, fig_height)
            )

            gs = fig.add_gridspec(
                1,
                2,
                width_ratios=[3, 1.4],
                wspace=0.08,
            )

            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])

            plot_categorical_by_group(
                ax=ax,
                legend_ax=legend_ax,
                embedding=embedding,
                values=joint[col],
                groups=joint["source"],
                group1_label=group1_label,
                group2_label=group2_label,
                title=f"UMAP colored by {col}",
            )

            pdf.savefig(fig)

            plt.close(fig)

        # ====================================================
        # Numeric metadata
        # ====================================================

        for col in num_cols:

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            plot_numeric_by_group(
                ax=ax,
                embedding=embedding,
                values=joint[col],
                groups=joint["source"],
                group1_label=group1_label,
                group2_label=group2_label,
                title=f"UMAP colored by {col}",
            )

            fig.tight_layout()

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            plt.close(fig)

        # ====================================================
        # Matched samples
        # ====================================================

        if match_col is not None:

            if match_col not in joint.columns:

                print(
                    f"Skipping matched plot: "
                    f"{match_col!r} not found"
                )

            else:

                fig, ax = plt.subplots(
                    figsize=(12, 8)
                )

                plot_matched_groups(
                    ax=ax,
                    embedding=embedding,
                    joint=joint,
                    match_col=match_col,
                    source_col="source",
                    group1_label=group1_label,
                    group2_label=group2_label,
                )

                fig.tight_layout()

                pdf.savefig(
                    fig,
                    bbox_inches="tight",
                )

                plt.close(fig)

    print(
        f"Saved PDF to: {out_pdf}"
    )

    return joint



def plot_categorical_by_group(
    ax,
    legend_ax,
    embedding,
    values,
    groups,
    group1_label,
    group2_label,
    title,
):
    """
    Color = metadata category
    Shape = group

    ax        = UMAP axes
    legend_ax = dedicated axes for legends
    """

    vals = (
        pd.Series(values)
        .astype("string")
        .fillna("NA")
        .reset_index(drop=True)
    )

    groups = (
        pd.Series(groups)
        .astype("string")
        .fillna("NA")
        .reset_index(drop=True)
    )

    cats = pd.unique(vals)
    n = len(cats)

    # -----------------------
    # Colors
    # -----------------------
    cmap_name = "tab20" if n <= 20 else "hsv"
    cmap = mpl.colormaps.get_cmap(cmap_name)

    colors = [
        cmap(i / max(1, n - 1))
        for i in range(n)
    ]

    color_map = dict(zip(cats, colors))

    # -----------------------
    # Group markers
    # -----------------------
    markers = {
        group1_label: "o",
        group2_label: "x",
    }

    # -----------------------
    # Plot UMAP
    # -----------------------
    for cat in cats:

        for group, marker in markers.items():

            mask = (
                vals.eq(cat)
                & groups.eq(group)
            ).fillna(False).to_numpy(dtype=bool)

            if not mask.any():
                continue

            if marker == "o":
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=35,
                    alpha=0.85,
                    color=color_map[cat],
                    marker="o",
                    edgecolors="none",
                )

            else:
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=45,
                    alpha=0.9,
                    color=color_map[cat],
                    marker="x",
                    linewidths=1.5,
                )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # ========================================================
    # Legend panel
    # ========================================================
    legend_ax.axis("off")

    # -----------------------
    # Color legend
    # -----------------------
    color_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[cat],
            markeredgecolor="none",
            markersize=6,
            label=str(cat),
        )
        for cat in cats
    ]

    legend_title = title.replace(
        "UMAP colored by ",
        ""
    )

    color_legend = legend_ax.legend(
        handles=color_handles,
        title=legend_title,
        fontsize=7,
        title_fontsize=9,
        frameon=False,

        # IMPORTANT:
        # keep legend vertical so it cannot run off right side
        ncol=1,

        loc="upper left",
        bbox_to_anchor=(0, 1),
        borderaxespad=0,

        handletextpad=0.5,
        labelspacing=0.5,
    )

    legend_ax.add_artist(color_legend)

    # -----------------------
    # Group legend
    # -----------------------
    shape_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="none",
            markersize=7,
            label=group1_label,
        ),
        plt.Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="black",
            markersize=7,
            markeredgewidth=1.5,
            label=group2_label,
        ),
    ]

    legend_ax.legend(
        handles=shape_handles,
        title="Group",
        fontsize=7,
        title_fontsize=9,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 0),
        borderaxespad=0,
    )

def plot_numeric_by_group(
    ax,
    embedding,
    values,
    groups,
    group1_label,
    group2_label,
    title,
):
    vals = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    groups = (
        pd.Series(groups)
        .astype("string")
    )

    valid = vals.notna()

    if not valid.any():
        ax.set_title(title)
        return

    norm = mpl.colors.Normalize(
        vmin=vals[valid].min(),
        vmax=vals[valid].max(),
    )

    cmap = mpl.colormaps["viridis"]

    markers = {
        group1_label: "o",
        group2_label: "x",
    }

    for group, marker in markers.items():

        mask = (
            valid
            & groups.eq(group)
        ).to_numpy(dtype=bool)

        if not mask.any():
            continue

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=vals[mask],
            cmap=cmap,
            norm=norm,
            s=30,
            alpha=0.9,
            marker=marker,
            edgecolors="none"
            if marker == "o"
            else None,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    plt.colorbar(
        sm,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    shape_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="black",
            label=group1_label,
        ),
        plt.Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="black",
            label=group2_label,
        ),
    ]

    ax.legend(
        handles=shape_handles,
        title="Group",
        frameon=False,
    )

def plot_matched_groups(
    ax,
    embedding,
    joint,
    match_col,
    source_col,
    group1_label,
    group2_label,
):
    group1_ids = set(
        joint.loc[
            joint[source_col] == group1_label,
            match_col,
        ]
        .dropna()
        .astype("string")
    )

    group2_ids = set(
        joint.loc[
            joint[source_col] == group2_label,
            match_col,
        ]
        .dropna()
        .astype("string")
    )

    # Only IDs present in BOTH groups
    matched_ids = sorted(
        group1_ids & group2_ids
    )

    n = len(matched_ids)

    cmap_name = "tab20" if n <= 20 else "hsv"
    cmap = mpl.colormaps.get_cmap(cmap_name)

    colors = [
        cmap(i / max(1, n - 1))
        for i in range(n)
    ]

    color_map = dict(
        zip(matched_ids, colors)
    )

    # Everything else grey
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=18,
        alpha=0.25,
        color="lightgrey",
        edgecolors="none",
        zorder=1,
    )

    markers = {
        group1_label: "o",
        group2_label: "x",
    }

    for sample_id in matched_ids:

        for group, marker in markers.items():

            mask = (
                joint[match_col]
                .astype("string")
                .eq(sample_id)
                & joint[source_col].eq(group)
            ).fillna(False).to_numpy(dtype=bool)

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=45,
                marker=marker,
                color=color_map[sample_id],
                edgecolors="none"
                if marker == "o"
                else None,
                linewidths=1.5,
                zorder=3,
            )

    ax.set_title(
        f"Matched {group1_label}–{group2_label} samples"
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # Sample ID legend
    sample_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[sample_id],
            markeredgecolor="none",
            label=str(sample_id),
        )
        for sample_id in matched_ids
    ]

    sample_legend = ax.legend(
        handles=sample_handles,
        title="Sample ID",
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.add_artist(sample_legend)

    # Group shape legend
    shape_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="black",
            label=group1_label,
        ),
        plt.Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="black",
            label=group2_label,
        ),
    ]

    ax.legend(
        handles=shape_handles,
        title="Group",
        frameon=False,
        loc="lower right",
    )



## three groups
def plot_categorical_by_three_groups(
    ax,
    legend_ax,
    embedding,
    values,
    groups,
    group_labels,
    title,
):
    """
    Color = metadata category
    Shape = group

    Expected group_labels:
        [group1, group2, group3]
    """

    vals = (
        pd.Series(values)
        .astype("string")
        .fillna("NA")
        .reset_index(drop=True)
    )

    groups = (
        pd.Series(groups)
        .astype("string")
        .fillna("NA")
        .reset_index(drop=True)
    )

    cats = pd.unique(vals)
    n = len(cats)

    # -----------------------
    # Colors
    # -----------------------
    cmap_name = "tab20" if n <= 20 else "hsv"
    cmap = mpl.colormaps.get_cmap(cmap_name)

    colors = [
        cmap(i / max(1, n - 1))
        for i in range(n)
    ]

    color_map = dict(zip(cats, colors))

    # -----------------------
    # Group markers
    # -----------------------
    marker_list = ["o", "x", "^"]

    markers = dict(
        zip(group_labels, marker_list)
    )

    # -----------------------
    # Plot
    # -----------------------
    for cat in cats:

        for group, marker in markers.items():

            mask = (
                vals.eq(cat)
                & groups.eq(group)
            ).fillna(False).to_numpy(dtype=bool)

            if not mask.any():
                continue

            if marker == "x":
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=45,
                    alpha=0.9,
                    color=color_map[cat],
                    marker=marker,
                    linewidths=1.5,
                )

            else:
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=38,
                    alpha=0.85,
                    color=color_map[cat],
                    marker=marker,
                    edgecolors="none",
                )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # ========================================================
    # Legend panel
    # ========================================================

    legend_ax.axis("off")

    # -----------------------
    # Color legend
    # -----------------------
    color_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[cat],
            markeredgecolor="none",
            markersize=6,
            label=str(cat),
        )
        for cat in cats
    ]

    legend_title = title.replace(
        "UMAP colored by ",
        ""
    )

    color_legend = legend_ax.legend(
        handles=color_handles,
        title=legend_title,
        fontsize=7,
        title_fontsize=9,
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        borderaxespad=0,
        handletextpad=0.5,
        labelspacing=0.5,
    )

    legend_ax.add_artist(color_legend)

    # -----------------------
    # Group / shape legend
    # -----------------------
    shape_handles = []

    for group, marker in markers.items():

        if marker == "x":
            handle = plt.Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                color="black",
                markersize=7,
                markeredgewidth=1.5,
                label=group,
            )

        else:
            handle = plt.Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="none",
                color="black",
                markersize=7,
                label=group,
            )

        shape_handles.append(handle)

    legend_ax.legend(
        handles=shape_handles,
        title="Group",
        fontsize=7,
        title_fontsize=9,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 0),
        borderaxespad=0,
    )

def plot_numeric_by_three_groups(
    ax,
    embedding,
    values,
    groups,
    group_labels,
    title,
):
    vals = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).reset_index(drop=True)

    groups = (
        pd.Series(groups)
        .astype("string")
        .reset_index(drop=True)
    )

    valid = vals.notna()

    if not valid.any():
        ax.set_title(title)
        return

    norm = mpl.colors.Normalize(
        vmin=vals[valid].min(),
        vmax=vals[valid].max(),
    )

    cmap = mpl.colormaps["viridis"]

    markers = dict(
        zip(
            group_labels,
            ["o", "x", "^"],
        )
    )

    for group, marker in markers.items():

        mask = (
            valid
            & groups.eq(group)
        ).to_numpy(dtype=bool)

        if not mask.any():
            continue

        kwargs = {}

        if marker != "x":
            kwargs["edgecolors"] = "none"

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=vals[mask],
            cmap=cmap,
            norm=norm,
            s=38,
            alpha=0.9,
            marker=marker,
            **kwargs,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    plt.colorbar(
        sm,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    shape_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            color="black",
            label=group,
        )
        for group, marker in markers.items()
    ]

    ax.legend(
        handles=shape_handles,
        title="Group",
        frameon=False,
    )

def make_joint_three_group_umap_pdf(
    group1_h5_root: Path,
    group2_h5_root: Path,
    group3_h5_root: Path,

    group1_metadata_path: Path,
    group2_metadata_path: Path,
    group3_metadata_path: Path,

    out_pdf: Path,

    group1_label: str,
    group2_label: str,
    group3_label: str,

    group1_id_col: str,
    group2_id_col: str,
    group3_id_col: str,

    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,

    group1_sample_ids=None,
    group2_sample_ids=None,
    group3_sample_ids=None,

    feature_key: str = "features",

    random_state: int = 0,

    use_harmony: bool = False,
    harmony_n_pca: int = 50,
):
    """
    Joint UMAP of three groups.

    Shape:
        group1 = circle
        group2 = cross
        group3 = triangle

    If use_harmony=True:
        StandardScaler -> PCA -> Harmony(source) -> UMAP
    """

    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []

    group_labels = [
        group1_label,
        group2_label,
        group3_label,
    ]

    # ========================================================
    # Load features
    # ========================================================

    print(f"\nLoading {group1_label} embeddings...")

    group1_features = load_features(
        h5_root=Path(group1_h5_root),
        feature_key=feature_key,
        sample_ids=group1_sample_ids,
    )

    print(f"\nLoading {group2_label} embeddings...")

    group2_features = load_features(
        h5_root=Path(group2_h5_root),
        feature_key=feature_key,
        sample_ids=group2_sample_ids,
    )

    print(f"\nLoading {group3_label} embeddings...")

    group3_features = load_features(
        h5_root=Path(group3_h5_root),
        feature_key=feature_key,
        sample_ids=group3_sample_ids,
    )

    # ========================================================
    # Metadata
    # ========================================================

    group1_metadata = load_metadata(
        metadata_path=Path(group1_metadata_path),
        sample_id_col=group1_id_col,
    )

    group2_metadata = load_metadata(
        metadata_path=Path(group2_metadata_path),
        sample_id_col=group2_id_col,
    )

    group3_metadata = load_metadata(
        metadata_path=Path(group3_metadata_path),
        sample_id_col=group3_id_col,
    )

    # ========================================================
    # Merge metadata
    # ========================================================

    group1 = merge_metadata(
        features=group1_features,
        metadata=group1_metadata,
        metadata_id_col=group1_id_col,
    )

    group2 = merge_metadata(
        features=group2_features,
        metadata=group2_metadata,
        metadata_id_col=group2_id_col,
    )

    group3 = merge_metadata(
        features=group3_features,
        metadata=group3_metadata,
        metadata_id_col=group3_id_col,
    )

    # ========================================================
    # Source
    # ========================================================

    group1["source"] = group1_label
    group2["source"] = group2_label
    group3["source"] = group3_label

    print(f"{group1_label}: {len(group1)}")
    print(f"{group2_label}: {len(group2)}")
    print(f"{group3_label}: {len(group3)}")

    # ========================================================
    # Combine
    # ========================================================

    joint = pd.concat(
        [
            group1,
            group2,
            group3,
        ],
        ignore_index=True,
        sort=False,
    )

    print(f"Joint: {len(joint)}")

    # ========================================================
    # Feature matrix
    # ========================================================

    X = np.stack(
        joint["features"].to_numpy()
    )

    # ========================================================
    # UMAP / Harmony UMAP
    # ========================================================

    if use_harmony:

        print(
            "\nRunning Harmony integration across: "
            + ", ".join(group_labels)
        )

        X_scaled = StandardScaler().fit_transform(X)

        n_pca = min(
            harmony_n_pca,
            X_scaled.shape[1],
            X_scaled.shape[0] - 1,
        )

        if n_pca < 2:
            raise ValueError(
                "Not enough samples/components for PCA + Harmony."
            )

        X_pca = PCA(
            n_components=n_pca,
            random_state=random_state,
        ).fit_transform(X_scaled)

        print(f"PCA shape: {X_pca.shape}")

        harmony_meta = pd.DataFrame({
            "source": (
                joint["source"]
                .astype("string")
                .astype(str)
                .to_numpy()
            )
        })

        harmony_result = hm.run_harmony(
            X_pca,
            harmony_meta,
            vars_use=["source"],
            random_state=random_state,
        )

        X_harmony = harmony_result.Z_corr.T

        print(
            f"Harmony corrected shape: "
            f"{X_harmony.shape}"
        )

        embedding = umap.UMAP(
            n_neighbors=min(
                15,
                max(2, X_harmony.shape[0] - 1),
            ),
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(X_harmony)

    else:

        embedding = compute_umap(
            X,
            random_state=random_state,
        )

    joint["UMAP1"] = embedding[:, 0]
    joint["UMAP2"] = embedding[:, 1]

    # ========================================================
    # Plot columns
    # ========================================================

    cat_cols = [
        c
        for c in categorical_cols
        if c in joint.columns
    ]

    num_cols = [
        c
        for c in numeric_cols
        if c in joint.columns
    ]

    # ========================================================
    # PDF
    # ========================================================

    out_pdf = Path(out_pdf)

    out_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markers = {
        group1_label: "o",
        group2_label: "x",
        group3_label: "^",
    }

    with PdfPages(out_pdf) as pdf:

        # ====================================================
        # Page 1 — groups
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        for group, marker in markers.items():

            mask = (
                joint["source"]
                .eq(group)
                .to_numpy(dtype=bool)
            )

            if marker == "x":

                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=50,
                    alpha=0.9,
                    marker=marker,
                    label=group,
                    linewidths=1.5,
                )

            else:

                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=40,
                    alpha=0.85,
                    marker=marker,
                    label=group,
                    edgecolors="none",
                )

        if use_harmony:
            ax.set_title(
                "Harmony UMAP: "
                + " vs ".join(group_labels)
            )
        else:
            ax.set_title(
                " vs ".join(group_labels)
            )

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

        ax.legend(
            title="Group",
            frameon=False,
        )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ====================================================
        # Categorical metadata
        # ====================================================

        for col in cat_cols:

            n_categories = (
                joint[col]
                .astype("string")
                .fillna("NA")
                .nunique()
            )

            fig_height = max(
                6,
                1.2 + n_categories * 0.28,
            )

            fig = plt.figure(
                figsize=(12, fig_height)
            )

            gs = fig.add_gridspec(
                1,
                2,
                width_ratios=[3, 1.4],
                wspace=0.08,
            )

            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])

            plot_categorical_by_three_groups(
                ax=ax,
                legend_ax=legend_ax,
                embedding=embedding,
                values=joint[col],
                groups=joint["source"],
                group_labels=group_labels,
                title=f"UMAP colored by {col}",
            )

            pdf.savefig(fig)
            plt.close(fig)

        # ====================================================
        # Numeric metadata
        # ====================================================

        for col in num_cols:

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            plot_numeric_by_three_groups(
                ax=ax,
                embedding=embedding,
                values=joint[col],
                groups=joint["source"],
                group_labels=group_labels,
                title=f"UMAP colored by {col}",
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(
        f"Saved PDF to: {out_pdf}"
    )

    return joint

