import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD + MATCH SAMPLE
# ============================================================

def load_and_match_sample(
    sample,
    image_root="/project/gutdecoder/kxu/hest/eval/ST_data_emb/XeniumPR1/hoptimus1",
    gene_root="/project/gutdecoder/kxu/hest/eval/data/XeniumPR1/adata",
):
    """
    Load image embeddings and gene expression for one sample,
    then barcode-align them.

    Returns
    -------
    img : AnnData
        Image embeddings.

    gene_subset : AnnData
        Gene expression restricted and reordered to image barcodes.
    """

    img_path = os.path.join(
        image_root,
        f"{sample}.h5"
    )

    gene_path = os.path.join(
        gene_root,
        f"{sample}.h5ad"
    )

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    if not os.path.exists(gene_path):
        raise FileNotFoundError(gene_path)

    # --------------------------------------------------------
    # Image embeddings
    # --------------------------------------------------------
    with h5py.File(img_path, "r") as f:
        embeddings = f["embeddings"][:]
        coords = f["coords"][:]
        barcodes = f["barcodes"][:].reshape(-1)

    barcodes = np.array([
        x.decode() if isinstance(x, bytes) else str(x)
        for x in barcodes
    ])

    # --------------------------------------------------------
    # Gene expression
    # --------------------------------------------------------
    gene = sc.read_h5ad(gene_path)

    # Make sure all image barcodes exist in gene object
    missing = ~np.isin(barcodes, gene.obs_names)

    if missing.any():
        missing_barcodes = barcodes[missing]
        raise ValueError(
            f"{missing.sum()} image barcodes are missing "
            f"from gene AnnData. Example: {missing_barcodes[:5]}"
        )

    # Reorder gene object to EXACT image order
    gene_subset = gene[barcodes].copy()

    # --------------------------------------------------------
    # Check spatial correspondence
    # --------------------------------------------------------
    gene_coords = np.rint(
        gene_subset.obsm["spatial"]
    ).astype(int)

    image_coords = np.rint(coords).astype(int)

    print(f"Sample: {sample}")
    print(f"Image embeddings: {embeddings.shape}")
    print(f"Gene data:        {gene_subset.shape}")

    # --------------------------------------------------------
    # Image AnnData
    # --------------------------------------------------------
    img = sc.AnnData(
        embeddings
    )

    img.obs_names = barcodes
    img.obsm["spatial"] = coords

    return img, gene_subset


# ============================================================
# 2. CLUSTER IMAGE EMBEDDING
# ============================================================

def cluster_image_embedding(
    img,
    n_neighbors=15,
    n_pcs=30,
    resolution=0.5,
):
    """
    PCA -> neighbors -> UMAP -> Leiden for image embeddings.
    """

    # PCA on image embeddings
    sc.pp.scale(
        img,
        zero_center=True,
        max_value=10,
    )

    sc.tl.pca(
        img,
        n_comps=n_pcs,
    )

    # Build graph in PCA space
    sc.pp.neighbors(
        img,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
    )

    # UMAP from the same graph
    sc.tl.umap(img)

    # Leiden clustering
    sc.tl.leiden(
        img,
        resolution=resolution,
        key_added="image_cluster",
    )

    print(
        f"Image clusters: "
        f"{img.obs['image_cluster'].nunique()}"
    )

    return img


# ============================================================
# 3. CLUSTER GENE EXPRESSION
# ============================================================

def cluster_gene_expression(
    gene_subset,
    n_hvg=2000,
    n_pcs=30,
    n_neighbors=15,
    resolution=0.5,
):
    """
    Normalize, HVG select, PCA, UMAP and Leiden cluster
    gene expression.

    Original gene_subset is not modified.
    """

    gene_clust = gene_subset.copy()

    # sc.pp.normalize_total(
    #     gene_clust,
    #     target_sum=1e4,
    # )

    sc.pp.log1p(gene_clust)

    sc.pp.highly_variable_genes(
        gene_clust,
        n_top_genes=n_hvg,
    )

    gene_clust = gene_clust[
        :,
        gene_clust.var["highly_variable"]
    ].copy()

    sc.pp.scale(
        gene_clust,
        max_value=10,
    )

    sc.tl.pca(
        gene_clust,
        n_comps=n_pcs,
    )

    sc.pp.neighbors(
        gene_clust,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
    )

    sc.tl.umap(gene_clust)

    sc.tl.leiden(
        gene_clust,
        resolution=resolution,
        key_added="gene_cluster",
    )

    print(
        f"Gene clusters: "
        f"{gene_clust.obs['gene_cluster'].nunique()}"
    )

    return gene_clust


# ============================================================
# 4. MATCH CLUSTERS
# ============================================================

def match_clusters_unique_with_fallback(
    image_adata,
    gene_adata,
    image_cluster_key="image_cluster",
    gene_cluster_key="gene_cluster",
    threshold=0.30,
):
    """
    Match gene clusters to image clusters using patch-level
    spatial overlap.

    Rules
    -----
    - Original gene clustering is never modified.
    - Each image cluster can be assigned to at most one gene cluster.
    - If two gene clusters want the same image cluster, the one
      with larger overlap gets the image-cluster label.
    - Other gene clusters receive new numeric labels beginning
      after the largest image-cluster number.
    """

    if image_adata.n_obs != gene_adata.n_obs:
        raise ValueError(
            "Image and gene objects have different numbers "
            "of observations."
        )

    image_labels = (
        image_adata.obs[image_cluster_key]
        .astype(str)
        .values
    )

    gene_labels = (
        gene_adata.obs[gene_cluster_key]
        .astype(str)
        .values
    )

    image_clusters = np.unique(image_labels)
    gene_clusters = np.unique(gene_labels)

    # --------------------------------------------------------
    # Count overlap
    # --------------------------------------------------------
    overlap = pd.DataFrame(
        0,
        index=gene_clusters,
        columns=image_clusters,
        dtype=int,
    )

    for gc in gene_clusters:
        for ic in image_clusters:
            overlap.loc[gc, ic] = np.sum(
                (gene_labels == gc) &
                (image_labels == ic)
            )

    gene_sizes = pd.Series(
        gene_labels
    ).value_counts()

    overlap_fraction = overlap.div(
        gene_sizes,
        axis=0,
    )

    # --------------------------------------------------------
    # All possible candidate matches
    # --------------------------------------------------------
    candidates = []

    for gc in gene_clusters:
        for ic in image_clusters:

            candidates.append({
                "gene_cluster": gc,
                "image_cluster": ic,
                "overlap_fraction": (
                    overlap_fraction.loc[gc, ic]
                ),
                "overlap_n": (
                    overlap.loc[gc, ic]
                ),
            })

    candidates_df = pd.DataFrame(candidates)

    # Strongest matches first
    candidates_df = candidates_df.sort_values(
        "overlap_fraction",
        ascending=False,
    )

    # --------------------------------------------------------
    # Unique image assignment
    # --------------------------------------------------------
    assigned_gene = set()
    assigned_image = set()

    cluster_mapping = {}

    for _, row in candidates_df.iterrows():

        gc = row["gene_cluster"]
        ic = row["image_cluster"]
        frac = row["overlap_fraction"]

        if frac < threshold:
            continue

        if gc in assigned_gene:
            continue

        if ic in assigned_image:
            continue

        cluster_mapping[gc] = ic

        assigned_gene.add(gc)
        assigned_image.add(ic)

    # --------------------------------------------------------
    # New numeric labels for remaining gene clusters
    # --------------------------------------------------------
    image_numbers = [
        int(x)
        for x in image_clusters
        if str(x).isdigit()
    ]

    next_label = (
        max(image_numbers)
        if image_numbers
        else -1
    ) + 1

    for gc in gene_clusters:

        if gc not in cluster_mapping:

            cluster_mapping[gc] = str(
                next_label
            )

            next_label += 1

    # --------------------------------------------------------
    # Mapping table
    # --------------------------------------------------------
    mapping_rows = []

    for gc in gene_clusters:

        best_ic = (
            overlap_fraction.loc[gc].idxmax()
        )

        best_frac = (
            overlap_fraction.loc[gc].max()
        )

        best_n = overlap.loc[
            gc,
            best_ic
        ]

        mapped = cluster_mapping[gc]

        mapping_rows.append({
            "gene_cluster": gc,
            "best_image_cluster": best_ic,
            "best_overlap_fraction": best_frac,
            "best_overlap_n": best_n,
            "mapped_cluster": mapped,
            "matched_to_image": (
                mapped in image_clusters
            ),
        })

    mapping_df = pd.DataFrame(
        mapping_rows
    )

    return mapping_df, cluster_mapping


# ============================================================
# 5. MAKE ONE SHARED PALETTE
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


def make_shared_cluster_palette(
    gene_subset,
    image_cluster_key,
    mapped_gene_labels,
):
    """
    Create a single shared cluster palette.

    The image cluster colours are generated once using Scanpy's
    standard categorical palette, then reused everywhere.

    New gene-only mapped clusters receive colours that are not
    already used.

    Returns
    -------
    palette : dict
        cluster label -> colour
    """

    # --------------------------------------------------------
    # Image cluster categories
    # --------------------------------------------------------
    gene_subset.obs[image_cluster_key] = (
        gene_subset.obs[image_cluster_key]
        .astype("category")
    )

    image_categories = [
        str(x)
        for x in gene_subset.obs[
            image_cluster_key
        ].cat.categories
    ]

    # --------------------------------------------------------
    # Generate the same default categorical palette Scanpy
    # normally uses for categorical observations
    # --------------------------------------------------------
    image_colors = sc.pl.palettes.default_20[
        :len(image_categories)
    ]

    palette = {
        cluster: color
        for cluster, color in zip(
            image_categories,
            image_colors,
        )
    }

    # --------------------------------------------------------
    # Mapped gene labels
    # --------------------------------------------------------
    mapped_gene_labels = [
        str(x)
        for x in mapped_gene_labels
    ]

    all_labels = sorted(
        set(image_categories) |
        set(mapped_gene_labels),
        key=lambda x: int(x),
    )

    new_labels = [
        label
        for label in all_labels
        if label not in palette
    ]

    # --------------------------------------------------------
    # Generate additional colours for new gene clusters
    # --------------------------------------------------------
    # Use a large colour space so new labels are distinct
    extra_cmap = plt.get_cmap(
        "gist_ncar",
        max(256, len(new_labels) + 1),
    )

    used = {
        to_hex(color)
        for color in palette.values()
    }

    color_idx = 0

    for label in new_labels:

        while color_idx < extra_cmap.N:

            color = to_hex(
                extra_cmap(color_idx)
            )

            color_idx += 1

            if color not in used:
                palette[label] = color
                used.add(color)
                break

        else:
            raise ValueError(
                "Could not find enough unique colours."
            )

    return palette

# ============================================================
# 6. SET PALETTE FOR SCANPY SPATIAL PLOTS
# ============================================================

def set_scanpy_palette(
    adata,
    cluster_key,
    palette,
):
    """
    Force Scanpy to use the supplied cluster palette.
    """

    adata.obs[cluster_key] = (
        adata.obs[cluster_key]
        .astype("category")
    )

    categories = [
        str(x)
        for x in adata.obs[
            cluster_key
        ].cat.categories
    ]

    adata.uns[
        f"{cluster_key}_colors"
    ] = [
        palette[str(x)]
        for x in categories
    ]


# ============================================================
# 7. PLOT SPATIAL CLUSTERS
# ============================================================

def plot_spatial_clusters(
    gene_subset,
    palette,
    image_cluster_key="image_cluster",
    gene_mapped_key="gene_cluster_mapped",
    library_id="ST",
    img_key="downscaled_fullres",
    spot_size=100,
    alpha=0.85,
    output_path=None,
):
    """
    Plot image clusters and mapped gene clusters on H&E.
    """

    set_scanpy_palette(
        gene_subset,
        image_cluster_key,
        palette,
    )

    set_scanpy_palette(
        gene_subset,
        gene_mapped_key,
        palette,
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 7),
    )

    sc.pl.spatial(
        gene_subset,
        color=image_cluster_key,
        library_id=library_id,
        img_key=img_key,
        spot_size=spot_size,
        alpha=alpha,
        ax=ax[0],
        show=False,
    )

    ax[0].set_title(
        "Image embedding clusters"
    )

    sc.pl.spatial(
        gene_subset,
        color=gene_mapped_key,
        library_id=library_id,
        img_key=img_key,
        spot_size=spot_size,
        alpha=alpha,
        ax=ax[1],
        show=False,
    )

    ax[1].set_title(
        "Gene clusters — mapped"
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return fig, ax


# ============================================================
# 8. PLOT UMAPS
# ============================================================

def plot_matched_umaps(
    image_adata,
    gene_adata,
    palette,
    image_cluster_key="image_cluster",
    gene_cluster_key="gene_cluster",
    gene_spatial_adata=None,
    output_path=None,
    point_size=12,
    alpha=0.85,
    figsize=(12, 5),
):
    """
    Plot image and gene UMAPs using EXACTLY the same colours
    as the spatial plots.

    gene_cluster_key refers to the original gene clustering.
    gene_spatial_adata contains gene_cluster_mapped.
    """

    if gene_spatial_adata is None:
        raise ValueError(
            "gene_spatial_adata is required."
        )

    image_labels = (
        image_adata.obs[
            image_cluster_key
        ].astype(str).values
    )

    gene_original_labels = (
        gene_spatial_adata.obs[
            gene_cluster_key
        ].astype(str).values
    )

    gene_mapped_labels = (
        gene_spatial_adata.obs[
            "gene_cluster_mapped"
        ].astype(str).values
    )

    image_colors = [
        palette[str(x)]
        for x in image_labels
    ]

    gene_colors = [
        palette[str(x)]
        for x in gene_mapped_labels
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    ax[0].scatter(
        image_adata.obsm["X_umap"][:, 0],
        image_adata.obsm["X_umap"][:, 1],
        c=image_colors,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    ax[0].set_title(
        "Image embedding"
    )

    ax[0].set_xlabel("UMAP1")
    ax[0].set_ylabel("UMAP2")

    ax[1].scatter(
        gene_adata.obsm["X_umap"][:, 0],
        gene_adata.obsm["X_umap"][:, 1],
        c=gene_colors,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    ax[1].set_title(
        "Gene expression — mapped clusters"
    )

    ax[1].set_xlabel("UMAP1")
    ax[1].set_ylabel("UMAP2")

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------
    all_labels = sorted(
        palette.keys(),
        key=lambda x: int(x),
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[label],
            markeredgecolor="none",
            markersize=7,
            label=label,
        )
        for label in all_labels
        if (
            label in set(image_labels)
            or label in set(gene_mapped_labels)
        )
    ]

    fig.legend(
        handles=handles,
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return fig, ax


# ============================================================
# 9. WRAPPER
# ============================================================

def plot_embedding(
    sample,
    output_dir="plot",
    image_root=(
        "/project/gutdecoder/kxu/hest/eval/"
        "ST_data_emb/XeniumPR1/hoptimus1"
    ),
    gene_root=(
        "/project/gutdecoder/kxu/hest/eval/"
        "data/XeniumPR1/adata"
    ),
    threshold=0.30,
    n_neighbors=15,
    image_resolution=0.5,
    gene_resolution=0.5,
    n_hvg=2000,
    n_pcs=30,
    spot_size=100,
):
    """
    Complete image/gene clustering, spatial matching, UMAP and
    H&E plotting pipeline for one sample.
    """

    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # 1. Load + barcode-align
    # ========================================================
    img, gene_subset = load_and_match_sample(
        sample=sample,
        image_root=image_root,
        gene_root=gene_root,
    )

    # ========================================================
    # 2. Image clustering + UMAP
    # ========================================================
    img = cluster_image_embedding(
        img,
        n_neighbors=n_neighbors,
        resolution=image_resolution,
    )

    # ========================================================
    # 3. Gene clustering + UMAP
    # ========================================================
    gene_clust = cluster_gene_expression(
        gene_subset,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        resolution=gene_resolution,
    )

    # Make absolutely sure the expected column exists
    if "gene_cluster" not in gene_clust.obs.columns:
        raise KeyError(
            "gene_cluster was not created in gene_clust. "
            f"Available columns: {gene_clust.obs.columns.tolist()}"
        )

    # ========================================================
    # 4. Transfer image cluster labels to spatial object
    # ========================================================
    gene_subset.obs["image_cluster"] = (
        img.obs["image_cluster"]
        .astype(str)
        .values
    )

    # Transfer original gene cluster labels
    gene_subset.obs["gene_cluster_original"] = (
        gene_clust.obs["gene_cluster"]
        .astype(str)
        .values
    )

    # ========================================================
    # 5. Match clusters
    # ========================================================
    mapping_df, cluster_mapping = (
        match_clusters_unique_with_fallback(
            image_adata=img,
            gene_adata=gene_clust,
            image_cluster_key="image_cluster",
            gene_cluster_key="gene_cluster",
            threshold=threshold,
        )
    )


    # ========================================================
    # 6. Create mapped gene labels
    # ========================================================
    gene_subset.obs["gene_cluster_mapped"] = (
        gene_subset.obs["gene_cluster_original"]
        .map(cluster_mapping)
        .astype(str)
    )

    # ========================================================
    # 7. Make ONE shared palette
    # ========================================================
    mapped_gene_labels = (
        gene_subset.obs["gene_cluster_mapped"]
        .astype(str)
        .values
    )

    palette = make_shared_cluster_palette(
        gene_subset=gene_subset,
        image_cluster_key="image_cluster",
        mapped_gene_labels=mapped_gene_labels,
    )

    # ========================================================
    # 8. Explicitly set Scanpy palettes
    # ========================================================
    set_scanpy_palette(
        gene_subset,
        "image_cluster",
        palette,
    )

    set_scanpy_palette(
        gene_subset,
        "gene_cluster_mapped",
        palette,
    )

    # ========================================================
    # 9. Spatial plot
    # ========================================================
    spatial_path = os.path.join(
        output_dir,
        f"{sample}_spatial_clusters.png",
    )

    plot_spatial_clusters(
        gene_subset=gene_subset,
        palette=palette,
        image_cluster_key="image_cluster",
        gene_mapped_key="gene_cluster_mapped",
        library_id="ST",
        img_key="downscaled_fullres",
        spot_size=spot_size,
        alpha=0.85,
        output_path=spatial_path,
    )

    # ========================================================
    # 10. UMAP plot
    # ========================================================
    umap_path = os.path.join(
        output_dir,
        f"{sample}_UMAP_mapped_clusters.png",
    )

    plot_matched_umaps(
        image_adata=img,
        gene_adata=gene_clust,
        gene_spatial_adata=gene_subset,
        palette=palette,
        image_cluster_key="image_cluster",
        gene_cluster_key="gene_cluster_original",
        output_path=umap_path,
    )

    # ========================================================
    # 11. Save mapping
    # ========================================================
    mapping_path = os.path.join(
        output_dir,
        f"{sample}_cluster_mapping.csv",
    )

    mapping_df.to_csv(
        mapping_path,
        index=False,
    )

    print("\nSaved:")
    print(f"  Spatial: {spatial_path}")
    print(f"  UMAP:    {umap_path}")

    return {
        "img": img,
        "gene_subset": gene_subset,
        "gene_clust": gene_clust,
        "mapping_df": mapping_df,
        "cluster_mapping": cluster_mapping,
        "palette": palette,
    }


import os
import glob
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# 1. GET ALL SAMPLES
# ============================================================

def get_samples(
    image_root,
):
    """
    Get sample names from image embedding files.
    """
    files = glob.glob(
        os.path.join(image_root, "*.h5")
    )

    samples = sorted([
        os.path.splitext(
            os.path.basename(f)
        )[0]
        for f in files
    ])

    return samples


# ============================================================
# 2. PLOT ALL FOUR PANELS FOR ONE SAMPLE
# ============================================================

def plot_sample_four_panel(
    sample,
    result,
    figsize=(16, 12),
    spot_size=100,
    alpha=0.85,
    point_size=12,
):
    """
    Create the four-panel figure for one sample.

    Panels
    ------
    1. Image clusters on H&E
    2. Mapped gene clusters on H&E
    3. Image embedding UMAP
    4. Gene expression UMAP

    Returns
    -------
    fig
    """

    img = result["img"]
    gene_subset = result["gene_subset"]
    gene_clust = result["gene_clust"]
    palette = result["palette"]

    # --------------------------------------------------------
    # Ensure spatial categories use the shared palette
    # --------------------------------------------------------
    set_scanpy_palette(
        gene_subset,
        "image_cluster",
        palette,
    )

    set_scanpy_palette(
        gene_subset,
        "gene_cluster_mapped",
        palette,
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        2,
        2,
        figsize=figsize,
    )

    # ========================================================
    # 1. IMAGE CLUSTERS ON H&E
    # ========================================================
    sc.pl.spatial(
        gene_subset,
        color="image_cluster",
        library_id="ST",
        img_key="downscaled_fullres",
        spot_size=spot_size,
        alpha=alpha,
        ax=ax[0, 0],
        show=False,
    )

    ax[0, 0].set_title(
        f"{sample} - Image clusters"
    )

    # ========================================================
    # 2. GENE CLUSTERS ON H&E
    # ========================================================
    sc.pl.spatial(
        gene_subset,
        color="gene_cluster_mapped",
        library_id="ST",
        img_key="downscaled_fullres",
        spot_size=spot_size,
        alpha=alpha,
        ax=ax[0, 1],
        show=False,
    )

    ax[0, 1].set_title(
        f"{sample} - Gene clusters (mapped)"
    )

    # ========================================================
    # 3. IMAGE UMAP
    # ========================================================
    image_labels = (
        img.obs["image_cluster"]
        .astype(str)
        .values
    )

    image_colors = [
        palette[str(x)]
        for x in image_labels
    ]

    ax[1, 0].scatter(
        img.obsm["X_umap"][:, 0],
        img.obsm["X_umap"][:, 1],
        c=image_colors,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    ax[1, 0].set_title(
        f"{sample} - Image embedding UMAP"
    )
    ax[1, 0].set_xlabel("UMAP1")
    ax[1, 0].set_ylabel("UMAP2")

    # ========================================================
    # 4. GENE UMAP
    # ========================================================
    gene_labels = (
        gene_subset["gene_cluster_mapped"]
        .astype(str)
        .values
    )

    gene_colors = [
        palette[str(x)]
        for x in gene_labels
    ]

    ax[1, 1].scatter(
        gene_clust.obsm["X_umap"][:, 0],
        gene_clust.obsm["X_umap"][:, 1],
        c=gene_colors,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    ax[1, 1].set_title(
        f"{sample} - Gene expression UMAP"
    )
    ax[1, 1].set_xlabel("UMAP1")
    ax[1, 1].set_ylabel("UMAP2")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------
    fig.suptitle(
        sample,
        fontsize=16,
        y=0.995,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    return fig


# ============================================================
# 3. RUN ALL SAMPLES -> ONE PDF
# ============================================================

def plot_embedding_all_samples(
    image_root,
    gene_root,
    output_pdf,
    threshold=0.30,
    n_neighbors=15,
    n_pcs=30,
    image_resolution=0.5,
    gene_resolution=0.5,
    n_hvg=2000,
    spot_size=100,
):
    """
    Run the complete pipeline for every sample and save
    four plots per sample on one PDF page.

    Also saves a CSV with cluster mappings for every sample.

    Returns
    -------
    all_results : dict
        sample -> result dictionary
    """

    os.makedirs(
        os.path.dirname(output_pdf)
        if os.path.dirname(output_pdf)
        else ".",
        exist_ok=True,
    )

    samples = get_samples(
        image_root
    )

    print(
        f"Found {len(samples)} samples."
    )

    all_results = []
    result_dict = {}

    with PdfPages(output_pdf) as pdf:

        for i, sample in enumerate(samples, start=1):

            print(
                f"\n[{i}/{len(samples)}] {sample}"
            )

            try:

                # ------------------------------------------------
                # Run pipeline
                # ------------------------------------------------
                result = plot_embedding(
                    sample=sample,
                    output_dir="plot",
                    image_root=image_root,
                    gene_root=gene_root,
                    threshold=threshold,
                    n_neighbors=n_neighbors,
                    n_pcs=n_pcs,
                    image_resolution=image_resolution,
                    gene_resolution=gene_resolution,
                    n_hvg=n_hvg,
                    spot_size=spot_size,
                )

                # ------------------------------------------------
                # Four-panel figure
                # ------------------------------------------------
                fig = plot_sample_four_panel(
                    sample=sample,
                    result=result,
                    spot_size=spot_size,
                )

                # ------------------------------------------------
                # Add page to PDF
                # ------------------------------------------------
                pdf.savefig(
                    fig,
                    bbox_inches="tight",
                )

                plt.close(fig)

                # ------------------------------------------------
                # Store results
                # ------------------------------------------------
                result_dict[sample] = result

                # Add sample identifier to mapping table
                mapping = result[
                    "mapping_df"
                ].copy()

                mapping.insert(
                    0,
                    "sample",
                    sample,
                )

                all_results.append(
                    mapping
                )

                print(
                    f"Completed {sample}"
                )

            except Exception as e:

                print(
                    f"ERROR in {sample}: {e}"
                )

                # Continue to next sample
                continue

    # ========================================================
    # Save all cluster mappings
    # ========================================================

    if all_results:

        mapping_all = pd.concat(
            all_results,
            ignore_index=True,
        )

        mapping_csv = output_pdf.replace(
            ".pdf",
            "_cluster_mapping.csv",
        )

        mapping_all.to_csv(
            mapping_csv,
            index=False,
        )

        print(
            f"\nCluster mappings saved to:\n"
            f"{mapping_csv}"
        )

    print(
        f"\nPDF saved to:\n{output_pdf}"
    )

    return result_dict