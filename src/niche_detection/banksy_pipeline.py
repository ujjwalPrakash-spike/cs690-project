"""BANKSY Spatial Niche Detection Pipeline.

Uses pybanksy with standard recommended settings:
- Fixed-neighbor spatial graph (k=15)
- Scaled Gaussian weight decay
- Spatial lambda parameter = 0.2
- AGF order = 1
- PCA dimensionality reduction (20 PCs)
- Deterministic clustering into 7 niches (matching DLPFC cortical architecture)
"""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix


def run_banksy_clustering(
    adata_input: sc.AnnData,
    output_dir: str = "outputs/banksy",
    k_geom: int = 15,
    lambda_param: float = 0.2,
    n_pcs: int = 20,
    n_clusters: int = 7,
    random_state: int = 42
) -> sc.AnnData:
    os.makedirs(output_dir, exist_ok=True)
    adata = adata_input.copy()

    # Ensure coordinates in obs for BANKSY
    if "spatial" in adata.obsm:
        adata.obs["spatial_x"] = adata.obsm["spatial"][:, 0]
        adata.obs["spatial_y"] = adata.obsm["spatial"][:, 1]
    else:
        adata.obs["spatial_x"] = adata.obs["array_row"]
        adata.obs["spatial_y"] = adata.obs["array_col"]

    # Use Highly Variable Genes if available, or all genes
    if "highly_variable" in adata.var and adata.var["highly_variable"].sum() > 500:
        adata_sub = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata_sub = adata.copy()

    # Step 1: Initialize BANKSY spatial weights
    print(f"[BANKSY] Initializing spatial graph (k_geom={k_geom}, decay='scaled_gaussian')...")
    bdict = initialize_banksy(
        adata_sub,
        coord_keys=("spatial_x", "spatial_y", "spatial"),
        num_neighbours=k_geom,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False
    )

    # Step 2: Generate BANKSY augmented expression matrix
    print(f"[BANKSY] Generating augmented matrix with lambda={lambda_param}...")
    bdict, _ = generate_banksy_matrix(
        adata_sub,
        bdict,
        lambda_list=[lambda_param],
        max_m=1,
        verbose=False
    )

    # Extract the BANKSY feature AnnData
    b_adata = bdict["scaled_gaussian"][lambda_param]["adata"]

    # Step 3: Dimensionality reduction on BANKSY feature space
    print(f"[BANKSY] Running PCA ({n_pcs} PCs)...")
    sc.pp.pca(b_adata, n_comps=n_pcs, random_state=random_state)

    # Step 4: Clustering into discrete niches (K-Means with k=n_clusters for exact k partition)
    print(f"[BANKSY] Clustering into {n_clusters} niches...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(b_adata.obsm["X_pca"])

    # Map back to full AnnData
    adata.obs["niche_banksy"] = pd.Categorical([f"BANKSY_{c}" for c in cluster_labels])
    adata.obsm["X_banksy_pca"] = b_adata.obsm["X_pca"]

    # Save outputs
    # 1. Cluster assignments CSV
    clusters_df = pd.DataFrame({
        "barcode": adata.obs_names,
        "niche_id": adata.obs["niche_banksy"],
        "array_row": adata.obs["array_row"],
        "array_col": adata.obs["array_col"]
    })
    csv_path = os.path.join(output_dir, "clusters_banksy.csv")
    clusters_df.to_csv(csv_path, index=False)
    print(f"[BANKSY] Cluster CSV saved to {csv_path}")

    # 2. Parameters JSON
    params = {
        "method": "BANKSY",
        "implementation": "pybanksy 1.3.5",
        "k_geom": k_geom,
        "nbr_weight_decay": "scaled_gaussian",
        "lambda_param": lambda_param,
        "max_m": 1,
        "n_pcs": n_pcs,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_spots": int(adata.n_obs),
        "cluster_distribution": adata.obs["niche_banksy"].value_counts().to_dict()
    }
    params_path = os.path.join(output_dir, "parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # 3. Spatial Plot
    plt.figure(figsize=(8, 7), dpi=300)
    palette = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    categories = sorted(adata.obs["niche_banksy"].unique())
    coords = adata.obsm["spatial"]

    for idx, cat in enumerate(categories):
        mask = adata.obs["niche_banksy"] == cat
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[palette[idx]],
            label=cat,
            s=18,
            alpha=0.9,
            edgecolors="none"
        )

    plt.gca().invert_yaxis()
    plt.title(f"BANKSY Niche Detection (k={n_clusters}, λ={lambda_param})", fontsize=13, fontweight="bold")
    plt.xlabel("Spatial X", fontsize=11)
    plt.ylabel("Spatial Y", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spatial_clusters_banksy.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[BANKSY] Spatial visualization saved to {plot_path}")

    # 4. Save AnnData copy
    h5ad_out = os.path.join(output_dir, "adata_banksy.h5ad")
    adata.write_h5ad(h5ad_out)
    print(f"[BANKSY] AnnData copy saved to {h5ad_out}")

    return adata
