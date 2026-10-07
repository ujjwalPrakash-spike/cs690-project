"""CellCharter Spatial Niche Detection Pipeline.

Uses cellcharter (Varrone et al., Nature Genetics 2024):
- Squidpy spatial neighbor graph
- Multi-layer spatial neighborhood aggregation (n_layers=3)
- Gaussian Mixture Model clustering on aggregated representations
- Deterministic random seed
"""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import cellcharter as cc
import matplotlib.pyplot as plt


def run_cellcharter_clustering(
    adata_input: sc.AnnData,
    output_dir: str = "outputs/cellcharter",
    n_neighbors: int = 10,
    n_layers: int = 3,
    n_pcs: int = 20,
    n_clusters: int = 7,
    random_state: int = 42
) -> sc.AnnData:
    os.makedirs(output_dir, exist_ok=True)
    adata = adata_input.copy()

    # Step 1: PCA on HVGs or expression matrix
    if "X_pca" not in adata.obsm:
        print(f"[CellCharter] Computing PCA ({n_pcs} PCs)...")
        if "highly_variable" in adata.var and adata.var["highly_variable"].sum() > 500:
            sc.pp.pca(adata, n_comps=n_pcs, mask_var="highly_variable", random_state=random_state)
        else:
            sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

    # Step 2: Build spatial neighborhood graph
    print(f"[CellCharter] Constructing spatial neighbor graph (n_neighbors={n_neighbors})...")
    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        spatial_key="spatial",
        n_neighs=n_neighbors
    )

    # Step 3: Aggregate spatial neighborhoods across layers
    print(f"[CellCharter] Aggregating spatial neighborhoods across {n_layers} layers...")
    cc.gr.aggregate_neighbors(
        adata,
        n_layers=n_layers,
        use_rep="X_pca",
        out_key="X_cellcharter"
    )

    # Step 4: Fit GMM clustering model
    print(f"[CellCharter] Fitting GMM with {n_clusters} clusters (seed={random_state})...")
    gmm = cc.tl.Cluster(n_clusters=n_clusters, random_state=random_state)
    gmm.fit(adata, use_rep="X_cellcharter")
    predicted_labels = gmm.predict(adata, use_rep="X_cellcharter")

    # Map labels to obs
    adata.obs["niche_cellcharter"] = pd.Categorical([f"CellCharter_{c}" for c in predicted_labels])

    # Save outputs
    # 1. Cluster assignments CSV
    clusters_df = pd.DataFrame({
        "barcode": adata.obs_names,
        "niche_id": adata.obs["niche_cellcharter"],
        "array_row": adata.obs["array_row"],
        "array_col": adata.obs["array_col"]
    })
    csv_path = os.path.join(output_dir, "clusters_cellcharter.csv")
    clusters_df.to_csv(csv_path, index=False)
    print(f"[CellCharter] Cluster CSV saved to {csv_path}")

    # 2. Parameters JSON
    params = {
        "method": "CellCharter",
        "implementation": "cellcharter 0.3.7",
        "n_neighbors": n_neighbors,
        "n_layers": n_layers,
        "n_pcs": n_pcs,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_spots": int(adata.n_obs),
        "cluster_distribution": adata.obs["niche_cellcharter"].value_counts().to_dict()
    }
    params_path = os.path.join(output_dir, "parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # 3. Spatial Plot
    plt.figure(figsize=(8, 7), dpi=300)
    palette = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    categories = sorted(adata.obs["niche_cellcharter"].unique())
    coords = adata.obsm["spatial"]

    for idx, cat in enumerate(categories):
        mask = adata.obs["niche_cellcharter"] == cat
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
    plt.title(f"CellCharter Niche Detection (k={n_clusters}, layers={n_layers})", fontsize=13, fontweight="bold")
    plt.xlabel("Spatial X", fontsize=11)
    plt.ylabel("Spatial Y", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spatial_clusters_cellcharter.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[CellCharter] Spatial visualization saved to {plot_path}")

    # 4. Save AnnData copy
    h5ad_out = os.path.join(output_dir, "adata_cellcharter.h5ad")
    adata.write_h5ad(h5ad_out)
    print(f"[CellCharter] AnnData copy saved to {h5ad_out}")

    return adata
