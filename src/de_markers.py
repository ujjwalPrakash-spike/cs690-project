"""Differential Expression Marker Gene Extraction using Wilcoxon Rank-Sum Test.

Computes one-vs-rest DE for each spatial niche using the Wilcoxon rank-sum test.
Extracts exactly the top 10 marker genes per niche with p-values, adjusted p-values,
scores/statistics, and log-fold changes.
"""

import os
import pandas as pd
import scanpy as sc


def extract_top10_markers(
    adata: sc.AnnData,
    cluster_key: str,
    method_name: str,
    output_csv_path: str,
    top_n: int = 10
) -> pd.DataFrame:
    print(f"[{method_name}] Running one-vs-rest Wilcoxon rank-sum test on '{cluster_key}'...")
    adata_de = adata.copy()

    # Ensure cluster column is categorical
    adata_de.obs[cluster_key] = adata_de.obs[cluster_key].astype("category")

    # Wilcoxon rank-sum test (one-vs-rest)
    sc.tl.rank_genes_groups(
        adata_de,
        groupby=cluster_key,
        method="wilcoxon",
        tie_correct=True,
        use_raw=False,
        pts=True
    )

    rg = adata_de.uns["rank_genes_groups"]
    niches = rg["names"].dtype.names

    rows = []
    for niche_id in niches:
        names = rg["names"][niche_id][:top_n]
        scores = rg["scores"][niche_id][:top_n]
        pvals = rg["pvals"][niche_id][:top_n]
        pvals_adj = rg["pvals_adj"][niche_id][:top_n]
        logfc = rg["logfoldchanges"][niche_id][:top_n]

        for rank, (gene, score, pval, padj, lfc) in enumerate(zip(names, scores, pvals, pvals_adj, logfc), start=1):
            rows.append({
                "method": method_name,
                "niche_id": str(niche_id),
                "rank": rank,
                "gene": str(gene),
                "statistic": float(score),
                "p_value": float(pval),
                "adjusted_p_value": float(padj),
                "log_fold_change": float(lfc)
            })

    df = pd.DataFrame(rows)
    dirname = os.path.dirname(output_csv_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[{method_name}] Top-{top_n} marker genes saved to {output_csv_path}")
    return df
