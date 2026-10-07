"""Complete NicheAgent-Style Pipeline Implementation.

Unifies:
1. Spatial graph / neighborhood information
2. Curated niche representation / NicheCards
3. Prototype-based assignment
4. Confidence margin
5. LLM review ONLY for low-confidence cases
6. Constrained label vocabulary
7. One round of spatial smoothing

Provenance records clearly differentiate exact reproduction,
approximations, and newly implemented components.
"""

import os
import json
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Dict, Any, Optional

from src.nicheagent.nichecards import DLPFC_NICHECARDS, get_all_allowed_labels
from src.nicheagent.prototype_assigner import assign_prototypes_with_margin
from src.nicheagent.constrained_llm import ConstrainedNicheAgentLLM
from src.nicheagent.spatial_smoothing import spatial_graph_smoothing


class NicheAgentPipeline:
    def __init__(
        self,
        confidence_margin_tau: float = 0.15,
        spatial_smoothing_rounds: int = 1,
        k_spatial_neighbors: int = 6,
        client = None
    ):
        self.tau = confidence_margin_tau
        self.smoothing_rounds = spatial_smoothing_rounds
        self.k_spatial = k_spatial_neighbors
        self.llm = ConstrainedNicheAgentLLM(client=client)

    def run(
        self,
        adata: sc.AnnData,
        cluster_col: str,
        markers_df: pd.DataFrame,
        method_name: str,
        output_dir: str = "outputs/nicheagent"
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[NicheAgent] Running pipeline on {method_name} niches (tau={self.tau}, smoothing_rounds={self.smoothing_rounds})...")

        # Step 1 & 2 & 3: Prototype assignment & margin check
        proto_df = assign_prototypes_with_margin(markers_df, confidence_margin_tau=self.tau)

        # Compute spatial neighborhood composition per cluster
        # e.g., which clusters neighbor which clusters on the tissue
        coords = adata.obsm["spatial"]
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_spatial + 1).fit(coords)
        _, indices = nbrs.kneighbors(coords)

        # Build neighborhood summary for each niche
        spot_clusters = adata.obs[cluster_col].values
        niche_neighborhoods = {}
        for niche_id in proto_df["niche_id"].unique():
            niche_mask = np.where(spot_clusters == niche_id)[0]
            if len(niche_mask) > 0:
                neighbor_spots = indices[niche_mask].flatten()
                neighbor_cl = pd.Series(spot_clusters[neighbor_spots]).value_counts(normalize=True)
                top_neighbors = [f"{k} ({v*100:.1f}%)" for k, v in neighbor_cl.items() if k != niche_id][:3]
                niche_neighborhoods[niche_id] = ", ".join(top_neighbors) if top_neighbors else "homogeneous"
            else:
                niche_neighborhoods[niche_id] = "unknown"

        # Step 4: Constrained LLM Review ONLY for low-confidence cases
        niche_final_labels = {}
        llm_reviews = []

        for _, row in proto_df.iterrows():
            niche_id = row["niche_id"]
            if row["is_low_confidence"]:
                print(f"  [NicheAgent] Escalating low-confidence niche '{niche_id}' to LLM review (margin={row['margin']:.3f} < {self.tau})...")
                nhood_context = niche_neighborhoods.get(niche_id, "unknown")
                review = self.llm.review_low_confidence_niche(
                    niche_id=niche_id,
                    marker_genes=row["marker_genes"],
                    proposed_label=row["top1_prototype"],
                    neighborhood_context=nhood_context
                )
                niche_final_labels[niche_id] = review["adjudicated_label"]
                llm_reviews.append(review)
            else:
                print(f"  [NicheAgent] Niche '{niche_id}' accepted via prototype '{row['top1_prototype']}' (margin={row['margin']:.3f} >= {self.tau}).")
                niche_final_labels[niche_id] = row["top1_prototype"]

        # Map predictions to spots
        spot_raw_predictions = adata.obs[cluster_col].map(niche_final_labels)

        # Step 5: Spatial Smoothing (neighborhood-based majority voting)
        print(f"[NicheAgent] Performing {self.smoothing_rounds} round(s) of spatial neighborhood smoothing...")
        spot_smoothed_predictions = spatial_graph_smoothing(
            spot_labels=spot_raw_predictions,
            spatial_coords=coords,
            k_neighbors=self.k_spatial,
            num_rounds=self.smoothing_rounds
        )

        # Save results
        # 1. Spot-level predictions CSV
        spot_res_df = pd.DataFrame({
            "barcode": adata.obs_names,
            "cluster_id": adata.obs[cluster_col],
            "nicheagent_pre_smoothing": spot_raw_predictions,
            "nicheagent_post_smoothing": spot_smoothed_predictions
        })
        spot_csv = os.path.join(output_dir, f"spot_predictions_{method_name.lower()}.csv")
        spot_res_df.to_csv(spot_csv, index=False)

        # 2. Niche-level decisions CSV
        niche_decisions = []
        for _, row in proto_df.iterrows():
            nid = row["niche_id"]
            niche_decisions.append({
                "method": method_name,
                "niche_id": nid,
                "top1_prototype": row["top1_prototype"],
                "top1_score": row["top1_score"],
                "top2_prototype": row["top2_prototype"],
                "top2_score": row["top2_score"],
                "margin": row["margin"],
                "is_low_confidence": row["is_low_confidence"],
                "llm_adjudicated": row["is_low_confidence"],
                "final_niche_label": niche_final_labels[nid]
            })
        decisions_df = pd.DataFrame(niche_decisions)
        decisions_csv = os.path.join(output_dir, f"niche_decisions_{method_name.lower()}.csv")
        decisions_df.to_csv(decisions_csv, index=False)

        # 3. Provenance & metadata JSON
        provenance = {
            "method": "NicheAgent-Style Framework",
            "cluster_input_method": method_name,
            "implementation_fidelity": {
                "exact_reproduction": [
                    "NicheCards structure with canonical markers and aliases",
                    "Deterministic prototype expression/marker scoring",
                    "Confidence margin filtering (Delta = S(1) - S(2))",
                    "Constrained LLM review strictly for low-confidence cases (margin < tau)",
                    "Closed-world ontology enforcement",
                    "Spatial graph neighborhood smoothing"
                ],
                "approximations": [
                    "Spatial neighborhood context summarized from k=6 Visium graph rather than continuous attention weights",
                    "Closed-world vocabulary restricted to DLPFC cortical layers 1-6 and White Matter"
                ],
                "newly_implemented": [
                    "Direct wrapper supporting both BANKSY and CellCharter spatial cluster inputs",
                    "Automated JSON schema validation via Google GenAI SDK"
                ]
            },
            "parameters": {
                "confidence_margin_tau": self.tau,
                "spatial_smoothing_rounds": self.smoothing_rounds,
                "k_spatial_neighbors": self.k_spatial,
                "allowed_labels": get_all_allowed_labels()
            },
            "llm_reviews_conducted": len(llm_reviews),
            "reviews": llm_reviews
        }
        prov_path = os.path.join(output_dir, f"provenance_{method_name.lower()}.json")
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2)

        print(f"[NicheAgent] Complete! Outputs saved to {output_dir}")
        return {
            "spot_predictions": spot_res_df,
            "niche_decisions": decisions_df,
            "provenance": provenance
        }
