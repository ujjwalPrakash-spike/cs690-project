"""Gemini Free-Form Annotation and Reproducibility Runner.

Implements Phase 4 (free-form zero-shot labeling) and Phase 5 (repeated prompt reproducibility).
Structured JSON output, isolated client, experiment provenance tracking.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

from src.gemini.client import GeminiClient, FreeFormAnnotationResult
from src.gemini.prompts import get_prompt


class GeminiAnnotator:
    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    def annotate_niche(
        self,
        niche_id: str,
        marker_genes: List[str],
        method: str,
        variant: str = "baseline_marker_only",
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        prompt = get_prompt(variant=variant, genes=marker_genes)
        response_info = self.client.generate_annotation(
            prompt=prompt,
            schema=FreeFormAnnotationResult,
            temperature=temperature
        )

        parsed = response_info.get("parsed") or {}
        return {
            "method": method,
            "niche_id": niche_id,
            "prompt_variant": variant,
            "marker_genes": marker_genes,
            "biological_label": parsed.get("biological_label", "Unknown"),
            "short_justification": parsed.get("short_justification", ""),
            "confidence": parsed.get("confidence", 0),
            "model": response_info.get("model"),
            "raw_response": response_info.get("raw_response"),
            "timestamp": response_info.get("timestamp"),
            "status": response_info.get("status")
        }

    def run_free_form_experiment(
        self,
        markers_df: pd.DataFrame,
        method_name: str,
        variant: str = "baseline_marker_only",
        output_dir: str = "outputs/free_form_gemini"
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)
        results = []

        niches = markers_df["niche_id"].unique()
        print(f"[{method_name}] Running free-form Gemini annotation ({variant}) on {len(niches)} niches...")

        for niche_id in niches:
            genes = markers_df[markers_df["niche_id"] == niche_id].sort_values("rank")["gene"].tolist()[:10]
            print(f"  Annotating {niche_id} with genes: {genes[:5]}...")
            res = self.annotate_niche(
                niche_id=niche_id,
                marker_genes=genes,
                method=method_name,
                variant=variant
            )
            results.append(res)

        df_out = pd.DataFrame(results)
        csv_file = os.path.join(output_dir, f"free_form_{method_name.lower()}_{variant}.csv")
        df_out.to_csv(csv_file, index=False)

        # Save raw JSON logs for complete scientific provenance
        json_file = os.path.join(output_dir, f"raw_{method_name.lower()}_{variant}.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[{method_name}] Annotation results saved to {csv_file}")
        return df_out

    def run_reproducibility_experiment(
        self,
        markers_df: pd.DataFrame,
        method_name: str,
        n_repetitions: int = 5,
        n_sample_niches: int = 3,
        variant: str = "baseline_marker_only",
        output_dir: str = "outputs/reproducibility"
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)
        all_niches = list(markers_df["niche_id"].unique())
        selected_niches = all_niches[:n_sample_niches]

        records = []
        print(f"[{method_name}] Running reproducibility study ({n_repetitions} reps across {len(selected_niches)} niches)...")

        for niche_id in selected_niches:
            genes = markers_df[markers_df["niche_id"] == niche_id].sort_values("rank")["gene"].tolist()[:10]
            print(f"  Testing niche {niche_id} for {n_repetitions} repetitions...")
            for rep in range(1, n_repetitions + 1):
                res = self.annotate_niche(
                    niche_id=niche_id,
                    marker_genes=genes,
                    method=method_name,
                    variant=variant,
                    temperature=0.2  # Test stability under fixed temperature
                )
                res["repetition"] = rep
                records.append(res)

        rep_df = pd.DataFrame(records)
        csv_file = os.path.join(output_dir, f"reproducibility_{method_name.lower()}.csv")
        rep_df.to_csv(csv_file, index=False)

        # Compute exact self-agreement metrics per niche
        stability_records = []
        for niche_id, group in rep_df.groupby("niche_id"):
            labels = group["biological_label"].str.strip().str.lower().tolist()
            most_frequent_count = pd.Series(labels).value_counts().iloc[0]
            exact_self_agreement = most_frequent_count / len(labels)
            unique_labels = len(set(labels))
            conf_mean = group["confidence"].mean()
            conf_std = group["confidence"].std()
            stability_records.append({
                "method": method_name,
                "niche_id": niche_id,
                "repetitions": len(labels),
                "unique_labels_count": unique_labels,
                "exact_self_agreement": exact_self_agreement,
                "mean_confidence": conf_mean,
                "confidence_std": conf_std,
                "dominant_label": pd.Series(labels).value_counts().index[0],
                "all_observed_labels": list(set(labels))
            })

        stability_df = pd.DataFrame(stability_records)
        stability_csv = os.path.join(output_dir, f"stability_summary_{method_name.lower()}.csv")
        stability_df.to_csv(stability_csv, index=False)
        print(f"[{method_name}] Reproducibility summary saved to {stability_csv}")
        return rep_df
