"""Crosstab Evaluation and Dominant Region Determination.

Inspects or computes contingency tables between computational niches
(BANKSY, CellCharter) and ground-truth DLPFC regions (Maynard et al. 2021).
Derives dominant expert region mapping for evaluation without data leakage.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def inspect_or_create_crosstab(
    clusters_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    method_name: str,
    output_dir: str = "outputs/evaluation"
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    os.makedirs(output_dir, exist_ok=True)

    # Check if external user crosstab exists in home directory
    home_crosstab = os.path.expanduser(f"~/crosstab_{method_name.lower()}_vs_region.csv")
    if os.path.exists(home_crosstab):
        print(f"[{method_name}] Found user-provided crosstab at: {home_crosstab}")
        ct_df = pd.read_csv(home_crosstab, index_col=0)
    else:
        print(f"[{method_name}] Computing contingency crosstab against authentic DLPFC ground truth...")
        # Merge on barcode index
        clusters_indexed = clusters_df.set_index("barcode")
        gt_indexed = ground_truth_df.copy()
        if "barcode" in gt_indexed.columns:
            gt_indexed = gt_indexed.set_index("barcode")

        common_barcodes = clusters_indexed.index.intersection(gt_indexed.index)
        merged = pd.DataFrame({
            "niche": clusters_indexed.loc[common_barcodes, "niche_id"],
            "region": gt_indexed.loc[common_barcodes, "layer_guess"]
        }).dropna()

        ct_df = pd.crosstab(merged["niche"], merged["region"])

        # Save computed crosstab
        out_csv = os.path.join(output_dir, f"crosstab_{method_name.lower()}_vs_region.csv")
        ct_df.to_csv(out_csv)
        print(f"[{method_name}] Crosstab saved to {out_csv}")

    # Determine dominant expert region per niche
    # For each row (niche), find the column (region) with maximum count
    dominant_mapping = {}
    for niche in ct_df.index:
        top_region = ct_df.loc[niche].idxmax()
        count = ct_df.loc[niche, top_region]
        total = ct_df.loc[niche].sum()
        pct = (count / total) * 100 if total > 0 else 0
        dominant_mapping[str(niche)] = str(top_region)
        print(f"  [{method_name}] Niche '{niche}' -> Dominant Region: '{top_region}' ({count}/{total} spots, {pct:.1f}%)")

    # Save mapping documentation
    mapping_path = os.path.join(output_dir, f"dominant_mapping_{method_name.lower()}.json")
    with open(mapping_path, "w") as f:
        json.dump(dominant_mapping, f, indent=2)

    return ct_df, dominant_mapping
