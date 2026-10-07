"""Evaluation Metrics and Biological Label Matching Engine.

Computes:
- Exact / partial / mismatch classification respecting biological terminology
- Confident-wrong identification
- NMI, ARI, accuracy, macro-F1, balanced accuracy, homogeneity, completeness
- Confidence vs correctness correlation
"""

import os
import re
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    homogeneity_score,
    completeness_score
)


# Synonyms and layer mappings for human DLPFC
LAYER_SYNONYMS = {
    "Layer 1": ["layer 1", "layer1", "l1", "molecular layer", "marginal zone", "lamina i", "layer i"],
    "Layer 2": ["layer 2", "layer2", "l2", "external granular layer", "lamina ii", "layer ii"],
    "Layer 3": ["layer 3", "layer3", "l3", "external pyramidal layer", "lamina iii", "layer iii"],
    "Layer 4": ["layer 4", "layer4", "l4", "internal granular layer", "lamina iv", "layer iv"],
    "Layer 5": ["layer 5", "layer5", "l5", "internal pyramidal layer", "lamina v", "layer v"],
    "Layer 6": ["layer 6", "layer6", "l6", "multiform layer", "polymorphic layer", "lamina vi", "layer vi"],
    "WM": ["white matter", "wm", "subcortical white matter", "corpus callosum", "myelin"]
}

# Relevant cell types / partial matches per layer
LAYER_PARTIAL_ASSOCIATIONS = {
    "Layer 1": ["astrocyte", "astrocytic", "interneuron", "glia", "glial", "cajal-retzius", "superficial"],
    "Layer 2": ["excitatory neuron", "pyramidal", "cortex", "cortical", "upper layer", "supragranular", "layer 2/3"],
    "Layer 3": ["excitatory neuron", "pyramidal", "cortex", "cortical", "upper layer", "supragranular", "layer 2/3"],
    "Layer 4": ["granular", "interneuron", "excitatory", "cortex", "cortical", "middle layer"],
    "Layer 5": ["deep layer", "large pyramidal", "infragranular", "excitatory", "cortex", "cortical", "projection neuron"],
    "Layer 6": ["deep layer", "corticothalamic", "infragranular", "excitatory", "cortex", "cortical", "polymorphic"],
    "WM": ["oligodendrocyte", "oligodendroglial", "myelinated", "glial", "axon", "astrocytes"]
}


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", str(label).strip().lower())
    return cleaned


def classify_biological_match(predicted_label: str, true_region: str) -> Tuple[str, str]:
    """Classifies prediction as 'exact', 'partial', or 'mismatch' relative to true DLPFC region.

    Returns: (match_category, explanation)
    """
    pred_norm = normalize_label(predicted_label)
    true_norm = normalize_label(true_region)

    # Standardize true_region to canonical key
    canonical_true = None
    for c_key, syns in LAYER_SYNONYMS.items():
        if true_norm in [s.lower() for s in syns] or c_key.lower() == true_norm:
            canonical_true = c_key
            break

    if canonical_true is None:
        canonical_true = true_region

    # 1. Exact match check
    exact_syns = LAYER_SYNONYMS.get(canonical_true, [canonical_true.lower()])
    for syn in exact_syns:
        # Whole phrase match or distinct mention
        if re.search(rf"\b{re.escape(syn)}\b", pred_norm):
            return "exact", f"Direct biological match to {canonical_true} ({syn})"

    # 2. Partial match check (associated cell types / adjacent layer)
    partial_terms = LAYER_PARTIAL_ASSOCIATIONS.get(canonical_true, [])
    for term in partial_terms:
        if re.search(rf"\b{re.escape(term)}\b", pred_norm):
            return "partial", f"Biologically compatible marker/cell-type for {canonical_true} ({term})"

    # Check general neuron/cortex for cortical layers
    if "cort" in pred_norm or "neuron" in pred_norm:
        if canonical_true != "WM":
            return "partial", f"General cortical/neuronal label corresponding to {canonical_true}"

    return "mismatch", f"Discordant with dominant region {canonical_true}"


def evaluate_niche_predictions(
    predictions_df: pd.DataFrame,
    crosstab_mapping: Dict[str, str],
    pred_col: str = "biological_label",
    conf_col: str = "confidence"
) -> Dict[str, Any]:
    """Evaluates niche-level predictions against dominant ground truth regions."""
    records = []
    for _, row in predictions_df.iterrows():
        niche_id = row["niche_id"]
        true_region = crosstab_mapping.get(niche_id, "Unknown")
        pred_label = row[pred_col]
        conf = row.get(conf_col, 0)

        match_cat, explanation = classify_biological_match(pred_label, true_region)
        is_confident_wrong = (match_cat == "mismatch") and (conf >= 80)

        records.append({
            "niche_id": niche_id,
            "predicted_label": pred_label,
            "true_dominant_region": true_region,
            "match_category": match_cat,
            "explanation": explanation,
            "confidence": conf,
            "confident_wrong": is_confident_wrong
        })

    eval_df = pd.DataFrame(records)
    n_total = len(eval_df)

    exact_rate = (eval_df["match_category"] == "exact").mean() if n_total > 0 else 0.0
    partial_rate = (eval_df["match_category"] == "partial").mean() if n_total > 0 else 0.0
    mismatch_rate = (eval_df["match_category"] == "mismatch").mean() if n_total > 0 else 0.0

    # Agreement score: exact=1.0, partial=0.5, mismatch=0.0
    scores = eval_df["match_category"].map({"exact": 1.0, "partial": 0.5, "mismatch": 0.0})
    mean_agreement = scores.mean() if n_total > 0 else 0.0

    summary = {
        "n_niches": n_total,
        "exact_match_rate": float(exact_rate),
        "partial_match_rate": float(partial_rate),
        "mismatch_rate": float(mismatch_rate),
        "mean_agreement_score": float(mean_agreement),
        "confident_wrong_count": int(eval_df["confident_wrong"].sum()),
        "mean_confidence": float(eval_df["confidence"].mean()) if "confidence" in eval_df else 0.0,
        "details": eval_df.to_dict(orient="records")
    }
    return summary, eval_df


def compute_spatial_clustering_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> Dict[str, float]:
    """Computes spot-level clustering metrics (ARI, NMI, macro-F1, accuracy, etc.)

    Filters out NaN or unannotated ground truth spots.
    """
    # Filter valid
    mask = pd.notna(true_labels) & pd.notna(predicted_labels)
    y_true = np.array(true_labels)[mask]
    y_pred = np.array(predicted_labels)[mask]

    if len(y_true) == 0:
        return {
            "ARI": None,
            "NMI": None,
            "accuracy": None,
            "macro_F1": None,
            "balanced_accuracy": None,
            "homogeneity": None,
            "completeness": None
        }

    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    homo = float(homogeneity_score(y_true, y_pred))
    comp = float(completeness_score(y_true, y_pred))

    # Standardize names for classification metrics
    # Normalize labels
    y_true_clean = [normalize_label(x) for x in y_true]
    y_pred_clean = [normalize_label(x) for x in y_pred]

    # Map aliases (e.g. 'wm' -> 'white matter')
    alias_map = {"wm": "white matter"}
    y_true_clean = [alias_map.get(x, x) for x in y_true_clean]
    y_pred_clean = [alias_map.get(x, x) for x in y_pred_clean]

    acc = float(accuracy_score(y_true_clean, y_pred_clean))
    f1 = float(f1_score(y_true_clean, y_pred_clean, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_clean, y_pred_clean))

    return {
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "accuracy": round(acc, 4),
        "macro_F1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "homogeneity": round(homo, 4),
        "completeness": round(comp, 4)
    }
