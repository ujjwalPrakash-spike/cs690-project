"""Prototype-based Niche Assignment and Confidence Margin Evaluation.

Implements NicheAgent prototype scoring and confidence margin filtering.
Identifies low-confidence niche assignments for LLM review.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from src.nicheagent.nichecards import DLPFC_NICHECARDS, get_all_allowed_labels


def compute_niche_prototype_scores(
    top_markers: List[str],
    allowed_labels: List[str] = None
) -> Dict[str, float]:
    """Computes Jaccard / weighted overlap scores between niche markers and NicheCard markers."""
    if allowed_labels is None:
        allowed_labels = get_all_allowed_labels()

    scores = {}
    top_set = set([m.upper() for m in top_markers])

    for label in allowed_labels:
        card = DLPFC_NICHECARDS[label]
        canonical = set([m.upper() for m in card["canonical_markers"]])
        intersection = top_set.intersection(canonical)
        # Score combines recall and precision relative to top markers
        score = len(intersection) / max(len(canonical), 1)
        scores[label] = float(score)

    return scores


def assign_prototypes_with_margin(
    niche_markers_df: pd.DataFrame,
    confidence_margin_tau: float = 0.15
) -> pd.DataFrame:
    """Evaluates each niche against NicheCard prototypes, computing top-1, top-2,

    and confidence margin. Flags low-confidence assignments (margin < tau).
    """
    records = []
    allowed_labels = get_all_allowed_labels()

    for niche_id, group in niche_markers_df.groupby("niche_id"):
        genes = group.sort_values("rank")["gene"].tolist()[:10]
        scores = compute_niche_prototype_scores(genes, allowed_labels)

        # Rank prototypes by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1_label, top1_score = sorted_scores[0]
        top2_label, top2_score = sorted_scores[1] if len(sorted_scores) > 1 else ("None", 0.0)

        margin = top1_score - top2_score
        is_low_confidence = (margin < confidence_margin_tau) or (top1_score == 0.0)

        records.append({
            "niche_id": niche_id,
            "top1_prototype": top1_label,
            "top1_score": top1_score,
            "top2_prototype": top2_label,
            "top2_score": top2_score,
            "margin": margin,
            "margin_threshold_tau": confidence_margin_tau,
            "is_low_confidence": is_low_confidence,
            "all_prototype_scores": scores,
            "marker_genes": genes
        })

    return pd.DataFrame(records)
