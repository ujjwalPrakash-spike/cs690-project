"""Prompt definitions for Free-Form and Constrained LLM Experiments.

Baseline: Strictly marker-genes-only without anatomical or coordinate leakage.
Variants: Clearly separated and tagged so experiments cannot be confused.
"""

from typing import List


def format_genes(genes: List[str]) -> str:
    return ", ".join(genes)


# Variant 1 (BASELINE): Marker-genes only
BASELINE_MARKER_ONLY_PROMPT = """You are annotating a spatial transcriptomics niche from marker genes alone.

The top 10 differentially expressed marker genes are:
{GENES}

Based only on these marker genes, provide:
1. biological_label
2. short_justification
3. confidence (0-100)

Do not assume a predefined label set.
Do not use spatial coordinates, image information, cluster ID semantics, or external ground-truth labels.
Do not invent evidence that is not supported by the genes."""


# Variant 2 (EXPERIMENTAL): Marker genes + DLPFC tissue context
DLPFC_CONTEXT_PROMPT = """You are annotating a spatial transcriptomics niche from human dorsolateral prefrontal cortex (DLPFC).

The top 10 differentially expressed marker genes are:
{GENES}

Based on these marker genes and the DLPFC tissue context, provide:
1. biological_label
2. short_justification
3. confidence (0-100)

Do not assume a predefined label set.
Do not use spatial coordinates, image information, cluster ID semantics, or external ground-truth labels.
Do not invent evidence that is not supported by the genes."""


# Variant 3 (EXPERIMENTAL): Marker genes + Functional Description Request
FUNCTIONAL_DESCRIPTION_PROMPT = """You are annotating a spatial transcriptomics niche from marker genes alone with functional interpretation.

The top 10 differentially expressed marker genes are:
{GENES}

Based only on these marker genes, provide:
1. biological_label
2. short_justification (including inferred cellular processes, pathway activity, and tissue niche function)
3. confidence (0-100)

Do not assume a predefined label set.
Do not use spatial coordinates, image information, cluster ID semantics, or external ground-truth labels.
Do not invent evidence that is not supported by the genes."""


# Constrained NicheAgent-style prompt
CONSTRAINED_NICHEAGENT_PROMPT = """You are an adjudicator in a constrained spatial transcriptomics niche annotation system (NicheAgent-style).

The prototype matching step has flagged a low-confidence niche assignment.
Proposed label by prototype: {PROPOSED_LABEL} (similarity margin was low)

Allowed label set (closed-world vocabulary):
{ALLOWED_LABELS}

Evidence:
- Local top marker genes: {MARKER_GENES}
- Spatial neighborhood composition: {NEIGHBORHOOD_CONTEXT}

Instructions:
1. Review the proposed label against the allowed label set.
2. You MUST select exactly one label strictly from the allowed label set. Do NOT invent or output any label outside the allowed set.
3. Provide a concise justification explaining why this label best matches the marker genes and spatial neighborhood context.
4. Provide a confidence score between 0 and 100."""


def get_prompt(variant: str, genes: List[str], **kwargs) -> str:
    genes_str = format_genes(genes)
    if variant == "baseline_marker_only":
        return BASELINE_MARKER_ONLY_PROMPT.format(GENES=genes_str)
    elif variant == "dlpfc_context":
        return DLPFC_CONTEXT_PROMPT.format(GENES=genes_str)
    elif variant == "functional_description":
        return FUNCTIONAL_DESCRIPTION_PROMPT.format(GENES=genes_str)
    elif variant == "constrained_nicheagent":
        allowed_str = ", ".join(kwargs.get("allowed_labels", []))
        return CONSTRAINED_NICHEAGENT_PROMPT.format(
            PROPOSED_LABEL=kwargs.get("proposed_label", "Unknown"),
            ALLOWED_LABELS=allowed_str,
            MARKER_GENES=genes_str,
            NEIGHBORHOOD_CONTEXT=kwargs.get("neighborhood_context", "N/A")
        )
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")
