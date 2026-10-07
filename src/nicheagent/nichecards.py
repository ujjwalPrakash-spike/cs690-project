"""Curated NicheCards for Human Dorsolateral Prefrontal Cortex (DLPFC).

Faithfully represents the NicheCard ontology structure described in NicheAgent:
- Canonical marker genes
- Aliases / synonymous terminology
- Prototype expression representation
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


# Canonical marker genes based on Maynard et al. (2021) Nature Neuroscience
# and Hu et al. DLPFC reference layer studies
DLPFC_NICHECARDS: Dict[str, Dict[str, Any]] = {
    "Layer 1": {
        "label": "Layer 1",
        "canonical_markers": ["AQP4", "GFAP", "RELN", "FGFR3", "ALDH1L1", "SLC1A2"],
        "aliases": ["Layer 1", "L1", "Layer1", "Molecular layer", "Marginal zone"],
        "description": "Superficial cortical layer with sparse neurons and abundant astrocytic processes."
    },
    "Layer 2": {
        "label": "Layer 2",
        "canonical_markers": ["LAMP5", "CARTPT", "CUX2", "HPCAL1", "SV2C", "NDNF"],
        "aliases": ["Layer 2", "L2", "Layer2", "External granular layer"],
        "description": "Dense small pyramidal and non-pyramidal neurons expressing LAMP5 and CUX2."
    },
    "Layer 3": {
        "label": "Layer 3",
        "canonical_markers": ["CUX2", "PRSS12", "CARTPT", "RORB", "CALB1", "RASGRF2"],
        "aliases": ["Layer 3", "L3", "Layer3", "External pyramidal layer"],
        "description": "Medium-sized pyramidal neurons projecting cortico-cortically."
    },
    "Layer 4": {
        "label": "Layer 4",
        "canonical_markers": ["RORB", "FOXP2", "PCP4", "KCNAB1", "COL5A2"],
        "aliases": ["Layer 4", "L4", "Layer4", "Internal granular layer"],
        "description": "Thalamorecipient granular layer with high RORB expression."
    },
    "Layer 5": {
        "label": "Layer 5",
        "canonical_markers": ["PCP4", "KCNIP4", "NEFM", "FEZF2", "SYT2", "SLC17A7"],
        "aliases": ["Layer 5", "L5", "Layer5", "Internal pyramidal layer"],
        "description": "Large subcerebrally projecting pyramidal neurons with strong PCP4 and NEFM."
    },
    "Layer 6": {
        "label": "Layer 6",
        "canonical_markers": ["NR4A2", "SYT6", "FOXP2", "KCNAB1", "NTNG2", "SEMA3E"],
        "aliases": ["Layer 6", "L6", "Layer6", "Multiform layer", "Polymorphic layer"],
        "description": "Deep multiform layer projecting to thalamus with NR4A2 and SYT6."
    },
    "White Matter": {
        "label": "White Matter",
        "canonical_markers": ["MBP", "MOBP", "PLP1", "MAG", "MOG", "CNP"],
        "aliases": ["White Matter", "WM", "Subcortical white matter", "Myelinated axon tract"],
        "description": "Myelinated axonal tracts enriched in oligodendrocytes and myelin sheath proteins."
    }
}


def get_all_allowed_labels() -> List[str]:
    return list(DLPFC_NICHECARDS.keys())


def build_prototype_matrix(var_names: List[str]) -> pd.DataFrame:
    """Builds a binary/weighted marker incidence matrix across available genes."""
    labels = get_all_allowed_labels()
    proto_df = pd.DataFrame(0.0, index=labels, columns=var_names)

    for label, card in DLPFC_NICHECARDS.items():
        markers = card["canonical_markers"]
        present_markers = [m for m in markers if m in var_names]
        if present_markers:
            weight = 1.0 / len(present_markers)
            for m in present_markers:
                proto_df.loc[label, m] = weight

    return proto_df
