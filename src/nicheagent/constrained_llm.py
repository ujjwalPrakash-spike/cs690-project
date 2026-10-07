"""Constrained LLM Review Module for NicheAgent.

Escalates ONLY low-confidence cases to Gemini with a closed-world ontology
(allowed label vocabulary) and spatial neighborhood composition context.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from src.gemini.client import GeminiClient, ConstrainedAnnotationResult
from src.gemini.prompts import get_prompt
from src.nicheagent.nichecards import get_all_allowed_labels, DLPFC_NICHECARDS


class ConstrainedNicheAgentLLM:
    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()
        self.allowed_labels = get_all_allowed_labels()

    def review_low_confidence_niche(
        self,
        niche_id: str,
        marker_genes: List[str],
        proposed_label: str,
        neighborhood_context: str
    ) -> Dict[str, Any]:
        prompt = get_prompt(
            variant="constrained_nicheagent",
            genes=marker_genes,
            proposed_label=proposed_label,
            allowed_labels=self.allowed_labels,
            neighborhood_context=neighborhood_context
        )

        response_info = self.client.generate_annotation(
            prompt=prompt,
            schema=ConstrainedAnnotationResult,
            temperature=0.1  # Highly deterministic for constrained adjudication
        )

        parsed = response_info.get("parsed") or {}
        raw_label = parsed.get("biological_label", proposed_label)

        # Enforce closed-world ontology strictly: check exact or canonical aliases
        validated_label = self._map_to_allowed_ontology(raw_label, fallback=proposed_label)

        return {
            "niche_id": niche_id,
            "proposed_label": proposed_label,
            "raw_llm_label": raw_label,
            "adjudicated_label": validated_label,
            "short_justification": parsed.get("short_justification", ""),
            "confidence": parsed.get("confidence", 50),
            "raw_response": response_info.get("raw_response"),
            "timestamp": response_info.get("timestamp"),
            "status": response_info.get("status")
        }

    def _map_to_allowed_ontology(self, label: str, fallback: str) -> str:
        # Check direct match
        for allowed in self.allowed_labels:
            if label.strip().lower() == allowed.lower():
                return allowed

        # Check aliases in NicheCards
        for canonical, card in DLPFC_NICHECARDS.items():
            for alias in card["aliases"]:
                if alias.lower() in label.lower() or label.lower() in alias.lower():
                    return canonical

        # Strict fallback to proposed prototype label if out of bounds
        return fallback if fallback in self.allowed_labels else self.allowed_labels[0]
