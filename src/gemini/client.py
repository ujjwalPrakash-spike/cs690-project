"""Isolated Gemini API Client using Google's official google-genai SDK.

Keeps API communication strictly modular and decoupled from the rest
of the pipeline.
"""

import os
import json
import time
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import get_gemini_api_key, get_gemini_model


class FreeFormAnnotationResult(BaseModel):
    biological_label: str = Field(description="Predicted biological niche or cell type label")
    short_justification: str = Field(description="Concise justification based strictly on the provided marker genes")
    confidence: int = Field(description="Confidence integer score between 0 and 100")


class ConstrainedAnnotationResult(BaseModel):
    biological_label: str = Field(description="Selected label from the allowed vocabulary")
    short_justification: str = Field(description="Concise reasoning considering marker genes and neighborhood context")
    confidence: int = Field(description="Confidence integer score between 0 and 100")


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or get_gemini_api_key()
        self.model = model or get_gemini_model()
        self.client = genai.Client(api_key=self.api_key)

    def generate_annotation(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = FreeFormAnnotationResult,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Calls Gemini with structured JSON output schema and error handling."""
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if schema is not None:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = schema

        config = types.GenerateContentConfig(**config_params)

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
                raw_text = response.text or ""
                parsed = None
                if schema is not None:
                    try:
                        parsed = json.loads(raw_text)
                    except json.JSONDecodeError:
                        # Fallback parsing in case of markdown wrap
                        clean_text = raw_text.strip()
                        if clean_text.startswith("```json"):
                            clean_text = clean_text[7:]
                        if clean_text.startswith("```"):
                            clean_text = clean_text[3:]
                        if clean_text.endswith("```"):
                            clean_text = clean_text[:-3]
                        parsed = json.loads(clean_text.strip())

                return {
                    "model": self.model,
                    "prompt": prompt,
                    "raw_response": raw_text,
                    "parsed": parsed,
                    "status": "success",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            except Exception as e:
                last_exception = e
                time.sleep(1.5 * (attempt + 1))

        return {
            "model": self.model,
            "prompt": prompt,
            "raw_response": str(last_exception),
            "parsed": None,
            "status": "error",
            "error_message": str(last_exception),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
