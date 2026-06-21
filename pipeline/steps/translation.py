"""Normalize claim language into German and English before retrieval/classification."""

import json
import re
from typing import List, Tuple

from ..core.base import PipelineStep
from ..core.models import PipelineState


_TRANSLATION_SCHEMA = '{"de": "<German translation>", "en": "<English translation>"}'


def _strip_code_fence(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_translation_response(response: str) -> Tuple[str, str]:
    try:
        payload = json.loads(_strip_code_fence(response))
    except Exception as exc:
        raise ValueError("Translation response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Translation response must be a JSON object.")
    de = str(payload.get("de", "")).strip()
    en = str(payload.get("en", "")).strip()
    if not de or not en:
        raise ValueError("Translation response must contain non-empty 'de' and 'en'.")
    return de, en


class TranslateClaimStep(PipelineStep):
    """Translate each claim to German and English and build a normalized comparison text."""

    def execute(self, state: PipelineState) -> PipelineState:
        model = self.config.get("model")
        max_retries = int(self.config.get("max_retries", 1))
        if not model:
            raise ValueError("TranslateClaimStep requires an LLM model.")

        for stmt in state.statements:
            original = (stmt.text or "").strip()
            if not original:
                continue
            prompt = f'''Translate the medical claim below into German and English.
Return exactly one JSON object with this shape:
{_TRANSLATION_SCHEMA}
Rules:
- Preserve the medical meaning.
- Keep numbers, units, ages, timing, and negations exact.
- Do not add explanation or markdown.

CLAIM:
{original}
'''
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    response = self.llm.call(
                        prompt=prompt,
                        model=model,
                        temperature=float(self.config.get("temperature", 0.0)),
                        max_tokens=int(self.config.get("max_tokens", 256)),
                    )
                    de, en = parse_translation_response(response)
                    stmt.translated_text_de = de
                    stmt.translated_text_en = en
                    stmt.normalized_text = f"German: {de}\nEnglish: {en}"
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < max_retries:
                        prompt += (
                            "\n\nYour previous response was invalid. Return only the required JSON object."
                            f" Validation error: {exc}"
                        )
            if last_error is not None:
                stmt.translated_text_de = original
                stmt.translated_text_en = original
                stmt.normalized_text = original
        return state
