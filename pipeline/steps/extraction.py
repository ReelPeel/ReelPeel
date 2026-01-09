"""
Step 2: Extract medical claims from a transcript using an LLM.

This step expects PipelineState.transcript to already be populated (e.g. via a
Whisper transcription step or MockTranscriptLoader). It prompts an LLM with a
template that should return a JSON array of claim strings. The response is
cleaned by stripping Markdown code fences and trailing commas, then parsed and
converted into Statement objects with sequential IDs.

Inputs:
- state.transcript: raw transcript text.
- config keys: prompt_template, model, temperature, max_tokens.

Outputs:
- state.statements: list[Statement] with id and text filled in.
- state.generated_at: UTC ISO timestamp updated after extraction.

Fallback behavior:
- If the LLM call fails or JSON parsing fails, it falls back to naive sentence
  splitting and keeps the first few non-empty sentences as claims.

Side effects:
- Logs the raw LLM response via PipelineStep.log_artifact for traceability.
"""

import json
import re
from datetime import datetime

from ..core.base import PipelineStep
from ..core.models import PipelineState, Statement


class TranscriptToStatementStep(PipelineStep):
    """
    Refactored Step 2: Extracts medical claims from a transcript using an LLM.
    """

    def execute(self, state: PipelineState) -> PipelineState:
        transcript = state.transcript
        if not transcript or not transcript.strip():
            print(f"[{self.__class__.__name__}] Warning: No transcript found in state.")
            return state

        print(f"[{self.__class__.__name__}] Starting extraction of medical claims...")


        # Use the prompt provided in the config
        prompt = self.config.get('prompt_template').format(transcript=transcript.strip())

        try:
            resp = self.llm.call(
                model=self.config.get('model'),
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 128),
                prompt=prompt,
            )

            self.log_artifact(f"Raw Output for Transcript Extraction", resp)

            cleaned_content = self._clean_json(resp)
            claims = json.loads(cleaned_content)

            # Map strings to Statement models
            state.statements = [
                Statement(id=i, text=text) for i, text in enumerate(claims, 1)
            ]

            print(f"[{self.__class__.__name__}] Extracted {len(state.statements)} claims.")

        except Exception as e:
            print(f"[{self.__class__.__name__}] Error during LLM extraction: {e}")
            # Fallback logic: naive sentence splitting
            rough = re.split(r"[.!?]\s+", transcript)
            fallback_claims = [s.strip() for s in rough if s.strip()][:3]
            state.statements = [
                Statement(id=i, text=text) for i, text in enumerate(fallback_claims, 1)
            ]
            print(f"[{self.__class__.__name__}] Applied sentence-split fallback.")

        # Update timestamp
        state.generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return state

    def _clean_json(self, content: str) -> str:
        """Internal helper to clean LLM output for JSON parsing."""
        # Remove Markdown-style code blocks
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        # Remove trailing commas before closing brackets
        content = re.sub(r",\s*(?=[\]}])", "", content)
        return content
