#!/usr/bin/env python3
"""
TRANSCRIPT → MEDICAL-STATEMENTS
──────────────────────────────
• Reads a FAXXCHEXX-style JSON file that contains a "transcript".
• Asks a local LLM to extract ONLY medically-relevant factual claims.
• Replaces/creates the "statements" list; each entry has the usual
  skeleton (id, text, verdict=None, …).
• No PubMed / evidence handling.
• Writes the updated JSON to the output path and stamps generated_at.
"""

import argparse
import datetime as dt
import json
import re
import sys
from typing import Any, Dict, List

import openai

from ..app2.core.preprompts import *
from .llmconfigs import *


TRAILING_COMMAS_RE = re.compile(r",\s*(?=[\]}])")

# ────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────
def load_json_relaxed(path: str) -> Any:
    raw = open(path, "r", encoding="utf-8").read()
    return json.loads(TRAILING_COMMAS_RE.sub("", raw))


def clean_json_content(content: str) -> str:
    # Remove Markdown-style code blocks if present
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)


def split_into_medical_statements(transcript: str) -> List[str]:
    """LLM → JSON array of medically-relevant claims."""
    print("Starting Step2: Transcript to Medical Statements")
    print("...")
    print("...")
    
    prompt = PROMPT_TMPL_S2.format(transcript=transcript.strip())
    try:
        resp = CLIENT_2.chat.completions.create(
            model=MODEL_2,
            temperature=TEMP_2,
            max_tokens=MAX_TOKENS_2, 
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip()
        cleaned = clean_json_content(content)
        print(" --- Step2 Transcript to Statements ---> LLM Output: ---")
        print(cleaned)
        return json.loads(cleaned)
    except Exception as e:
        print(e)
        print("FALLBACK")
        # Fallback: naïve sentence split; real pipeline should re-prompt or log
        rough = re.split(r"[.!?]\s+", transcript)
        return [s.strip() for s in rough if s.strip()]


def statement_skeleton(text: str, idx: int) -> Dict[str, Any]:
    return {
        "id": idx,
        "text": text,
        "verdict": None,
        "rationale": None,
        "confidence": None,
        "query": None,
        "evidence": []
    }


# ────────────────────────────────────────────────────────────────────
# Core
# ────────────────────────────────────────────────────────────────────
def update_statements(data: Dict[str, Any]) -> Dict[str, Any]:
    transcript: str = data.get("transcript", "")
    if not transcript.strip():
        print("Input JSON has no transcript text.", file=sys.stderr)


    claims = split_into_medical_statements(transcript)
    data["statements"] = [
        statement_skeleton(text, i) for i, text in enumerate(claims, 1)
    ]

    data["generated_at"] = (
        dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    )

    return data
