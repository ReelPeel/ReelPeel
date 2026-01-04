#!/usr/bin/env python3
"""
STEP 3 – STATEMENT → PUBMED QUERY
────────────────────────────────
• Reads a FAXXCHEXX JSON that already contains *medical* statements
• For every statement without a "query", asks a local LLM to create
  a concise PubMed search string (plain words, 4-8 terms, one line)
• Writes the updated JSON and time-stamp
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
# ───────── config ─────────

TRAILING_COMMAS_RE = re.compile(r",\s*(?=[\]}])")

# PICO = Patient/Population, Intervention, Comparison und Outcome

# first visible line that starts with a letter
QUERY_LINE_RE = re.compile(r"^(?!\s*$).+", re.MULTILINE)

# ───────── helpers ─────────
def load_json(path: str) -> Dict[str, Any]:
    raw = open(path, "r", encoding="utf-8").read()
    return json.loads(TRAILING_COMMAS_RE.sub("", raw))


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sanitise_words(words: List[str]) -> List[str]:
    """Remove Boolean operators and empty strings, keep unique order."""
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if lw in {"and", "or", ""}:
            continue
        if lw not in seen:
            seen.add(lw)
            out.append(w)
    return out



def clean_query(raw: str, fallback_source: str) -> str:
    """
    • Take the first non‐empty line from the model output
      and strip surrounding backticks / quotes.
    • Do NOT drop AND/OR or truncate to a fixed word count.
    • If no valid line is found, fall back to a crude keyword list
      (deduplicated, up to 8 terms) drawn from fallback_source.
    """
    m = QUERY_LINE_RE.search(raw)
    if m:
        # strip backticks and quotes, then collapse internal whitespace
        q = m.group(0).strip("`'\" ").replace("\n", " ").strip()
        return q

    # ── fallback: dedupe ≤8 keywords from fallback_source ──
    kws = re.findall(r"[A-Za-z']+", fallback_source)
    # lower‐case, remove duplicates while preserving order
    seen = set()
    deduped = []
    for w in kws:
        w_lower = w.lower()
        if w_lower not in seen:
            seen.add(w_lower)
            deduped.append(w_lower)
    return " ".join(deduped[:8])


def make_query(claim: str, client: openai.OpenAI) -> str:
    prompt = PROMPT_TMPL_S3.format(claim=claim)
    try:
        resp = CLIENT_3.chat.completions.create(
            model=MODEL_3,
            temperature=TEMP_3,
            max_tokens=MAX_TOKENS_3,
            stop=["\n"],                         # cut at first newline
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content
        print(" --- Step3 Statement to Query ---> LLM Output: ---")
        print(raw)
        
        return clean_query(raw, claim)
    except Exception as e:
        
        print("ERROR ERROR ERROR --> FALLBACK \n")
        print(e)
        return " "

# ───────── core ─────────
def update_query(data: Dict[str, Any]) -> Dict[str, Any]:
    print("Starting Step3: Statement to PubMed Query")
    print("...")
    print("...")
    
    new_queries: List[str] = []

    for stmt in data.get("statements", []):
        if stmt.get("query"):            # already filled
            continue
        query = make_query(stmt.get("text", ""), CLIENT_3)
        stmt["query"] = query
        new_queries.append(query)

    data["generated_at"] = (
        dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    )

    return data

