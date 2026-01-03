from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = "/api/generate"


PROMPT_TEMPLATE = """You convert fact-check claims written as questions into declarative statements.

Rules:
- Preserve the original meaning exactly.
- Preserve polarity: DO NOT introduce or remove negation.
  - If the question contains negation (e.g., "can't", "doesn't", "not"), keep it in the statement.
  - If the question is affirmative, the statement must be affirmative (no new "not", "no", "cannot", etc.).
- Do not add new claims, numbers, or qualifiers that are not present.
- Keep it a single sentence whenever possible.
- Output ONLY the final statement text (no explanations, no quotes, no bullet points).

Examples:
Q: "Can masks reduce corona infections when worn by a large proportion of the population?"
A: "Masks can reduce corona infections when worn by a large proportion of the population."

Q: "Can't masks reduce corona infections when worn by a large proportion of the population?"
A: "Masks cannot reduce corona infections when worn by a large proportion of the population."

Q: "Do surgical masks not reduce the risk of infection?"
A: "Surgical masks do not reduce the risk of infection."

Now convert:
Q: "{claim}"
A:"""


def ollama_generate_statement(
    claim: str,
    model: str,
    base_url: str = OLLAMA_DEFAULT_URL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout_s: int = 120,
) -> str:
    """
    Calls Ollama /api/generate (non-streaming) and returns the model's response text.
    """
    url = base_url.rstrip("/") + OLLAMA_GENERATE_ENDPOINT

    prompt = PROMPT_TEMPLATE.format(claim=claim.strip())

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()

    data = r.json()
    text = (data.get("response") or "").strip()

    # Safety cleanup: remove wrapping quotes or leading "A:" if the model includes it.
    text = text.strip().lstrip("A:").strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Ensure it's not accidentally empty
    if not text:
        # Fallback: just remove trailing '?' to at least produce something
        text = claim.strip().rstrip("?").strip()

    return text


def build_long_form(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Create a long-form dataframe with columns:
      row_id, lang, claim, label
    """
    claim_col = f"{lang}_claim"
    if claim_col not in df.columns:
        raise ValueError(f"Missing column '{claim_col}' in input CSV.")

    if "label" not in df.columns:
        raise ValueError("Missing column 'label' in input CSV.")

    out = pd.DataFrame(
        {
            "row_id": df.index.astype(int),
            "lang": lang,
            "claim": df[claim_col].astype(str),
            "label": df["label"],
        }
    )

    # Drop rows with empty claims (still preserving relative order of remaining rows)
    out["claim"] = out["claim"].fillna("").str.strip()
    out = out[out["claim"] != ""].copy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV.")
    ap.add_argument("--output", required=True, help="Path to output CSV.")
    ap.add_argument("--lang", choices=["en", "de", "both"], default="en", help="Which claim column(s) to use.")
    ap.add_argument("--model", default="gemma3:27b", help="Ollama model name, e.g. gemma3:27b")
    ap.add_argument("--ollama-url", default=OLLAMA_DEFAULT_URL, help="Ollama base URL, e.g. http://localhost:11434")
    ap.add_argument("--no-llm", action="store_true", help="Only extract claim+label (no statement generation).")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for quick tests (0 = no limit).")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if args.lang == "both":
        long_df = pd.concat([build_long_form(df, "en")], ignore_index=True)
    else:
        long_df = build_long_form(df, args.lang)

    # Optional limit (keeps order)
    if args.limit and args.limit > 0:
        long_df = long_df.iloc[: args.limit].copy()


    statements: List[str] = []
    # only 5 rows
    for i, row in long_df.iterrows():
        claim = str(row["claim"])
        try:
            stmt = ollama_generate_statement(
                claim=claim,
                model=args.model,
                base_url=args.ollama_url,
                temperature=0.0,
                top_p=1.0,
                timeout_s=180,
            )
        except Exception as e:
            # Hard fallback: keep order, but use minimally transformed claim
            stmt = claim.strip().rstrip("?").strip()
            print(f"[WARN] LLM failed at row_id={row['row_id']} lang={row['lang']}: {e}", file=sys.stderr)

        statements.append(stmt)

        # Lightweight progress
        if (len(statements) % 25) == 0:
            print(f"Processed {len(statements)}/{len(long_df)}")

    out_df = long_df.copy()
    out_df["statement"] = statements

    # Keep a clear column order
    out_df = out_df[["statement", "label"]]
    records = out_df.to_dict(orient="records")  # keeps row order

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())