from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import Dict, Any
import openai

from .preprompts import *
from .llmconfigs import *



def statement_to_truthness(data: Dict[str, Any]) -> Dict[str, Any]:
    print("Starting Step7: Statement to Truthness")
    print("...")
    print("...")
    
    transcript = data["transcript"]
    # -------- iterate over statements --------
    for stmt in data.get("statements", []):
        claim_text: str = stmt.get("text", "").strip()
        evidences   = stmt.get("evidence", [])

        # build evidence block (summaries only)
        evidence_lines = []
        for ev in evidences:
            pmid = ev.get("pubmed_id") or "N/A"
            summary = (ev.get("abstract") or "").strip().replace("\n", " ")
            print(" --- Step7 Statement to Truthness --> Evidence Summary: ---")
            print(f"Evidence: {summary}")
            
            evidence_lines.append(f"- PMID {pmid}: {summary}")
        evidence_block = "\n".join(evidence_lines) or "No evidence provided."

        print(" --- Step7 Statement to Truthness --> Evidence-Block and claim_text: ---")
        print(f"Claim Text: {claim_text}")
        print(f"Evidence: {evidence_block}")
        try:
            prompt = PROMPT_TMPL_S7.format(claim_text=claim_text, evidence_block=evidence_block, transcript=transcript)
            res = CLIENT_7.chat.completions.create(
                model=MODEL_7,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMP_7,
                max_tokens=MAX_TOKENS_7,
            )
            reply: str = res.choices[0].message.content.strip()
        except Exception as exc:
            # fall back to uncertain if model call fails
            stmt["verdict"]    = "error - Fallback to uncertain"
            stmt["confidence"] = 0.0
            stmt["rationale"]  = f"Model call failed: {exc}"
            continue

        # -------- parse reply --------
        verdict_match = re.search(r"VERDICT:\s*(true|false|uncertain)", reply, re.I)
        print(" --- Step7-Statement to Truthness --> Model output --- ")
        print(f"Model output:\n{reply}  ")
        
        score_match   = re.search(r"FINALSCORE:\s*([0-1](?:\.\d+)?)", reply, re.I)
        print(" --- Extracted Information from prompt: ---")
        print(f"Parsed verdict: {verdict_match.group(1) if verdict_match else 'N/A'} ")
        
        if verdict_match and score_match:
            print(f"Parsed score: {score_match.group(1)}")
            stmt["verdict"]    = verdict_match.group(1).lower()
            stmt["confidence"] = float(score_match.group(1))
            # keep the raw model reply for transparency
            stmt["rationale"]  = reply
        else:
            # model did not follow instructions → mark as uncertain
            print(f"ERROR ERROR ERROR --> Unparsable model output:\n{reply}")
            stmt["verdict"]    = "uncertain"
            stmt["confidence"] = 0.0
            stmt["rationale"]  = f"Unparsable model output:\n{reply}"

    return data
