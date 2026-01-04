import subprocess
from typing import Dict, Any

from ..app2.core.preprompts import *
from .llmconfigs import *

def is_related(statement_text: str, evidence_summary: str) -> bool:
    """Use Ollama+Gemma3 to judge whether evidence summary is relevant to the statement text."""

    try:
        prompt = PROMPT_TMPL_S6.format(statement=statement_text, evidence_summary=evidence_summary)
        res = CLIENT_6.chat.completions.create(
            model=MODEL_6,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMP_6,
            max_tokens=MAX_TOKENS_6,
        )
        reply: str = res.choices[0].message.content.strip()
        print(" --- Step6 Reduce Evidence ---> Response ---")
        print(f" Statement Text: {statement_text} ")
        print(f" Evidence Summary: {evidence_summary} ")
        print(f" Response: {reply} ")
        return reply.startswith("yes")
    except Exception as exc:
        print(f"Error during Ollama call Step6: {exc}")
        return False


    return False



    
    
    #return True

def reduce_to_evidence(data: Dict[str, Any]) -> Dict[str, Any]:
    print("Starting Step6: Reduce to Evidence")
    print("...")
    print("...")
    
    # Iterate through statements
    for stmt in data.get("statements", []):
        filtered = []
        statement_text = stmt.get("text", "")

        for ev in stmt.get("evidence", []):
            summary_text = ev.get("abstract", "").strip()
            # If there's a summary, only keep if related; if summary is empty, retain for manual review
            if summary_text:
                if is_related(statement_text, summary_text):
                    filtered.append(ev)
            else:
                filtered.append(ev)
        stmt["abstract"] = filtered

    return data