from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

# If you want relaxed JSON loading here too, keep it in your existing IO utilities.
# This module only transforms the in-memory dict.

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


# Heuristic evidence hierarchy weights (0..1).
# We compute the MAX match across all publication types on the evidence item.
PUBTYPE_RULES: List[Tuple[re.Pattern, float, str]] = [
    # Highest: systematic syntheses
    (re.compile(r"\bmeta[- ]analysis\b"), 0.95, "meta-analysis"),
    (re.compile(r"\bsystematic review\b"), 0.95, "systematic review"),
    (re.compile(r"\bnetwork meta[- ]analysis\b"), 0.95, "network meta-analysis"),
    (re.compile(r"\bumbrella review\b"), 0.95, "umbrella review"),

    # Randomized evidence
    (re.compile(r"\brandomized controlled trial\b|\brandomised controlled trial\b"), 0.90, "RCT"),
    (re.compile(r"\bcontrolled clinical trial\b"), 0.88, "controlled clinical trial"),
    (re.compile(r"\bclinical trial,\s*phase iii\b"), 0.90, "phase III clinical trial"),
    (re.compile(r"\bclinical trial,\s*phase ii\b"), 0.86, "phase II clinical trial"),
    (re.compile(r"\bclinical trial,\s*phase iv\b"), 0.84, "phase IV clinical trial"),
    (re.compile(r"\bclinical trial\b"), 0.85, "clinical trial"),

    # Guidelines / consensus
    (re.compile(r"\bpractice guideline\b"), 0.88, "practice guideline"),
    (re.compile(r"\bguideline\b"), 0.86, "guideline"),
    (re.compile(r"\bconsensus development conference\b|\bconsensus\b"), 0.82, "consensus"),

    # Observational analytic
    (re.compile(r"\bcohort\b|\bprospective\b|\bretrospective\b"), 0.75, "cohort/prospective/retrospective"),
    (re.compile(r"\bcase[- ]control\b"), 0.70, "case-control"),
    (re.compile(r"\bcross[- ]sectional\b"), 0.65, "cross-sectional"),
    (re.compile(r"\bobservational study\b"), 0.65, "observational study"),
    (re.compile(r"\bcomparative study\b"), 0.65, "comparative study"),

    # Narrative reviews (below systematic review/meta-analysis)
    (re.compile(r"^\s*review\s*$|\breview\b"), 0.62, "narrative review"),

    # Lower: descriptive / anecdotal
    (re.compile(r"\bcase reports?\b"), 0.45, "case report"),
    (re.compile(r"\bcase series\b"), 0.45, "case series"),

    # Very low: opinion/correspondence
    (re.compile(r"\beditorial\b"), 0.35, "editorial"),
    (re.compile(r"\bcomment\b"), 0.35, "comment"),
    (re.compile(r"\bletter\b"), 0.35, "letter"),
    (re.compile(r"\bexpert opinion\b"), 0.30, "expert opinion"),
]


def _iter_evidence_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return references to evidence dicts in:
      - data["statements"][i]["evidence"][j]
      - data["evidence"][k] (if present)
    """
    items: List[Dict[str, Any]] = []

    for stmt in (data.get("statements") or []):
        for ev in (stmt.get("evidence") or []):
            if isinstance(ev, dict):
                items.append(ev)

    for ev in (data.get("evidence") or []):
        if isinstance(ev, dict):
            items.append(ev)

    return items


def _weight_from_pubtypes(pubtypes: Any, default: float) -> Tuple[float, Optional[str]]:
    """
    Compute a weight from PubMed publication types.
    - pubtypes may be list[str], str, None
    - returns (weight, reason)
    """
    if pubtypes is None:
        return default, None

    if isinstance(pubtypes, str):
        pts = [pubtypes]
    elif isinstance(pubtypes, list):
        pts = [str(x) for x in pubtypes if x is not None]
    else:
        pts = [str(pubtypes)]

    pts = [p for p in pts if str(p).strip()]
    if not pts:
        return default, None

    best_w = 0.0
    best_reason: Optional[str] = None

    for pt in pts:
        n = _norm(pt)
        for rx, w, reason in PUBTYPE_RULES:
            if rx.search(n) and w > best_w:
                best_w = w

    if best_w <= 0.0:
        return default, None

    return best_w


def pubtype_to_weight(data: Dict[str, Any], default: float = 0.5) -> Dict[str, Any]:
    """
    Add a numeric evidence weight per evidence item:
      ev["weight"] = float
      
    Default if type is None/unknown/not matched: 0.5
    """
    print("Starting StepX: PubType to Weight")
    print("...")
    print("...")

    for ev in _iter_evidence_items(data):
        w = _weight_from_pubtypes(ev.get("type"), default=default)
        ev["weight"] = float(w)

    data["weighted_at"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return data
