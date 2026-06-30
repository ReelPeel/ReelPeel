from __future__ import annotations

import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.core.models import PipelineState
from pipeline.steps.stance import StanceEvidenceStep


OFFLINE_ROOT = REPO_ROOT / "offline_mock"
REPORT_PATH = OFFLINE_ROOT / "section_stance_eval.json"
MODEL_NAME = "cnut1648/biolinkbert-mednli"
MAX_LENGTH = 512
CHUNK_OVERLAP = 64

SECTION_HEADERS = [
    "CONCLUSIONS AND CLINICAL RELEVANCE",
    "DATA EXTRACTION AND SYNTHESIS",
    "CONCLUSIONS AND RELEVANCE",
    "DATA COLLECTION AND ANALYSIS",
    "MAIN OUTCOMES AND MEASURES",
    "MAIN OUTCOME AND MEASURES",
    "MAIN OUTCOME AND MEASURE",
    "MATERIALS AND METHODS",
    "BACKGROUND AND OBJECTIVE",
    "PURPOSE OF REVIEW",
    "SELECTION CRITERIA",
    "TRIAL REGISTRATION",
    "RECENT FINDINGS",
    "STUDY SELECTION",
    "SEARCH METHODS",
    "MATERIAL AND METHOD",
    "DATA EXTRACTION",
    "MAIN RESULTS",
    "STUDY DESIGN",
    "DATA SOURCES",
    "INTERPRETATION",
    "INTERVENTIONS",
    "CONCLUSIONS",
    "LIMITATIONS",
    "OBJECTIVES",
    "CONCLUSION",
    "IMPORTANCE",
    "BACKGROUND",
    "OBJECTIVE",
    "RATIONALE",
    "UNLABELLED",
    "EXPOSURES",
    "FINDINGS",
    "SUMMARY",
    "METHODS",
    "RESULTS",
    "FUNDING",
    "DESIGN",
    "AIMS",
    "AIM",
]

SECTION_RE = re.compile(
    r"(^|(?<=[.!?])\s+)("
    + "|".join(re.escape(h) for h in SECTION_HEADERS)
    + r")\s*:",
    re.I,
)


def stance_label(ev: Dict[str, Any]) -> Optional[str]:
    return (ev.get("stance") or {}).get("abstract_label")


def summary_label(text: str) -> str:
    t = text.lower()
    neutral_hits = [
        "unclear",
        "not definitively",
        "does not directly",
        "doesn't directly",
        "not directly support or refute",
    ]
    refute_hits = [
        "contradict",
        "incorrect",
        "refute",
        "not through skin",
        "not directly support",
    ]
    support_hits = [
        "supports the statement",
        "support the statement",
        "supports",
        "may reduce",
        "help prevent",
        "reduced",
    ]
    if any(hit in t for hit in neutral_hits):
        return "Neutral"
    if any(hit in t for hit in refute_hits):
        return "Refutes/Neutral"
    if any(hit in t for hit in support_hits):
        return "Supports"
    return "Unknown"


def is_summary_conflict(expected: Optional[str], stance: Optional[str]) -> bool:
    if not expected or expected == "Unknown" or not stance:
        return False
    if expected == "Supports":
        return stance != "Supports"
    if expected == "Neutral":
        return stance != "Neutral"
    if expected == "Refutes/Neutral":
        return stance == "Supports"
    return False


def extract_sections(text: str) -> List[Tuple[str, str]]:
    text = " ".join(str(text or "").split())
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("UNKNOWN", text)] if text else []

    sections: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("UNKNOWN", preamble))
    for idx, match in enumerate(matches):
        label = match.group(2).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((label, section_text))
    return sections


def chunk_count(tokenizer: Any, claim: str, text: str, use_sections: bool) -> Tuple[List[str], int]:
    statement_token_count = len(tokenizer.encode(claim or "", add_special_tokens=False))
    chunk_size = MAX_LENGTH - (statement_token_count + 16)
    if chunk_size < 1:
        chunk_size = 1
    chunk_overlap = CHUNK_OVERLAP
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)
    chunk_step = max(1, chunk_size - chunk_overlap)

    sections = extract_sections(text) if use_sections else [("UNKNOWN", " ".join(str(text or "").split()))]
    labels: List[str] = []
    count = 0
    for label, section_text in sections:
        token_ids = tokenizer.encode(section_text, add_special_tokens=False)
        if not token_ids:
            continue
        labels.append(label)
        start = 0
        while start < len(token_ids):
            count += 1
            if start + chunk_size >= len(token_ids):
                break
            start += chunk_step
    return labels or ["UNKNOWN"], count


def load_summary_map(process_dir: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    manifest_path = process_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.load(open(manifest_path))
    except Exception:
        return {}

    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for item in manifest.get("evidence_summaries", []) or []:
        response_file = process_dir / item.get("response_file", "")
        if not response_file.exists():
            continue
        try:
            summary = json.load(open(response_file)).get("summary", "")
        except Exception:
            continue
        out[(item["statement_index"], item["evidence_index"])] = {
            "summary": summary,
            "summary_label": summary_label(summary),
        }
    return out


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    process_files = sorted(OFFLINE_ROOT.glob("**/process.json"))

    fixed_step = StanceEvidenceStep(
        {
            "model_name": MODEL_NAME,
            "use_fp16": True,
            "batch_size": 16,
            "max_length": MAX_LENGTH,
            "evidence_fields": ["abstract"],
            "section_chunking_enabled": False,
        }
    )
    section_step = StanceEvidenceStep(
        {
            "model_name": MODEL_NAME,
            "use_fp16": True,
            "batch_size": 16,
            "max_length": MAX_LENGTH,
            "evidence_fields": ["abstract"],
            "section_chunking_enabled": True,
        }
    )

    transition_old_fixed: Counter[str] = Counter()
    transition_fixed_section: Counter[str] = Counter()
    transition_old_section: Counter[str] = Counter()
    old_counts: Counter[str] = Counter()
    fixed_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    section_header_counts: Counter[str] = Counter()
    summary_conflicts_fixed = 0
    summary_conflicts_section = 0
    summary_items = 0
    evidence_items: List[Dict[str, Any]] = []

    for process_file in process_files:
        original = json.load(open(process_file))
        fixed_state = fixed_step.run(PipelineState.model_validate(deepcopy(original)))
        section_state = section_step.run(PipelineState.model_validate(deepcopy(original)))
        fixed_json = fixed_state.model_dump(mode="json")
        section_json = section_state.model_dump(mode="json")
        summary_map = load_summary_map(process_file.parent)

        for si, original_stmt in enumerate(original.get("statements", []) or []):
            fixed_stmt = fixed_json.get("statements", [])[si]
            section_stmt = section_json.get("statements", [])[si]
            claim = original_stmt.get("text") or ""

            for ei, original_ev in enumerate(original_stmt.get("evidence") or []):
                fixed_ev = fixed_stmt.get("evidence", [])[ei]
                section_ev = section_stmt.get("evidence", [])[ei]
                abstract = original_ev.get("abstract") or ""

                old_label = stance_label(original_ev)
                fixed_label = stance_label(fixed_ev)
                section_label = stance_label(section_ev)
                labels, section_chunks = chunk_count(tokenizer, claim, abstract, use_sections=True)
                _, fixed_chunks = chunk_count(tokenizer, claim, abstract, use_sections=False)
                summary_info = summary_map.get((si, ei), {})
                expected = summary_info.get("summary_label")

                transition_old_fixed[f"{old_label}->{fixed_label}"] += 1
                transition_fixed_section[f"{fixed_label}->{section_label}"] += 1
                transition_old_section[f"{old_label}->{section_label}"] += 1
                old_counts[str(old_label)] += 1
                fixed_counts[str(fixed_label)] += 1
                section_counts[str(section_label)] += 1
                section_header_counts.update(labels)

                if expected:
                    summary_items += 1
                    if is_summary_conflict(expected, fixed_label):
                        summary_conflicts_fixed += 1
                    if is_summary_conflict(expected, section_label):
                        summary_conflicts_section += 1

                evidence_items.append(
                    {
                        "process_file": str(process_file.relative_to(REPO_ROOT)),
                        "statement_index": si,
                        "evidence_index": ei,
                        "statement": claim,
                        "pubmed_id": original_ev.get("pubmed_id"),
                        "title": original_ev.get("title"),
                        "old_stance": old_label,
                        "fixed_stance": fixed_label,
                        "section_stance": section_label,
                        "fixed_probs": fixed_ev.get("stance"),
                        "section_probs": section_ev.get("stance"),
                        "section_labels": labels,
                        "has_sections": labels != ["UNKNOWN"],
                        "fixed_chunk_count": fixed_chunks,
                        "section_chunk_count": section_chunks,
                        "summary_label": expected,
                        "summary": summary_info.get("summary"),
                    }
                )

    report = {
        "model": MODEL_NAME,
        "process_files": len(process_files),
        "evidence_items": len(evidence_items),
        "summary_items": summary_items,
        "counts": {
            "old": dict(old_counts),
            "fixed": dict(fixed_counts),
            "section": dict(section_counts),
        },
        "transitions": {
            "old_to_fixed": dict(transition_old_fixed),
            "fixed_to_section": dict(transition_fixed_section),
            "old_to_section": dict(transition_old_section),
        },
        "sections": {
            "header_counts": dict(section_header_counts),
            "with_sections": sum(1 for item in evidence_items if item["has_sections"]),
            "unknown": sum(1 for item in evidence_items if not item["has_sections"]),
        },
        "summary_conflicts": {
            "fixed": summary_conflicts_fixed,
            "section": summary_conflicts_section,
        },
        "items": evidence_items,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2, ensure_ascii=False))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
