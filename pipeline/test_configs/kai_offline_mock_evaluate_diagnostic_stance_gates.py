from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = REPO_ROOT / "offline_mock" / "sweep_section_aware_stance"
REPORT_PATH = REPO_ROOT / "offline_mock" / "diagnostic_stance_gate_eval.json"

NEUTRAL_LABEL = "Neutral"
SUPPORTS_LABEL = "Supports"

DIAGNOSTIC_CLAIM_RE = re.compile(
    r"\b("
    r"test|tests|tested|testing|"
    r"diagnos(?:e|es|ed|ing|is|tic|tics)|"
    r"detect|detects|detected|detecting|"
    r"screen|screens|screened|screening|"
    r"check|checks|checked|checking|"
    r"identify|identifies|identified|identifying|"
    r"confirm|confirms|confirmed|confirming|"
    r"rule\s*out"
    r")\b",
    re.I,
)
PROCEDURE_SKIN_CLAIM_RE = re.compile(
    r"\b(apply|applying|rub|rubbing|place|placing|small\s+amount)\b.*\bskin\b"
    r"|\bskin\b.*\b(apply|applying|rub|rubbing|place|placing|small\s+amount)\b",
    re.I,
)
DIAGNOSTIC_UNCERTAINTY_RE = re.compile(
    r"\b("
    r"gold\s+standard|oral\s+food\s+challenge|"
    r"double[- ]blind|placebo[- ]controlled|"
    r"mainly\s+clinical|primarily\s+clinical|"
    r"not\s+(?:so\s+)?accurate|not\s+reliable|not\s+definitive|"
    r"limited\s+accuracy|poor\s+accuracy|single\s+diagnosis"
    r")\b",
    re.I,
)
FORMAL_DIAGNOSTIC_TEST_RE = re.compile(
    r"\b("
    r"skin[- ]prick|prick\s+test|patch\s+test|atopy\s+patch|"
    r"sige|specific\s+ige|oral\s+food\s+challenge|food\s+challenge|"
    r"double[- ]blind\s+placebo[- ]controlled"
    r")\b",
    re.I,
)


def summary_label(text: str) -> str:
    lowered = (text or "").lower()
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
        "does not support",
    ]
    support_hits = [
        "supports the statement",
        "support the statement",
        "supports",
        "may reduce",
        "help prevent",
        "reduced",
    ]
    if any(hit in lowered for hit in neutral_hits):
        return "Neutral"
    if any(hit in lowered for hit in refute_hits):
        return "Refutes/Neutral"
    if any(hit in lowered for hit in support_hits):
        return "Supports"
    return "Unknown"


def is_summary_conflict(expected: str, stance: str) -> bool:
    if expected == "Unknown":
        return False
    if expected == "Supports":
        return stance != "Supports"
    if expected == "Neutral":
        return stance != "Neutral"
    if expected == "Refutes/Neutral":
        return stance == "Supports"
    return False


def is_diagnostic_claim(claim: str) -> bool:
    return bool(
        DIAGNOSTIC_CLAIM_RE.search(claim or "")
        or PROCEDURE_SKIN_CLAIM_RE.search(claim or "")
    )


def has_diagnostic_uncertainty(text: str) -> bool:
    return bool(DIAGNOSTIC_UNCERTAINTY_RE.search(text or ""))


def has_method_mismatch(claim: str, evidence_text: str) -> bool:
    return bool(
        PROCEDURE_SKIN_CLAIM_RE.search(claim or "")
        and FORMAL_DIAGNOSTIC_TEST_RE.search(evidence_text or "")
    )


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for manifest_file in sorted(SWEEP_ROOT.glob("run_*/DT*/manifest.json")):
        process_file = manifest_file.parent / "process.json"
        if not process_file.exists():
            continue

        process_data = json.loads(process_file.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        for item in manifest.get("evidence_summaries", []) or []:
            response_file = manifest_file.parent / item.get("response_file", "")
            if not response_file.exists():
                continue
            response_data = json.loads(response_file.read_text(encoding="utf-8"))
            if "error" in response_data:
                continue

            statement = process_data["statements"][item["statement_index"]]
            evidence = statement["evidence"][item["evidence_index"]]
            claim = statement.get("text") or ""
            evidence_text = " ".join(
                [
                    evidence.get("title") or "",
                    evidence.get("abstract") or "",
                ]
            )
            stance = evidence.get("stance") or {}
            baseline_label = stance.get("abstract_label") or "Unknown"

            rows.append(
                {
                    "process_dir": str(manifest_file.parent.relative_to(REPO_ROOT)),
                    "statement_index": item["statement_index"],
                    "evidence_index": item["evidence_index"],
                    "statement": claim,
                    "pubmed_id": evidence.get("pubmed_id"),
                    "title": evidence.get("title"),
                    "relevance": float(evidence.get("relevance") or 0.0),
                    "baseline_label": baseline_label,
                    "summary": response_data.get("summary", ""),
                    "summary_label": summary_label(response_data.get("summary", "")),
                    "is_diagnostic_claim": is_diagnostic_claim(claim),
                    "has_diagnostic_uncertainty": has_diagnostic_uncertainty(evidence_text),
                    "has_method_mismatch": has_method_mismatch(claim, evidence_text),
                }
            )
    return rows


def apply_variant(row: Dict[str, Any], rule1: bool, rule2_scope: str) -> Tuple[str, List[str]]:
    label = row["baseline_label"]
    reasons: List[str] = []
    if label != SUPPORTS_LABEL:
        return label, reasons

    if rule1 and row["is_diagnostic_claim"] and (
        row["has_diagnostic_uncertainty"] or row["has_method_mismatch"]
    ):
        label = NEUTRAL_LABEL
        reasons.append("diagnostic_uncertainty_or_method_mismatch")

    rule2_enabled = rule2_scope in {"diagnostic", "global"}
    rule2_applies = rule2_scope == "global" or (
        rule2_scope == "diagnostic" and row["is_diagnostic_claim"]
    )
    if label == SUPPORTS_LABEL and rule2_enabled and rule2_applies and row["relevance"] < 0.80:
        label = NEUTRAL_LABEL
        reasons.append("low_relevance_support_cap")

    return label, reasons


def is_egg_skin_row(row: Dict[str, Any]) -> bool:
    claim = (row.get("statement") or "").lower()
    return "small amount of egg" in claim and "skin" in claim


def evaluate_variant(rows: List[Dict[str, Any]], name: str, rule1: bool, rule2_scope: str) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    summary_counts: Counter[str] = Counter()
    changed_examples: List[Dict[str, Any]] = []
    mismatch_examples: List[Dict[str, Any]] = []
    egg_skin_examples: List[Dict[str, Any]] = []
    changed = 0
    mismatches = 0
    egg_skin_mismatches = 0

    for row in rows:
        label, reasons = apply_variant(row, rule1=rule1, rule2_scope=rule2_scope)
        counts[label] += 1
        summary_counts[row["summary_label"]] += 1

        if label != row["baseline_label"]:
            changed += 1
            if len(changed_examples) < 15:
                changed_examples.append(
                    {
                        "process_dir": row["process_dir"],
                        "statement": row["statement"],
                        "pubmed_id": row["pubmed_id"],
                        "title": row["title"],
                        "relevance": row["relevance"],
                        "baseline_label": row["baseline_label"],
                        "new_label": label,
                        "reasons": reasons,
                        "summary_label": row["summary_label"],
                        "summary": row["summary"],
                    }
                )

        if is_summary_conflict(row["summary_label"], label):
            mismatches += 1
            if len(mismatch_examples) < 15:
                mismatch_examples.append(
                    {
                        "process_dir": row["process_dir"],
                        "statement": row["statement"],
                        "pubmed_id": row["pubmed_id"],
                        "title": row["title"],
                        "relevance": row["relevance"],
                        "label": label,
                        "summary_label": row["summary_label"],
                        "summary": row["summary"],
                    }
                )

        if is_egg_skin_row(row):
            if is_summary_conflict(row["summary_label"], label):
                egg_skin_mismatches += 1
            if len(egg_skin_examples) < 20:
                egg_skin_examples.append(
                    {
                        "process_dir": row["process_dir"],
                        "pubmed_id": row["pubmed_id"],
                        "relevance": row["relevance"],
                        "baseline_label": row["baseline_label"],
                        "new_label": label,
                        "reasons": reasons,
                        "summary_label": row["summary_label"],
                        "title": row["title"],
                    }
                )

    return {
        "name": name,
        "rule1_diagnostic_uncertainty_gate": rule1,
        "rule2_relevance_scope": rule2_scope,
        "items": len(rows),
        "stance_counts": dict(counts),
        "summary_direction_counts": dict(summary_counts),
        "mismatches": mismatches,
        "mismatch_rate": round(mismatches / len(rows), 4) if rows else 0.0,
        "changed_labels": changed,
        "egg_skin_mismatches": egg_skin_mismatches,
        "changed_examples": changed_examples,
        "mismatch_examples": mismatch_examples,
        "egg_skin_examples": egg_skin_examples,
    }


def print_table(results: List[Dict[str, Any]]) -> None:
    print("| Variant | Supports | Refutes | Neutral | Mismatch | Changed labels | Egg-skin mismatches |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        counts = result["stance_counts"]
        print(
            f"| {result['name']} | "
            f"{counts.get('Supports', 0)} | "
            f"{counts.get('Refutes', 0)} | "
            f"{counts.get('Neutral', 0)} | "
            f"{result['mismatches']} | "
            f"{result['changed_labels']} | "
            f"{result['egg_skin_mismatches']} |"
        )


def main() -> None:
    rows = load_rows()
    variants = [
        ("Current section-aware", False, "none"),
        ("Rule 1 only", True, "none"),
        ("Rule 2 diagnostic only", False, "diagnostic"),
        ("Rule 1 + Rule 2 diagnostic", True, "diagnostic"),
        ("Rule 2 global", False, "global"),
        ("Rule 1 + Rule 2 global", True, "global"),
    ]
    results = [
        evaluate_variant(rows, name=name, rule1=rule1, rule2_scope=rule2_scope)
        for name, rule1, rule2_scope in variants
    ]

    report = {
        "source_sweep": str(SWEEP_ROOT.relative_to(REPO_ROOT)),
        "items": len(rows),
        "rules": {
            "rule1": "Supports -> Neutral for diagnostic/test claims with uncertainty or method mismatch cues.",
            "rule2": "Supports -> Neutral for relevance < 0.80, tested diagnostic-only and global.",
            "downgrade_target": "Neutral",
        },
        "results": results,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_table(results)
    print()
    print(f"Report written: {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
