import json
import time
import copy
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any
import random
from app2.core.models import PipelineState
from app2.core.orchestrator import PipelineOrchestrator

import app2.test_configs.raw_eval_config as config_module


# Mapping: pipeline verdict -> dataset label
# Order requested: true -> 0, uncertain -> 1, false -> 2
VERDICT_TO_LABEL = {
    "true": 0,
    "uncertain": 1,
    "false": 2,
}

# Optional: common aliases if your pipeline emits variants
VERDICT_ALIASES = {
    "unknown": "uncertain",
    "inconclusive": "uncertain",
    "partly true": "uncertain",
    "partially true": "uncertain",
    "mostly true": "uncertain",
    "mostly false": "uncertain",
}


def normalize_verdict(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    v = v.strip().lower()
    v = VERDICT_ALIASES.get(v, v)
    return v


def set_mock_statement(cfg: dict, text: str, stmt_id: int = 1) -> None:
    """Mutate cfg in-place: replace the mock_statement text with `text`."""
    for step in cfg.get("steps", []):
        if step.get("type") == "mock_statements":
            step.setdefault("settings", {})
            step["settings"]["statements"] = [{"id": stmt_id, "text": text}]
            return
    raise ValueError("No step with type='mock_statements' found in config.")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      1) JSON list:        [{"statement": "...", "label": 0}, ...]
      2) JSONL:            one JSON object per line
      3) TSV-like text:    "<statement>\\t<label>" or "<label>\\t<statement>"
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # Case 1: JSON list
    if raw[0] == "[":
        data = json.loads(raw)
        return [{"statement": x["statement"], "label": int(x["label"])} for x in data]

    rows: List[Dict[str, Any]] = []
    for ln, line in enumerate(raw.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue

        # Case 2: JSONL
        if s[0] == "{":
            obj = json.loads(s)
            stmt = obj.get("statement", obj.get("text"))
            if stmt is None or "label" not in obj:
                raise ValueError(f"JSONL line {ln} missing 'statement'/'text' or 'label'.")
            rows.append({"statement": stmt, "label": int(obj["label"])})
            continue

        # Case 3: TSV-like
        parts = [p.strip() for p in s.split("\t")]
        if len(parts) != 2:
            raise ValueError(
                f"Unrecognized dataset line format at line {ln}. "
                f"Expected JSON, JSONL, or 2-column TSV. Got: {s[:120]!r}"
            )

        a, b = parts
        # detect where the label is
        if a.isdigit() and not b.isdigit():
            label, stmt = int(a), b
        elif b.isdigit() and not a.isdigit():
            stmt, label = a, int(b)
        else:
            raise ValueError(
                f"Ambiguous TSV at line {ln}. One column must be an int label. Got: {parts!r}"
            )

        rows.append({"statement": stmt, "label": label})

    return rows


def evaluate(dataset_path: str, *, base_config: dict, limit: Optional[int] = None) -> dict:
    if base_config is None:
        raise ValueError("base_config must be provided.")

    data = load_dataset(dataset_path)
    if limit is not None:
        data = data[: int(limit)]

    y_true = []
    y_pred = []
    errors = []
    confusion = defaultdict(lambda: Counter())

    for i, item in enumerate(data, start=1):
        text = item["statement"]
        true_label = int(item["label"])

        cfg = copy.deepcopy(base_config)
        set_mock_statement(cfg, text=text, stmt_id=i)

        orchestrator = PipelineOrchestrator(cfg)
        state = PipelineState()
        final_state = orchestrator.run(state)

        if not getattr(final_state, "statements", None):
            verdict = None
            pred_label = None
        else:
            stmt = final_state.statements[0]
            verdict = normalize_verdict(getattr(stmt, "verdict", None))
            pred_label = VERDICT_TO_LABEL.get(verdict)

        if pred_label is None:
            errors.append({
                "index": i,
                "statement": text,
                "true_label": true_label,
                "verdict": verdict,
                "note": "Unexpected or missing verdict; not counted in accuracy.",
            })
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)
        confusion[true_label][pred_label] += 1
        # random sleep between requests to avoid rate limits
        time.sleep(random.uniform(2, 5))

        if i % 50 == 0:
            print(f"Processed {i}/{len(data)}")

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = correct / total if total else 0.0

    per_class = {}
    by_class = defaultdict(list)
    for t, p in zip(y_true, y_pred):
        by_class[t].append(p)

    for cls, preds in by_class.items():
        cls_total = len(preds)
        cls_correct = sum(1 for p in preds if p == cls)
        per_class[cls] = cls_correct / cls_total if cls_total else 0.0

    return {
        "n_total_in_file": len(data),
        "n_scored": total,
        "n_skipped": len(errors),
        "accuracy": accuracy,
        "per_class_recall": per_class,
        "confusion_matrix": {t: dict(c) for t, c in confusion.items()},
        "skipped_examples": errors[:10],
    }


def main():
    dataset_path = "evaluation/data_set/claims_statements_train.txt"

    configs = [
        config_module.RAW_PIPELINE_CONFIG,
        config_module.PUBMED_PIPELINE_CONFIG,
        config_module.PUBMED_PIPELINE_CONFIG_NO_WEIGHTS,
    ]

    all_results = {}

    for base in configs:
        name = base.get("name", "Unnamed_Pipeline")
        print(f"\n=== EVALUATION RESULTS FOR PIPELINE: {name} ===")
        res = evaluate(dataset_path, base_config=base)
        all_results[name] = res

        print(f"Scored items: {res['n_scored']} / {res['n_total_in_file']}")
        print(f"Skipped items: {res['n_skipped']}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print(f"Per-class recall: {res['per_class_recall']}")
        print("Confusion matrix (rows=true, cols=pred):")
        for t, row in sorted(res["confusion_matrix"].items()):
            print(f"  true={t}: {row}")

    with open("truthness_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nSaved to truthness_eval_results.json")


if __name__ == "__main__":
    main()
