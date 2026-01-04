#!/usr/bin/env python3
import json
import time
import copy
import os
import re
import random
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any

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


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_verdict(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    return v.strip().lower()


def sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "Unnamed_Pipeline"


def atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_json_if_exists(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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
      1) JSON list:        [{"id": 123, "statement": "...", "label": 0}, ...]   (id optional)
      2) JSONL:            one JSON object per line (id optional)
      3) TSV-like text:    "<statement>\\t<label>" or "<label>\\t<statement>" (id generated sequentially)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # Case 1: JSON list
    if raw[0] == "[":
        data = json.loads(raw)
        rows: List[Dict[str, Any]] = []
        for idx, x in enumerate(data, start=1):
            stmt = x.get("statement", x.get("text"))
            if stmt is None or "label" not in x:
                raise ValueError("JSON list item missing 'statement'/'text' or 'label'.")
            stmt_id = int(x["id"]) if "id" in x and x["id"] is not None else idx
            rows.append({"id": stmt_id, "statement": stmt, "label": int(x["label"])})
        return rows

    rows: List[Dict[str, Any]] = []
    next_id = 1

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
            stmt_id = int(obj["id"]) if "id" in obj and obj["id"] is not None else next_id
            rows.append({"id": stmt_id, "statement": stmt, "label": int(obj["label"])})
            next_id += 1
            continue

        # Case 3: TSV-like
        parts = [p.strip() for p in s.split("\t")]
        if len(parts) != 2:
            raise ValueError(
                f"Unrecognized dataset line format at line {ln}. "
                f"Expected JSON, JSONL, or 2-column TSV. Got: {s[:120]!r}"
            )

        a, b = parts
        if a.isdigit() and not b.isdigit():
            label, stmt = int(a), b
        elif b.isdigit() and not a.isdigit():
            stmt, label = a, int(b)
        else:
            raise ValueError(
                f"Ambiguous TSV at line {ln}. One column must be an int label. Got: {parts!r}"
            )

        rows.append({"id": next_id, "statement": stmt, "label": label})
        next_id += 1

    return rows


def load_or_init_output(output_path: str, *, dataset_path: str, config_name: str) -> dict:
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("meta", {})
        payload.setdefault("items", {})
        return payload

    return {
        "meta": {
            "dataset_path": dataset_path,
            "config_name": config_name,
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "completed_at": None,
            "completed": False,
            # Accumulated across sessions; includes your sleep for rate limiting
            "total_wall_seconds": 0.0,
            # Accumulated pipeline runtime only (orchestrator.run)
            "total_pipeline_seconds": 0.0,
            "n_total_in_file": None,
            "n_done": 0,
        },
        # Keyed by string id for JSON compatibility and easy resume
        "items": {}
    }


def summarize_per_config_output(per_config_payload: dict) -> dict:
    """
    Build a truthness_eval_results.json-style summary from a per-config output payload.
    Rules:
      - predicted_label == None -> skipped (not scored)
      - predicted_label in {0,1,2} -> scored
    """
    items = per_config_payload.get("items", {}) or {}
    meta = per_config_payload.get("meta", {}) or {}

    y_true: List[int] = []
    y_pred: List[int] = []
    confusion = defaultdict(lambda: Counter())
    skipped = 0

    for rec in items.values():
        t = rec.get("true_label")
        p = rec.get("predicted_label")
        if p is None:
            skipped += 1
            continue
        y_true.append(int(t))
        y_pred.append(int(p))
        confusion[int(t)][int(p)] += 1

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
        "n_total_in_file": meta.get("n_total_in_file"),
        "n_scored": total,
        "n_skipped": skipped,
        "accuracy": accuracy,
        "per_class_recall": per_class,
        "confusion_matrix": {t: dict(c) for t, c in confusion.items()},
        "total_wall_seconds": meta.get("total_wall_seconds"),
        "total_pipeline_seconds": meta.get("total_pipeline_seconds"),
        "completed": meta.get("completed"),
        "updated_at": meta.get("updated_at"),
    }


def update_truthness_results(
    summary_path: str,
    config_name: str,
    per_config_payload: dict,
    per_config_output_path: str
) -> None:
    """
    Atomically update truthness_eval_results.json with latest summary for this config.
    """
    summary = load_json_if_exists(summary_path)

    entry = summarize_per_config_output(per_config_payload)
    entry["per_config_output_file"] = per_config_output_path
    summary[config_name] = entry

    atomic_write_json(summary_path, summary)


def evaluate_to_json(
    dataset_path: str,
    *,
    base_config: dict,
    output_path: str,
    limit: Optional[int] = None,
    seed: int = 42,
    sleep_min: float = 2.0,
    sleep_max: float = 5.0,
    progress_every: int = 50,
) -> dict:
    if base_config is None:
        raise ValueError("base_config must be provided.")

    rng = random.Random(seed)

    data = load_dataset(dataset_path)
    if limit is not None:
        data = data[: int(limit)]

    config_name = base_config.get("name", "Unnamed_Pipeline")
    out = load_or_init_output(output_path, dataset_path=dataset_path, config_name=config_name)
    out["meta"]["n_total_in_file"] = len(data)

    # Resume support: skip IDs already present in output.
    done_ids = set()
    for k in (out.get("items", {}) or {}).keys():
        try:
            done_ids.add(int(k))
        except Exception:
            pass

    n_done_before = len(done_ids)

    for item in data:
        stmt_id = int(item["id"])
        if stmt_id in done_ids:
            continue

        text = item["statement"]
        true_label = int(item["label"])

        # Wall time includes everything in this iteration (including sleep)
        iter_wall_start = time.perf_counter()

        cfg = copy.deepcopy(base_config)
        set_mock_statement(cfg, text=text, stmt_id=stmt_id)

        verdict = None
        pred_label = None
        error_note = None

        # Pipeline runtime: orchestrator.run only
        pipeline_start = time.perf_counter()
        try:
            orchestrator = PipelineOrchestrator(cfg)
            state = PipelineState()
            final_state = orchestrator.run(state)

            if not getattr(final_state, "statements", None):
                error_note = "No statements returned by pipeline."
            else:
                stmt = final_state.statements[0]
                verdict = normalize_verdict(getattr(stmt, "verdict", None))
                pred_label = VERDICT_TO_LABEL.get(verdict)
                if pred_label is None:
                    error_note = f"Unexpected or missing verdict: {verdict!r}"
        except Exception as e:
            error_note = f"Exception during pipeline run: {type(e).__name__}: {e}"
        pipeline_end = time.perf_counter()

        # Rate-limit sleep
        sleep_seconds = float(rng.uniform(sleep_min, sleep_max)) if sleep_max > 0 else 0.0
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        iter_wall_end = time.perf_counter()

        pipeline_seconds = float(pipeline_end - pipeline_start)
        wall_seconds = float(iter_wall_end - iter_wall_start)

        # Persist one record keyed by statement id (resumable)
        out.setdefault("items", {})
        out["items"][str(stmt_id)] = {
            "id": stmt_id,
            "true_label": true_label,
            "predicted_label": pred_label,  # null if failed/unknown verdict
            # Per-statement runtime reporting
            "pipeline_seconds": pipeline_seconds,
            "sleep_seconds": sleep_seconds,
            "wall_seconds": wall_seconds,
            # Optional debug fields
            "verdict": verdict,
            "error": error_note,
        }

        # Update meta (accumulated, resumable)
        out["meta"]["total_pipeline_seconds"] = float(out["meta"].get("total_pipeline_seconds", 0.0)) + pipeline_seconds
        out["meta"]["total_wall_seconds"] = float(out["meta"].get("total_wall_seconds", 0.0)) + wall_seconds
        out["meta"]["updated_at"] = iso_now()
        out["meta"]["n_done"] = len(out["items"])

        # Write after every statement so resume never recomputes completed ids
        atomic_write_json(output_path, out)

        if progress_every and out["meta"]["n_done"] % progress_every == 0:
            print(f"[{config_name}] Done {out['meta']['n_done']}/{len(data)} (output: {output_path})")

    out["meta"]["completed"] = True
    out["meta"]["completed_at"] = iso_now()
    out["meta"]["updated_at"] = iso_now()
    atomic_write_json(output_path, out)

    n_added = len(out["items"]) - n_done_before
    print(f"[{config_name}] Finished. Added {n_added} new items. Total done: {len(out['items'])}.")
    print(f"[{config_name}] Total pipeline seconds (accumulated): {out['meta']['total_pipeline_seconds']:.2f}")
    print(f"[{config_name}] Total wall seconds (accumulated): {out['meta']['total_wall_seconds']:.2f}")
    return out


def main():
    dataset_path = "evaluation/data_set/claims_statements_train.txt"

    configs = [
        config_module.RAW_PIPELINE_CONFIG,
        config_module.PUBMED_PIPELINE_CONFIG,
        config_module.PUBMED_PIPELINE_CONFIG_NO_WEIGHTS,
    ]

    # All outputs go here as requested
    out_dir = "evaluation/eval_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Also place the summary here
    truthness_summary_path = os.path.join(out_dir, "truthness_eval_results.json")

    for base in configs:
        name = base.get("name", "Unnamed_Pipeline")
        safe = sanitize_filename(name)
        per_config_output_path = os.path.join(out_dir, f"{safe}.json")

        print(f"\n=== RUNNING PIPELINE: {name} ===")

        per_config_payload = evaluate_to_json(
            dataset_path,
            base_config=base,
            output_path=per_config_output_path,
            limit=None,
            seed=42,
            sleep_min=2.0,
            sleep_max=5.0,
            progress_every=50,
        )

        # Update the global summary after each config run (including resumed runs)
        update_truthness_results(
            summary_path=truthness_summary_path,
            config_name=name,
            per_config_payload=per_config_payload,
            per_config_output_path=per_config_output_path,
        )

        # Print summary to console
        s = summarize_per_config_output(per_config_payload)
        print(f"\n=== EVALUATION RESULTS FOR PIPELINE: {name} ===")
        print(f"Scored items: {s['n_scored']} / {s['n_total_in_file']}")
        print(f"Skipped items: {s['n_skipped']}")
        print(f"Accuracy: {s['accuracy']:.4f}")
        print(f"Per-class recall: {s['per_class_recall']}")
        print("Confusion matrix (rows=true, cols=pred):")
        for t, row in sorted(s["confusion_matrix"].items()):
            print(f"  true={t}: {row}")
        print(f"Updated summary: {truthness_summary_path}")

    print(f"\nDone. Final summary saved in {truthness_summary_path}")


if __name__ == "__main__":
    main()
