#!/usr/bin/env python3
"""Evaluate guideline retrieval from JSONL relevance judgments."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .query_guideline_vdb import DEFAULT_EMBED_MODEL, _load_sentence_transformer, retrieve
except ImportError:
    from query_guideline_vdb import DEFAULT_EMBED_MODEL, _load_sentence_transformer, retrieve

def load_cases(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not case.get("claim"):
                raise ValueError(f"Line {line_number} has no non-empty 'claim'")
            if not case.get("relevant_chunk_ids") and not case.get("relevant_pages"):
                raise ValueError(f"Line {line_number} needs relevant_chunk_ids or relevant_pages")
            yield case

def is_relevant(result: Dict[str, Any], case: Dict[str, Any]) -> bool:
    chunk_ids = {str(value) for value in case.get("relevant_chunk_ids", [])}
    pages = {int(value) for value in case.get("relevant_pages", [])}
    chunk_match = result["chunk_id"] in chunk_ids
    page_match = any(page in pages for page in range(result["page_start"], result["page_end"] + 1))
    return chunk_match or page_match

def evaluate_cases(con: sqlite3.Connection, cases: List[Dict[str, Any]], embed_model: str, model, dense_k: int = 30, bm25_k: int = 30) -> Dict[str, float]:
    recall5 = recall10 = reciprocal_rank = 0.0
    for case in cases:
        results = retrieve(con, case["claim"], embed_model, model, dense_k, bm25_k, 10)
        ranks = [index for index, result in enumerate(results, 1) if is_relevant(result, case)]
        recall5 += float(any(rank <= 5 for rank in ranks))
        recall10 += float(any(rank <= 10 for rank in ranks))
        reciprocal_rank += 1.0 / ranks[0] if ranks and ranks[0] <= 10 else 0.0
    count = len(cases)
    return {"queries": float(count), "recall_at_5": recall5 / count if count else 0.0, "recall_at_10": recall10 / count if count else 0.0, "mrr_at_10": reciprocal_rank / count if count else 0.0}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db_path", default="guidelines_vdb.sqlite")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--dense_k", type=int, default=30)
    parser.add_argument("--bm25_k", type=int, default=30)
    args = parser.parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    eval_path = Path(args.eval_jsonl).expanduser().resolve()
    if not db_path.is_file():
        parser.error(f"Database not found: {db_path}")
    if not eval_path.is_file():
        parser.error(f"Evaluation JSONL not found: {eval_path}")
    try:
        cases = list(load_cases(eval_path))
        model = _load_sentence_transformer(args.embed_model)
        with sqlite3.connect(str(db_path)) as con:
            metrics = evaluate_cases(con, cases, args.embed_model, model, args.dense_k, args.bm25_k)
        print(f"Queries: {int(metrics['queries'])}")
        print(f"Recall@5: {metrics['recall_at_5']:.4f}")
        print(f"Recall@10: {metrics['recall_at_10']:.4f}")
        print(f"MRR@10: {metrics['mrr_at_10']:.4f}")
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

if __name__ == "__main__":
    main()
