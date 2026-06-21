#!/usr/bin/env python3
"""Hybrid dense/FTS5 retrieval for a guideline SQLite vector database."""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_EMBED_MODEL = "NeuML/pubmedbert-base-embeddings"

def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use dense retrieval.") from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {exc}") from exc

def _fts_query(query: str) -> str:
    terms = list(dict.fromkeys(re.findall(r"[^\W_]{2,}", query, flags=re.UNICODE)))
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)

def _metadata(con: sqlite3.Connection, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = con.execute(f"""
        SELECT c.chunk_id, c.parent_chunk_id, c.section_path, c.page_start, c.page_end,
               c.chunk_type, c.text, d.source_path, d.title
        FROM chunks c JOIN documents d ON d.doc_id=c.doc_id
        WHERE c.chunk_id IN ({placeholders})
    """, chunk_ids).fetchall()
    return {str(row[0]): {"chunk_id": str(row[0]), "parent_chunk_id": row[1], "section_path": row[2], "page_start": int(row[3]), "page_end": int(row[4]), "chunk_type": str(row[5]), "text": str(row[6]), "source_path": str(row[7]), "title": row[8]} for row in rows}

def dense_search(con: sqlite3.Connection, query: str, embed_model: str, model, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    rows = con.execute("SELECT e.chunk_id, e.dim, e.embedding FROM embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id WHERE e.embed_model=? AND c.chunk_type IN ('evidence', 'recommendation')", (embed_model,)).fetchall()
    if not rows:
        available = [str(row[0]) for row in con.execute("SELECT DISTINCT embed_model FROM embeddings ORDER BY embed_model")]
        suffix = f" Available models: {', '.join(available)}" if available else " The database has no embeddings."
        raise RuntimeError(f"No embeddings found for model '{embed_model}'.{suffix}")
    query_vector = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
    scored = []
    for chunk_id, dim, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.shape[0] != int(dim) or vector.shape[0] != query_vector.shape[0]:
            continue
        scored.append((str(chunk_id), float(vector @ query_vector)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [{"chunk_id": chunk_id, "dense_score": score, "dense_rank": rank} for rank, (chunk_id, score) in enumerate(scored[:limit], 1)]

def bm25_search(con: sqlite3.Connection, query: str, limit: int) -> List[Dict[str, Any]]:
    expression = _fts_query(query)
    if limit <= 0 or not expression:
        return []
    try:
        rows = con.execute("SELECT f.chunk_id, bm25(chunks_fts) AS score FROM chunks_fts f JOIN chunks c ON c.chunk_id=f.chunk_id WHERE chunks_fts MATCH ? AND c.chunk_type IN ('evidence', 'recommendation') ORDER BY score LIMIT ?", (expression, limit)).fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(f"FTS5 query failed: {exc}") from exc
    return [{"chunk_id": str(chunk_id), "bm25_score": float(score), "bm25_rank": rank} for rank, (chunk_id, score) in enumerate(rows, 1)]

def retrieve(con: sqlite3.Connection, query: str, embed_model: str = DEFAULT_EMBED_MODEL, model=None, dense_k: int = 30, bm25_k: int = 30, final_k: int = 10, rrf_k: int = 60, claim_mode: bool = False, reranker=None, rerank_top_n: int = 30) -> List[Dict[str, Any]]:
    model = model or _load_sentence_transformer(embed_model)
    dense = dense_search(con, query, embed_model, model, dense_k)
    lexical = bm25_search(con, query, bm25_k)
    fused: Dict[str, Dict[str, Any]] = {}
    for result in dense:
        fused.setdefault(result["chunk_id"], {"chunk_id": result["chunk_id"], "fused_score": 0.0}).update(result)
        fused[result["chunk_id"]]["fused_score"] += 1.0 / (rrf_k + result["dense_rank"])
    for result in lexical:
        fused.setdefault(result["chunk_id"], {"chunk_id": result["chunk_id"], "fused_score": 0.0}).update(result)
        fused[result["chunk_id"]]["fused_score"] += 1.0 / (rrf_k + result["bm25_rank"])
    metadata = _metadata(con, list(fused))
    candidates = []
    for chunk_id, result in fused.items():
        if chunk_id not in metadata:
            continue
        result.update(metadata[chunk_id])
        if claim_mode and result["chunk_type"] == "recommendation":
            result["fused_score"] *= 1.05
        candidates.append(result)
    candidates.sort(key=lambda item: item["fused_score"], reverse=True)
    if reranker and candidates:
        rerank_candidates = candidates[:max(final_k, rerank_top_n)]
        scores = reranker.predict([(query, item["text"]) for item in rerank_candidates])
        for item, score in zip(rerank_candidates, scores):
            item["reranker_score"] = float(score)
        rerank_candidates.sort(key=lambda item: item["reranker_score"], reverse=True)
        candidates = rerank_candidates + candidates[len(rerank_candidates):]
    return candidates[:max(0, final_k)]

def add_parent_context(con: sqlite3.Connection, results: List[Dict[str, Any]]) -> None:
    parent_ids = list(dict.fromkeys(str(item["parent_chunk_id"]) for item in results if item.get("parent_chunk_id")))
    parents = _metadata(con, parent_ids)
    for item in results:
        parent = parents.get(str(item.get("parent_chunk_id")))
        item["parent_text"] = parent["text"] if parent else None

def relevance_terms(query: str, text: str) -> Dict[str, List[str]]:
    combined = f"{query} {text}".casefold()
    patterns = {
        "population_terms": r"\b(?:adult|adults|child|children|infant|infants|pregnan\w*|elderly|patient\w*|population)\b",
        "intervention_terms": r"\b(?:therapy|treatment|drug|dose|screening|surgery|intervention|first-line|second-line)\b",
        "negation_terms": r"\b(?:no|not|never|avoid|contraindicated|without|do not|should not)\b",
        "recommendation_terms": r"\b(?:recommend\w*|suggest\w*|should|offer\w*|first-line|second-line|consensus)\b",
    }
    return {name: sorted(set(re.findall(pattern, combined, flags=re.I))) for name, pattern in patterns.items()}

def _print_results(query: str, results: List[Dict[str, Any]], claim_mode: bool) -> None:
    print(f"Query: {query}\nResults: {len(results)}")
    for index, item in enumerate(results, 1):
        pages = str(item["page_start"]) if item["page_start"] == item["page_end"] else f"{item['page_start']}-{item['page_end']}"
        print(f"\n[{index}] {item['chunk_id']} | fused={item['fused_score']:.6f}")
        print(f"Source: {item['source_path']} | title={item.get('title') or '-'} | pages={pages}")
        print(f"Section: {item.get('section_path') or '-'} | type={item['chunk_type']}")
        if "dense_score" in item:
            print(f"Dense: score={item['dense_score']:.4f} rank={item['dense_rank']}")
        if "bm25_rank" in item:
            print(f"BM25: rank={item['bm25_rank']} raw={item['bm25_score']:.4f}")
        if "reranker_score" in item:
            print(f"Reranker: {item['reranker_score']:.4f}")
        print(f"Text: {item['text'][:700]}")
        if claim_mode:
            fields = relevance_terms(query, item["text"])
            print("Claim relevance: " + "; ".join(f"{key}={','.join(values) or '-'}" for key, values in fields.items()))
        if item.get("parent_text"):
            print(f"Parent context: {item['parent_text'][:1400]}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db_path", default="guidelines_vdb.sqlite")
    parser.add_argument("--query", required=True)
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--dense_k", type=int, default=30)
    parser.add_argument("--bm25_k", type=int, default=30)
    parser.add_argument("--final_k", type=int, default=10)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--show_parent", action="store_true")
    parser.add_argument("--claim_mode", action="store_true")
    parser.add_argument("--rerank_model")
    parser.add_argument("--rerank_top_n", type=int, default=30)
    args = parser.parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    if not db_path.is_file():
        parser.error(f"Database not found: {db_path}")
    try:
        model = _load_sentence_transformer(args.embed_model)
        reranker: Optional[Any] = None
        if args.rerank_model:
            try:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder(args.rerank_model)
            except Exception as exc:
                raise RuntimeError(f"Failed to load reranker '{args.rerank_model}': {exc}") from exc
        with sqlite3.connect(str(db_path)) as con:
            results = retrieve(con, args.query, args.embed_model, model, args.dense_k, args.bm25_k, args.final_k, args.rrf_k, args.claim_mode, reranker, args.rerank_top_n)
            if args.show_parent or args.claim_mode:
                add_parent_context(con, results)
        _print_results(args.query, results, args.claim_mode)
    except (RuntimeError, sqlite3.Error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
