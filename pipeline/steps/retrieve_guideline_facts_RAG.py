#!/usr/bin/env python3
"""
retrieve_guideline_facts.py

Given a statement (claim), retrieve relevant guideline content from the SQLite vector DB:

- Loads embeddings and texts for all chunks
- Encodes the statement with the same embedding model (must match DB)
- Computes cosine similarity (dot product because vectors are normalized)
- Returns top-k chunks with metadata (doc path + pages)
- Optional: extracts "facts" as the most relevant sentences from the retrieved chunks
  (deterministic: sentence-level embedding similarity, no LLM required)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    source_path: str
    pages: List[int]
    text: str


@dataclass
class ExtractedFact:
    fact: str
    score: float
    chunk_id: str
    source_path: str
    pages: List[int]


def _parse_pages(pages_json: str) -> List[int]:
    try:
        v = json.loads(pages_json)
        if isinstance(v, list):
            return [int(x) for x in v]
    except Exception:
        pass
    return []


def _load_db_metadata(con: sqlite3.Connection) -> Tuple[str, int]:
    row = con.execute("SELECT embed_model, dim FROM documents LIMIT 1;").fetchone()
    if not row:
        raise RuntimeError("DB has no documents. Run build_guideline_vdb.py first.")
    return str(row[0]), int(row[1])


def _load_all_chunks(con: sqlite3.Connection, dim: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Returns:
      chunk_rows: list of dicts with chunk_id, source_path, pages, text
      E: (N, dim) normalized embeddings float32
    """
    rows = con.execute(
        """
        SELECT
            c.chunk_id,
            d.source_path,
            c.pages_json,
            c.text,
            c.embedding
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id;
        """
    ).fetchall()

    chunk_rows: List[Dict[str, Any]] = []
    E = np.zeros((len(rows), dim), dtype=np.float32)

    for i, (chunk_id, source_path, pages_json, text, emb_blob) in enumerate(rows):
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        if emb.shape[0] != dim:
            raise RuntimeError(f"Embedding dim mismatch for {chunk_id}: got {emb.shape[0]} expected {dim}")
        E[i] = emb
        chunk_rows.append(
            {
                "chunk_id": str(chunk_id),
                "source_path": str(source_path),
                "pages": _parse_pages(str(pages_json)),
                "text": str(text),
            }
        )

    return chunk_rows, E


def retrieve_chunks(
    db_path: Path,
    statement: str,
    top_k: int = 5,
    min_score: float = 0.25,
) -> Tuple[str, int, List[RetrievedChunk]]:
    con = sqlite3.connect(str(db_path))
    try:
        embed_model, dim = _load_db_metadata(con)
        model = SentenceTransformer(embed_model)

        chunk_rows, E = _load_all_chunks(con, dim=dim)

        q = model.encode([statement], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)[0]  # (dim,)

        # Because E and q are normalized, cosine similarity == dot product
        scores = E @ q  # (N,)

        # top-k selection
        if top_k <= 0:
            top_k = 5
        k = min(top_k, scores.shape[0])
        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        out: List[RetrievedChunk] = []
        for i in idxs.tolist():
            s = float(scores[i])
            if s < min_score:
                continue
            r = chunk_rows[i]
            out.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    score=s,
                    source_path=r["source_path"],
                    pages=r["pages"],
                    text=r["text"],
                )
            )

        return embed_model, dim, out
    finally:
        con.close()


def _split_sentences(text: str) -> List[str]:
    # Simple, language-agnostic-ish heuristic; deterministic and dependency-free.
    # You can replace with spaCy later if you want better sentence boundaries.
    text = " ".join(text.split())
    if not text:
        return []
    # Split on common end punctuation.
    sents: List[str] = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            sent = "".join(buf).strip()
            buf = []
            if sent:
                sents.append(sent)
    tail = "".join(buf).strip()
    if tail:
        sents.append(tail)
    return sents


def extract_facts_from_chunks(
    embed_model: str,
    statement: str,
    chunks: List[RetrievedChunk],
    max_facts_total: int = 10,
    max_facts_per_chunk: int = 2,
    min_fact_chars: int = 40,
    max_fact_chars: int = 320,
) -> List[ExtractedFact]:
    model = SentenceTransformer(embed_model)
    q = model.encode([statement], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]

    facts: List[ExtractedFact] = []
    for ch in chunks:
        sents = _split_sentences(ch.text)
        sents = [
            s for s in sents
            if min_fact_chars <= len(s) <= max_fact_chars
        ]
        if not sents:
            continue

        S = model.encode(sents, normalize_embeddings=True)
        S = np.asarray(S, dtype=np.float32)

        scores = S @ q
        k = min(max_facts_per_chunk, len(sents))
        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        for i in idxs.tolist():
            facts.append(
                ExtractedFact(
                    fact=sents[i],
                    score=float(scores[i]),
                    chunk_id=ch.chunk_id,
                    source_path=ch.source_path,
                    pages=ch.pages,
                )
            )

    facts.sort(key=lambda x: x.score, reverse=True)
    return facts[:max_facts_total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", type=str, default="guidelines_vdb.sqlite")
    ap.add_argument("--statement", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.25)
    ap.add_argument("--extract_facts", action="store_true", help="Extract sentence-level facts from retrieved chunks.")
    ap.add_argument("--max_facts_total", type=int, default=10)
    ap.add_argument("--max_facts_per_chunk", type=int, default=2)
    args = ap.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()

    embed_model, dim, chunks = retrieve_chunks(
        db_path=db_path,
        statement=args.statement,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    result: Dict[str, Any] = {
        "statement": args.statement,
        "embed_model": embed_model,
        "embedding_dim": dim,
        "retrieved_chunks": [asdict(c) for c in chunks],
    }

    if args.extract_facts:
        facts = extract_facts_from_chunks(
            embed_model=embed_model,
            statement=args.statement,
            chunks=chunks,
            max_facts_total=args.max_facts_total,
            max_facts_per_chunk=args.max_facts_per_chunk,
        )
        result["facts"] = [asdict(f) for f in facts]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
