#!/usr/bin/env python3
"""
retrieve_guideline_facts.py

Given a statement (claim), retrieve relevant guideline content from the SQLite vector DB:

- Loads embeddings and abstracts for all chunks
- Encodes the statement with the same embedding model (must match DB)
- Computes cosine similarity (dot product because vectors are normalized)
- Returns top-k chunks with metadata (doc path + pages)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..core.base import PipelineStep
from ..core.models import PipelineState, RAGEvidence


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
      chunk_rows: list of dicts with chunk_id, source_path, pages, abstract
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
                "abstract": str(text),
            }
        )

    return chunk_rows, E


def load_guideline_index(db_path: Path) -> Tuple[str, int, List[Dict[str, Any]], np.ndarray]:
    con = sqlite3.connect(str(db_path))
    try:
        embed_model, dim = _load_db_metadata(con)
        chunk_rows, embeddings = _load_all_chunks(con, dim=dim)
        return embed_model, dim, chunk_rows, embeddings
    finally:
        con.close()


def retrieve_chunks_for_statement(
    statement: str,
    model: SentenceTransformer,
    chunk_rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.25,
) -> List[RAGEvidence]:
    if not chunk_rows:
        return []

    q = model.encode([statement], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]  # (dim,)

    # Because embeddings and q are normalized, cosine similarity == dot product
    scores = embeddings @ q  # (N,)

    # top-k selection
    if top_k <= 0:
        top_k = 5
    k = min(top_k, scores.shape[0])
    idxs = np.argpartition(-scores, kth=k - 1)[:k]
    idxs = idxs[np.argsort(-scores[idxs])]

    out: List[RAGEvidence] = []
    for i in idxs.tolist():
        s = float(scores[i])
        if s < min_score:
            continue
        r = chunk_rows[i]
        out.append(
            RAGEvidence(
                chunk_id=r["chunk_id"],
                score=s,
                source_path=r["source_path"],
                pages=r["pages"],
                abstract=r["abstract"],
                weight=1.0,
                relevance=s,
                relevance_abstract=s,
            )
        )

    return out


def retrieve_chunks(
    db_path: Path,
    statement: str,
    top_k: int = 5,
    min_score: float = 0.25,
) -> Tuple[str, int, List[RAGEvidence]]:
    embed_model, dim, chunk_rows, embeddings = load_guideline_index(db_path)
    model = SentenceTransformer(embed_model)
    chunks = retrieve_chunks_for_statement(
        statement=statement,
        model=model,
        chunk_rows=chunk_rows,
        embeddings=embeddings,
        top_k=top_k,
        min_score=min_score,
    )
    return embed_model, dim, chunks





class RetrieveGuidelineFactsStep(PipelineStep):
    """
    Config keys:
      - db_path: str (default: "guidelines_vdb.sqlite")
      - top_k: int (default: 5)
      - min_score: float (default: 0.25)
    """

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.statements:
            print(f"[{self.__class__.__name__}] No statements available; skipping guideline retrieval.")
            return state

        db_path = Path(self.config.get("db_path", "guidelines_vdb.sqlite")).expanduser().resolve()
        if not db_path.exists():
            raise FileNotFoundError(f"Guideline DB not found: {db_path}")

        top_k = int(self.config.get("top_k", 5))
        min_score = float(self.config.get("min_score", 0.25))

        embed_model, dim, chunk_rows, embeddings = load_guideline_index(db_path)
        model = SentenceTransformer(embed_model)

        print(
            f"[{self.__class__.__name__}] Loaded guideline index with "
            f"{len(chunk_rows)} chunks (dim={dim})."
        )

        for stmt in state.statements:
            statement_text = (stmt.text or "").strip()
            if not statement_text:
                continue

            chunks = retrieve_chunks_for_statement(
                statement=statement_text,
                model=model,
                chunk_rows=chunk_rows,
                embeddings=embeddings,
                top_k=top_k,
                min_score=min_score,
            )
            if stmt.evidence is None:
                stmt.evidence = []
            stmt.evidence.extend(chunks)

            if self.debug:
                print(f"   Statement {stmt.id}: +{len(chunks)} RAG chunks.")

        return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", type=str, default="guidelines_vdb.sqlite")
    ap.add_argument("--statement", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.25)
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
        "retrieved_chunks": [c.model_dump() for c in chunks],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
