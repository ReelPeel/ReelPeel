#!/usr/bin/env python3
"""
build_guideline_vdb.py

Ingest guideline PDFs into a lightweight SQLite "vector DB" for RAG retrieval.

- Reads PDFs from a directory (recursively)
- Extracts text (per page)
- Chunks text (word-based, with overlap)
- Embeds chunks using sentence-transformers
- Stores:
    documents(doc_id, source_path, file_hash, added_at, embed_model, dim)
    chunks(chunk_id, doc_id, pages_json, text, embedding_blob)

Notes:
- This uses brute-force cosine similarity at query time (fast enough for guideline corpora).
- For scanned PDFs with no embedded text, PyPDF extraction may return empty text.
"""

from __future__ import annotations
from transformers import AutoTokenizer
import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# Dependencies:
#   pip install pypdf sentence-transformers numpy
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "NeuML/pubmedbert-base-embeddings"


@dataclass(frozen=True)
class PageText:
    page_index_0: int
    text: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def chunk_by_tokens(
    pages: List[PageText],
    tokenizer,
    max_tokens: int=512,
    overlap_tokens: int=64,
) -> List[Tuple[List[int], str]]:
    
    assert max_tokens > 0
    assert 0 <= overlap_tokens < max_tokens

    token_ids: List[int] = []
    token_pages: List[int] = []

    for pt in pages:
        if not pt.text:
            continue
        page_no = pt.page_index_0 + 1
        ids = tokenizer.encode(pt.text, add_special_tokens=False)
        if not ids:
            continue
        token_ids.extend(ids)
        token_pages.extend([page_no] * len(ids))

    if not token_ids:
        return []

    step = max_tokens - overlap_tokens
    chunks: List[Tuple[List[int], str]] = []

    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        window_ids = token_ids[start:end]
        pages_in = sorted(set(token_pages[start:end]))

        # Decode back to text. This is deterministic and keeps the token budget guarantee.
        text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        if text:
            chunks.append((pages_in, text))

        if end == len(token_ids):
            break
        start += step

    return chunks


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                added_at TEXT NOT NULL,
                embed_model TEXT NOT NULL,
                dim INTEGER NOT NULL
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                pages_json TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
        con.commit()
    finally:
        con.close()


def pdf_to_pages(pdf_path: Path) -> List[PageText]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise PdfReadError(f"Failed to open PDF: {e}")

    # Try to decrypt with empty password (works for some “restricted but no user password” PDFs)
    try:
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # may return 0 if it didn't work; still worth trying
            except Exception:
                pass
    except DependencyError:
        # Missing AES deps
        raise

    pages: List[PageText] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = "\n".join(line.rstrip() for line in txt.splitlines()).strip()
        pages.append(PageText(page_index_0=i, text=txt))
    return pages



def iter_pdf_paths(pdf_dir: Path) -> Iterable[Path]:
    for p in pdf_dir.rglob("*.pdf"):
        if p.is_file():
            yield p




def doc_id_for(pdf_path: Path, file_hash: str) -> str:
    # Stable ID across runs as long as file content is stable
    key = f"{pdf_path.as_posix()}::{file_hash}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def upsert_document_and_chunks(
    con: sqlite3.Connection,
    pdf_path: Path,
    file_hash: str,
    embed_model: str,
    dim: int,
    chunks: List[Tuple[List[int], str]],
    embeddings: np.ndarray,
    force_reindex: bool,
) -> None:
    doc_id = doc_id_for(pdf_path, file_hash)

    # Check if already indexed
    row = con.execute(
        "SELECT file_hash, embed_model, dim FROM documents WHERE doc_id = ?;",
        (doc_id,),
    ).fetchone()

    if row and not force_reindex:
        # Already indexed with exact same doc_id (path+hash). Skip.
        return

    # If force or different settings, delete previous rows for this doc_id
    con.execute("DELETE FROM chunks WHERE doc_id = ?;", (doc_id,))
    con.execute("DELETE FROM documents WHERE doc_id = ?;", (doc_id,))

    con.execute(
        """
        INSERT INTO documents(doc_id, source_path, file_hash, added_at, embed_model, dim)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            doc_id,
            str(pdf_path),
            file_hash,
            dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            embed_model,
            int(dim),
        ),
    )

    # Insert chunks
    assert len(chunks) == embeddings.shape[0]
    for idx, ((pages_in, text), emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}:{idx:06d}"
        pages_json = json.dumps(pages_in, ensure_ascii=False)
        emb_blob = np.asarray(emb, dtype=np.float32).tobytes(order="C")

        con.execute(
            """
            INSERT INTO chunks(chunk_id, doc_id, pages_json, text, embedding)
            VALUES (?, ?, ?, ?, ?);
            """,
            (chunk_id, doc_id, pages_json, text, emb_blob),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True, help="Directory with guideline PDFs (recursive).")
    ap.add_argument("--db_path", type=str, default="guidelines_vdb.sqlite", help="SQLite DB path.")
    ap.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model name.")
    ap.add_argument("--chunk_words", type=int, default=240, help="Words per chunk.")
    ap.add_argument("--overlap_words", type=int, default=60, help="Overlap words between chunks.")
    ap.add_argument("--batch_size", type=int, default=32, help="Embedding batch size.")
    ap.add_argument("--force_reindex", action="store_true", help="Reindex even if doc_id already exists.")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    init_db(db_path)

    model = SentenceTransformer(args.embed_model)
    # We store normalized embeddings so cosine similarity == dot product
    # sentence-transformers supports normalize_embeddings=True
    dim = int(model.get_sentence_embedding_dimension())

    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        pdfs = list(iter_pdf_paths(pdf_dir))
        if not pdfs:
            print(f"No PDFs found under: {pdf_dir}")
            return

        for pdf_path in pdfs:
            file_hash = sha256_file(pdf_path)
            pages = pdf_to_pages(pdf_path)

            model = SentenceTransformer(args.embed_model)
            tokenizer = AutoTokenizer.from_pretrained(args.embed_model, use_fast=True)
            chunks = chunk_by_tokens(
                pages=pages,
                tokenizer=tokenizer,
                max_tokens=512,
                overlap_tokens=64,
            )

            if not chunks:
                print(f"[WARN] No extractable text/chunks for PDF: {pdf_path}")
                continue

            texts = [t for (_pages, t) in chunks]
            embeddings = model.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)

            upsert_document_and_chunks(
                con=con,
                pdf_path=pdf_path,
                file_hash=file_hash,
                embed_model=args.embed_model,
                dim=dim,
                chunks=chunks,
                embeddings=embeddings,
                force_reindex=args.force_reindex,
            )
            con.commit()
            print(f"Indexed: {pdf_path} | chunks={len(chunks)}")

        print(f"Done. DB: {db_path}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
