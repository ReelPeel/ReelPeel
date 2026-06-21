#!/usr/bin/env python3
"""Build a lightweight, provenance-rich SQLite guideline vector database."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PdfReadError
except ImportError:
    PdfReader = None

    class PdfReadError(Exception):
        pass

    class DependencyError(Exception):
        pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

DEFAULT_EMBED_MODEL = "NeuML/pubmedbert-base-embeddings"
LOW_TEXT_THRESHOLD = 100

@dataclass(frozen=True)
class PageText:
    page_index_0: int
    text: str
    low_text_warning: int = 0

@dataclass(frozen=True)
class SectionedPageText:
    page_index_0: int
    text: str
    section_path: Optional[str]

@dataclass(frozen=True)
class TokenChunk:
    text: str
    page_start: int
    page_end: int
    token_count: int
    section_path: Optional[str]
    token_start: int
    token_end: int

@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    parent_chunk_id: Optional[str]
    section_path: Optional[str]
    page_start: int
    page_end: int
    chunk_type: str
    text: str
    token_count: int

@dataclass(frozen=True)
class RecommendationCandidate:
    recommendation_text: str
    strength: Optional[str]
    evidence_quality: Optional[str]
    polarity: str

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def doc_id_for(file_hash_or_path: object, legacy_file_hash: Optional[str] = None) -> str:
    file_hash = legacy_file_hash if legacy_file_hash is not None else str(file_hash_or_path)
    return hashlib.sha256(file_hash.encode("utf-8")).hexdigest()[:32]

def normalize_pdf_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

def _try_ocr_page(pdf_path: Path, page_index_0: int) -> Optional[str]:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        print("[WARN] OCR requested but pytesseract/pdf2image are unavailable; continuing without OCR.", file=sys.stderr)
        return None
    try:
        images = convert_from_path(str(pdf_path), first_page=page_index_0 + 1, last_page=page_index_0 + 1)
        return normalize_pdf_text(pytesseract.image_to_string(images[0])) if images else None
    except Exception as exc:
        print(f"[WARN] OCR failed for {pdf_path} page {page_index_0 + 1}: {exc}", file=sys.stderr)
        return None

def pdf_to_pages(pdf_path: Path, ocr_if_low_text: bool = False) -> List[PageText]:
    if PdfReader is None:
        raise RuntimeError("Missing dependency 'pypdf'. Install it with: pip install pypdf")
    try:
        reader = PdfReader(str(pdf_path))
        if getattr(reader, "is_encrypted", False):
            reader.decrypt("")
    except (PdfReadError, DependencyError):
        raise
    except Exception as exc:
        raise PdfReadError(f"Failed to open PDF {pdf_path}: {exc}") from exc
    pages = []
    for index, page in enumerate(reader.pages):
        try:
            text = normalize_pdf_text(page.extract_text() or "")
        except Exception as exc:
            print(f"[WARN] Text extraction failed for {pdf_path} page {index + 1}: {exc}")
            text = ""
        if len(text) < LOW_TEXT_THRESHOLD:
            print(f"[WARN] {pdf_path} page {index + 1} has only {len(text)} extracted characters; it may be scanned or extraction may have failed.")
            if ocr_if_low_text:
                ocr_text = _try_ocr_page(pdf_path, index)
                if ocr_text and len(ocr_text) > len(text):
                    text = ocr_text
        pages.append(PageText(index, text, int(len(text) < LOW_TEXT_THRESHOLD)))
    return pages

COMMON_HEADINGS = {"recommendations", "diagnosis", "treatment", "management", "background", "evidence", "contraindications", "follow-up", "follow up", "monitoring", "special populations", "pregnancy", "children", "adults"}

def _heading_parts(line: str) -> Optional[Tuple[Optional[str], str]]:
    candidate = re.sub(r"\s+", " ", line).strip()
    if not candidate or len(candidate) > 120:
        return None
    numbered = re.match(r"^(\d+(?:\.\d+)*)(?:[.)])?\s+(.+?)\s*$", candidate)
    if numbered and not re.search(r"[.!?;:]$", numbered.group(2)):
        return numbered.group(1), candidate
    lowered = candidate.casefold().rstrip(":")
    letters = [char for char in candidate if char.isalpha()]
    is_upper = bool(letters) and candidate.upper() == candidate and 2 <= len(candidate.split()) <= 12
    looks_title = 1 <= len(candidate.split()) <= 10 and not re.search(r"[.!?;,]$", candidate) and candidate[0].isupper()
    if lowered in COMMON_HEADINGS or is_upper or looks_title:
        return None, candidate.rstrip(":")
    return None

def section_pages(pages: Sequence[PageText]) -> List[SectionedPageText]:
    headings = {}
    current = None
    output = []
    for page in pages:
        for line in page.text.splitlines():
            heading = _heading_parts(line)
            if not heading:
                continue
            number, title = heading
            if number:
                level = number.count(".") + 1
                headings[level] = title
                headings = {key: value for key, value in headings.items() if key <= level}
                current = " > ".join(headings[key] for key in sorted(headings))
            else:
                headings = {1: title}
                current = title
        output.append(SectionedPageText(page.page_index_0, page.text, current))
    return output

def chunk_sectioned_pages(pages: Sequence[SectionedPageText], tokenizer, max_tokens: int, overlap_tokens: int) -> List[TokenChunk]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Require max_tokens > 0 and 0 <= overlap_tokens < max_tokens")
    token_ids, token_pages, token_sections = [], [], []
    for page in pages:
        ids = tokenizer.encode(page.text, add_special_tokens=False) if page.text else []
        token_ids.extend(ids)
        token_pages.extend([page.page_index_0 + 1] * len(ids))
        token_sections.extend([page.section_path] * len(ids))
    chunks = []
    step = max_tokens - overlap_tokens
    for start in range(0, len(token_ids), step):
        end = min(start + max_tokens, len(token_ids))
        if start >= end:
            break
        text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        if text:
            sections = [section for section in token_sections[start:end] if section]
            section = max(set(sections), key=sections.count) if sections else None
            chunks.append(TokenChunk(text, min(token_pages[start:end]), max(token_pages[start:end]), end - start, section, start, end))
        if end == len(token_ids):
            break
    return chunks

def chunk_by_tokens(pages: Sequence[PageText], tokenizer, max_tokens: int = 220, overlap_tokens: int = 40) -> List[Tuple[List[int], str]]:
    chunks = chunk_sectioned_pages(section_pages(pages), tokenizer, max_tokens, overlap_tokens)
    return [(list(range(chunk.page_start, chunk.page_end + 1)), chunk.text) for chunk in chunks]

def assign_parent_chunks(evidence_chunks: Sequence[TokenChunk], parent_chunks: Sequence[TokenChunk]) -> List[Optional[int]]:
    assignments = []
    for evidence in evidence_chunks:
        best_index, best_score = None, -1
        for index, parent in enumerate(parent_chunks):
            token_overlap = max(0, min(evidence.token_end, parent.token_end) - max(evidence.token_start, parent.token_start))
            page_overlap = max(0, min(evidence.page_end, parent.page_end) - max(evidence.page_start, parent.page_start) + 1)
            score = token_overlap * 1000 + page_overlap
            if score > best_score:
                best_index, best_score = index, score
        assignments.append(best_index if best_score > 0 else None)
    return assignments

RECOMMENDATION_TRIGGER = re.compile(r"\b(?:we (?:strongly |conditionally )?recommend|we suggest|should (?:not )?(?:be offered|offer)|do not offer|is (?:not )?recommended|is contraindicated|avoid|first-line|second-line|strong recommendation|conditional recommendation)\b", re.I)

def extract_recommendations_from_chunk(chunk_text: str) -> List[RecommendationCandidate]:
    text = re.sub(r"\s*\n\s*", " ", chunk_text).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    output = []
    for sentence in sentences:
        if not RECOMMENDATION_TRIGGER.search(sentence):
            continue
        lower = sentence.casefold()
        if re.search(r"\b(?:do not offer|should not (?:be offered|offer)|not recommended|contraindicated|avoid)\b", lower):
            polarity = "negative"
        elif re.search(r"\b(?:recommend|suggest|should (?:be offered|offer)|first-line|second-line)\b", lower):
            polarity = "positive"
        else:
            polarity = "unknown"
        strength = "strong" if "strongly" in lower else next((value for value in ("strong", "conditional", "weak", "consensus", "good practice") if re.search(r"\b" + re.escape(value) + r"\b", lower)), None)
        quality = next((value for value in ("very low", "moderate", "high", "low") if re.search(r"\b" + re.escape(value) + r"\b", lower)), None)
        output.append(RecommendationCandidate(sentence, strength, quality, polarity))
    return output

def iter_pdf_paths(pdf_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())

def _columns(con: sqlite3.Connection, table: str) -> set:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}

def _schema_is_current(con: sqlite3.Connection) -> bool:
    required = {"documents": {"doc_id", "title", "file_hash"}, "pages": {"page_id", "low_text_warning"}, "chunks": {"parent_chunk_id", "chunk_type", "token_count"}, "embeddings": {"embed_model", "embedding"}, "recommendations": {"recommendation_text"}}
    return all(columns.issubset(_columns(con, table)) for table, columns in required.items())

def _drop_schema(con: sqlite3.Connection) -> None:
    con.execute("DROP TABLE IF EXISTS chunks_fts")
    for table in ("recommendations", "embeddings", "chunks", "pages", "documents"):
        con.execute(f"DROP TABLE IF EXISTS {table}")

def init_db(db_path: Path, reset_db: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        existing = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'").fetchone()
        if reset_db:
            _drop_schema(con)
        elif existing and not _schema_is_current(con):
            raise RuntimeError("Legacy schema detected. Re-run with --force_reindex --reset_db to rebuild it.")
        con.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, file_hash TEXT NOT NULL, title TEXT, organization TEXT, publication_year INTEGER, version TEXT, guideline_url TEXT, language TEXT, added_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages (page_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, page_number INTEGER NOT NULL, text TEXT NOT NULL, char_count INTEGER NOT NULL, low_text_warning INTEGER NOT NULL DEFAULT 0, FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, parent_chunk_id TEXT, section_path TEXT, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, chunk_type TEXT NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE, FOREIGN KEY(parent_chunk_id) REFERENCES chunks(chunk_id) ON DELETE SET NULL);
        CREATE TABLE IF NOT EXISTS embeddings (chunk_id TEXT NOT NULL, embed_model TEXT NOT NULL, dim INTEGER NOT NULL, embedding BLOB NOT NULL, PRIMARY KEY(chunk_id, embed_model), FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS recommendations (rec_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, source_chunk_id TEXT, section_path TEXT, page_start INTEGER, page_end INTEGER, recommendation_text TEXT NOT NULL, strength TEXT, evidence_quality TEXT, polarity TEXT, FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE, FOREIGN KEY(source_chunk_id) REFERENCES chunks(chunk_id) ON DELETE SET NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, section_path, text, tokenize='unicode61');
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(embed_model);
        CREATE INDEX IF NOT EXISTS idx_recommendations_doc_id ON recommendations(doc_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
        """)

def document_has_embeddings(con: sqlite3.Connection, doc_id: str, embed_model: str) -> bool:
    row = con.execute("SELECT COUNT(*), COUNT(e.chunk_id) FROM chunks c LEFT JOIN embeddings e ON e.chunk_id=c.chunk_id AND e.embed_model=? WHERE c.doc_id=?", (embed_model, doc_id)).fetchone()
    return bool(row and row[0] > 0 and row[0] == row[1])

def build_document_records(doc_id: str, pages: Sequence[PageText], tokenizer, chunk_tokens: int, overlap_tokens: int, parent_chunk_tokens: int, parent_overlap_tokens: int) -> Tuple[List[ChunkRecord], List[Tuple[str, RecommendationCandidate]]]:
    sectioned = section_pages(pages)
    parents = chunk_sectioned_pages(sectioned, tokenizer, parent_chunk_tokens, parent_overlap_tokens)
    evidence = chunk_sectioned_pages(sectioned, tokenizer, chunk_tokens, overlap_tokens)
    assignments = assign_parent_chunks(evidence, parents)
    records = [ChunkRecord(f"{doc_id}:parent:{index:06d}", doc_id, None, chunk.section_path, chunk.page_start, chunk.page_end, "parent", chunk.text, chunk.token_count) for index, chunk in enumerate(parents)]
    recommendations = []
    for index, (chunk, parent_index) in enumerate(zip(evidence, assignments)):
        parent_id = f"{doc_id}:parent:{parent_index:06d}" if parent_index is not None else None
        source_id = f"{doc_id}:evidence:{index:06d}"
        records.append(ChunkRecord(source_id, doc_id, parent_id, chunk.section_path, chunk.page_start, chunk.page_end, "evidence", chunk.text, chunk.token_count))
        for candidate in extract_recommendations_from_chunk(chunk.text):
            recommendations.append((source_id, candidate))
            rec_index = len(recommendations) - 1
            count = len(tokenizer.encode(candidate.recommendation_text, add_special_tokens=False))
            records.append(ChunkRecord(f"{doc_id}:recommendation:{rec_index:06d}", doc_id, parent_id, chunk.section_path, chunk.page_start, chunk.page_end, "recommendation", candidate.recommendation_text, count))
    return records, recommendations

def _insert_chunk(con: sqlite3.Connection, chunk: ChunkRecord) -> None:
    con.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (chunk.chunk_id, chunk.doc_id, chunk.parent_chunk_id, chunk.section_path, chunk.page_start, chunk.page_end, chunk.chunk_type, chunk.text, chunk.token_count))
    con.execute("INSERT INTO chunks_fts(chunk_id, doc_id, section_path, text) VALUES (?, ?, ?, ?)", (chunk.chunk_id, chunk.doc_id, chunk.section_path or "", chunk.text))

def insert_document_content(con: sqlite3.Connection, pdf_path: Path, file_hash: str, pages: Sequence[PageText], records: Sequence[ChunkRecord], recommendations: Sequence[Tuple[str, RecommendationCandidate]]) -> str:
    doc_id = doc_id_for(file_hash)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc_id,))
    con.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    con.execute("INSERT INTO documents(doc_id, source_path, file_hash, title, added_at) VALUES (?, ?, ?, ?, ?)", (doc_id, str(pdf_path), file_hash, pdf_path.stem, utc_now()))
    for page in pages:
        con.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?)", (f"{doc_id}:page:{page.page_index_0 + 1:06d}", doc_id, page.page_index_0 + 1, page.text, len(page.text), page.low_text_warning))
    for record in records:
        _insert_chunk(con, record)
    rec_chunks = [record for record in records if record.chunk_type == "recommendation"]
    for index, ((source_id, candidate), rec_chunk) in enumerate(zip(recommendations, rec_chunks)):
        con.execute("INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (f"{doc_id}:rec:{index:06d}", doc_id, source_id, rec_chunk.section_path, rec_chunk.page_start, rec_chunk.page_end, candidate.recommendation_text, candidate.strength, candidate.evidence_quality, candidate.polarity))
    return doc_id

def embed_missing_chunks(con: sqlite3.Connection, doc_id: str, embed_model: str, model, batch_size: int) -> int:
    rows = con.execute("SELECT c.chunk_id, c.text FROM chunks c LEFT JOIN embeddings e ON e.chunk_id=c.chunk_id AND e.embed_model=? WHERE c.doc_id=? AND e.chunk_id IS NULL ORDER BY c.chunk_id", (embed_model, doc_id)).fetchall()
    if not rows:
        return 0
    vectors = np.asarray(model.encode([str(row[1]) for row in rows], batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise RuntimeError("Embedding model returned an unexpected array shape")
    for (chunk_id, _), vector in zip(rows, vectors):
        con.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?)", (chunk_id, embed_model, int(vector.shape[0]), vector.tobytes(order="C")))
    return len(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--db_path", default="guidelines_vdb.sqlite")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chunk_tokens", "--chunk_words", dest="chunk_tokens", type=int, default=220)
    parser.add_argument("--overlap_tokens", "--overlap_words", dest="overlap_tokens", type=int, default=40)
    parser.add_argument("--parent_chunk_tokens", type=int, default=1000)
    parser.add_argument("--parent_overlap_tokens", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force_reindex", action="store_true")
    parser.add_argument("--reset_db", action="store_true")
    parser.add_argument("--ocr_if_low_text", action="store_true")
    args = parser.parse_args()
    if args.reset_db and not args.force_reindex:
        parser.error("--reset_db requires --force_reindex")
    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    if not pdf_dir.is_dir():
        parser.error(f"PDF directory does not exist: {pdf_dir}")
    try:
        init_db(db_path, reset_db=args.reset_db)
    except (RuntimeError, sqlite3.Error) as exc:
        raise SystemExit(f"Database initialization failed: {exc}") from exc
    pdfs = list(iter_pdf_paths(pdf_dir))
    if not pdfs:
        print(f"No PDFs found under: {pdf_dir}")
        return
    if AutoTokenizer is None or SentenceTransformer is None:
        raise SystemExit("Install sentence-transformers and transformers to build the VDB.")
    print(f"Loading tokenizer and embedding model: {args.embed_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.embed_model, use_fast=True)
        model = SentenceTransformer(args.embed_model)
    except Exception as exc:
        raise SystemExit(f"Failed to load embedding model '{args.embed_model}': {exc}") from exc
    with sqlite3.connect(str(db_path)) as con:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        for pdf_path in pdfs:
            try:
                file_hash = sha256_file(pdf_path)
                doc_id = doc_id_for(file_hash)
                existing = con.execute("SELECT doc_id FROM documents WHERE file_hash=?", (file_hash,)).fetchone()
                if existing and not args.force_reindex and document_has_embeddings(con, doc_id, args.embed_model):
                    print(f"Skipped (already indexed): {pdf_path}")
                    continue
                if existing and not args.force_reindex:
                    count = embed_missing_chunks(con, doc_id, args.embed_model, model, args.batch_size)
                    con.execute("UPDATE documents SET source_path=? WHERE doc_id=?", (str(pdf_path), doc_id))
                    con.commit()
                    print(f"Added {count} missing embeddings for {pdf_path}")
                    continue
                pages = pdf_to_pages(pdf_path, args.ocr_if_low_text)
                records, recommendations = build_document_records(doc_id, pages, tokenizer, args.chunk_tokens, args.overlap_tokens, args.parent_chunk_tokens, args.parent_overlap_tokens)
                if not records:
                    print(f"[WARN] No extractable text/chunks for PDF: {pdf_path}")
                    continue
                insert_document_content(con, pdf_path, file_hash, pages, records, recommendations)
                embedded = embed_missing_chunks(con, doc_id, args.embed_model, model, args.batch_size)
                con.commit()
                print(f"Indexed: {pdf_path} | pages={len(pages)} chunks={len(records)} recommendations={len(recommendations)} embeddings={embedded}")
            except (PdfReadError, DependencyError, RuntimeError, sqlite3.Error, ValueError) as exc:
                con.rollback()
                print(f"[ERROR] Failed to index {pdf_path}: {exc}", file=sys.stderr)
        print(f"Done. DB: {db_path}")

if __name__ == "__main__":
    main()
