#!/usr/bin/env python3
"""
Guideline RAG retrieval step and CLI utility.

This module loads a SQLite "vector DB" built from guideline PDFs and retrieves
relevant guideline chunks for a given claim. It supports multi-query retrieval
for topic-normalized claims and can retrieve either across the full topic DB or
per individual guideline document.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from ..RAG_vdb.query_guideline_vdb import (
        DEFAULT_EMBED_MODEL,
        add_parent_context,
        retrieve,
        _load_sentence_transformer,
    )
    from ..core.base import PipelineStep
    from ..core.models import GuidelineDocumentResult, PipelineState, RAGEvidence
except ImportError:
    from pipeline.RAG_vdb.query_guideline_vdb import (
        DEFAULT_EMBED_MODEL,
        add_parent_context,
        retrieve,
        _load_sentence_transformer,
    )
    from pipeline.core.base import PipelineStep
    from pipeline.core.models import GuidelineDocumentResult, PipelineState, RAGEvidence


def _merge_text_with_parent(text: str, parent_text: str) -> str:
    text = (text or "").strip()
    parent_text = (parent_text or "").strip()
    if not parent_text or parent_text == text:
        return text
    return f"{text}\n\nParent context:\n{parent_text}"


def _result_to_rag_evidence(result: Dict[str, Any]) -> RAGEvidence:
    page_start = int(result.get("page_start") or 0)
    page_end = int(result.get("page_end") or page_start)
    pages = list(range(page_start, page_end + 1)) if page_start > 0 and page_end >= page_start else []
    score = float(
        result.get("dense_score")
        if result.get("dense_score") is not None
        else result.get("reranker_score")
        if result.get("reranker_score") is not None
        else result.get("fused_score")
        if result.get("fused_score") is not None
        else 0.0
    )
    text = _merge_text_with_parent(result.get("text", ""), result.get("parent_text"))
    return RAGEvidence(
        chunk_id=str(result["chunk_id"]),
        score=score,
        source_path=str(result.get("source_path") or ""),
        document_id=str(result.get("doc_id") or "") or None,
        document_title=str(result.get("title") or "") or None,
        pages=pages,
        abstract=text,
        weight=1.0,
        relevance=score,
        relevance_abstract=score,
    )


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _query_keywords(queries: Sequence[str]) -> set[str]:
    keywords = set()
    combined = " ".join(queries).casefold()
    for keyword in (
        "allergen", "allergene", "allergie", "beikost", "kuhmilch", "milch", "ei", "eier",
        "erdnuss", "nuss", "soja", "weizen", "gluten", "fisch", "meeresfrüchte", "meeresfruechte",
        "einführung", "einfuhrung", "früh", "fruh", "spätestens", "spaetestens", "verzögert",
        "verzoegert", "häufig", "haeufig", "regelmäßig", "regelmaessig",
    ):
        if keyword in combined:
            keywords.add(keyword)
    return keywords


_BIBLIOGRAPHY_HINTS = (
    "http://", "https://", "zugriff", "am j clin nutr", "appetite.", "the lancet", "world health organization",
)


def _looks_bibliographic(result: Dict[str, Any]) -> bool:
    text = (result.get("text") or "").casefold()
    section = (result.get("section_path") or "").casefold()
    title = str(result.get("title") or "").casefold()
    if any(token in section for token in ("literatur", "referenz", "quellen", "bibliograph")):
        return True
    if any(token in title for token in ("literatur", "referenz", "quellen")):
        return True
    if sum(token in text for token in _BIBLIOGRAPHY_HINTS) >= 2 and len(re.findall(r"\b\d{4}\b", text)) >= 3:
        return True
    return False


def _is_relevant_chunk(result: Dict[str, Any], query_keywords: set[str]) -> bool:
    if result.get("chunk_type") == "recommendation":
        return True
    if _looks_bibliographic(result):
        return False
    if not query_keywords:
        return True
    text = f"{result.get('text', '')} {result.get('parent_text', '')}".casefold()
    overlap = sum(1 for keyword in query_keywords if keyword in text)
    if overlap > 0:
        return True
    if any(keyword in text for keyword in ("allerg", "beikost", "einfuhr", "einführ", "säugling", "saugling")):
        return True
    return False


def _merge_result(existing: Dict[str, Any], candidate: Dict[str, Any], query_text: str) -> Dict[str, Any]:
    existing.setdefault("matched_queries", []).append(query_text)
    existing["matched_queries"] = _dedupe_strings(existing["matched_queries"])
    for numeric_field in ("dense_score", "fused_score", "reranker_score"):
        score = candidate.get(numeric_field)
        if score is None:
            continue
        if existing.get(numeric_field) is None or float(score) > float(existing[numeric_field]):
            existing[numeric_field] = float(score)
    existing["query_hits"] = len(existing["matched_queries"])
    if candidate.get("chunk_type") == "recommendation":
        existing["chunk_type"] = "recommendation"
    if float(candidate.get("fused_score") or 0.0) > float(existing.get("best_fused_result") or -1.0):
        existing.update({k: v for k, v in candidate.items() if k != "matched_queries"})
        existing["best_fused_result"] = float(candidate.get("fused_score") or 0.0)
        existing["matched_queries"] = _dedupe_strings(existing.get("matched_queries", []) + [query_text])
        existing["query_hits"] = len(existing["matched_queries"])
    return existing


def _load_guideline_documents(db_path: Path) -> List[Dict[str, str]]:
    with sqlite3.connect(str(db_path)) as con:
        rows = con.execute(
            "SELECT doc_id, source_path, title FROM documents ORDER BY source_path"
        ).fetchall()
    return [
        {
            "document_id": str(row[0]),
            "source_path": str(row[1]),
            "title": str(row[2]) if row[2] is not None else "",
        }
        for row in rows
    ]


def retrieve_chunks(
    db_path: Path,
    statement: str,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    dense_k: int = 30,
    bm25_k: int = 30,
    final_k: int = 50,
    rrf_k: int = 60,
    claim_mode: bool = True,
    include_parent_context: bool = True,
    reranker_model: str = "",
    rerank_top_n: int = 30,
    queries: Sequence[str] | None = None,
    topic_flags: Sequence[str] | None = None,
    doc_id: str | None = None,
    model: Any = None,
    reranker: Any = None,
) -> Tuple[str, List[RAGEvidence], List[RAGEvidence]]:
    db_path = db_path.expanduser().resolve()
    model = model or _load_sentence_transformer(embed_model)
    if reranker is None and reranker_model:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(reranker_model)

    query_list = _dedupe_strings(list(queries or []) + [statement])
    query_keywords = _query_keywords(query_list)
    candidate_limit = max(final_k * 3, rerank_top_n, 12)

    with sqlite3.connect(str(db_path)) as con:
        merged: Dict[str, Dict[str, Any]] = {}
        for query_text in query_list:
            results = retrieve(
                con,
                query_text,
                embed_model=embed_model,
                model=model,
                dense_k=dense_k,
                bm25_k=bm25_k,
                final_k=candidate_limit,
                rrf_k=rrf_k,
                claim_mode=claim_mode,
                reranker=reranker,
                rerank_top_n=rerank_top_n,
                doc_id=doc_id,
            )
            for result in results:
                chunk_id = str(result["chunk_id"])
                if chunk_id in merged:
                    merged[chunk_id] = _merge_result(merged[chunk_id], dict(result), query_text)
                else:
                    seeded = dict(result)
                    seeded["matched_queries"] = [query_text]
                    seeded["query_hits"] = 1
                    seeded["best_fused_result"] = float(result.get("fused_score") or 0.0)
                    merged[chunk_id] = seeded

        candidates = list(merged.values())
        if include_parent_context:
            add_parent_context(con, candidates)

    if topic_flags and "timing" in topic_flags:
        query_keywords.update({"früh", "fruh", "spätestens", "spaetestens", "verzögert", "verzoegert"})
    if topic_flags and "frequency" in topic_flags:
        query_keywords.update({"häufig", "haeufig", "regelmäßig", "regelmaessig"})

    raw_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item.get("fused_score") or 0.0),
            float(item.get("dense_score") or -1.0),
            int(item.get("query_hits") or 0),
        ),
        reverse=True,
    )
    raw_evidence = [_result_to_rag_evidence(result) for result in raw_candidates[:candidate_limit]]

    usable_candidates = [result for result in raw_candidates if _is_relevant_chunk(result, query_keywords)]
    for result in usable_candidates:
        combined_score = float(result.get("fused_score") or 0.0)
        combined_score += 0.02 * int(result.get("query_hits") or 0)
        combined_score += 0.03 * max(float(result.get("dense_score") or 0.0), 0.0)
        if result.get("chunk_type") == "recommendation":
            combined_score += 0.03
        result["combined_score"] = combined_score
    usable_candidates.sort(key=lambda item: float(item.get("combined_score") or 0.0), reverse=True)
    usable_evidence = [_result_to_rag_evidence(result) for result in usable_candidates[: max(0, final_k)]]
    return embed_model, usable_evidence, raw_evidence


def _sorted_evidence(evidence: Sequence[RAGEvidence]) -> List[RAGEvidence]:
    return sorted(
        evidence,
        key=lambda item: (
            str(item.source_path),
            -(float(item.score) if item.score is not None else 0.0),
            str(item.chunk_id),
        ),
    )


def release_torch_cuda_memory() -> None:
    """Release cached CUDA allocations after statement-level model work."""
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


class RetrieveGuidelineFactsStep(PipelineStep):
    """Retrieve topic guideline evidence for each statement."""

    def execute(self, state: PipelineState) -> PipelineState:
        if not state.statements:
            print(f"[{self.__class__.__name__}] No statements available; skipping guideline retrieval.")
            return state

        db_path = Path(self.config.get("db_path", "guidelines_vdb.sqlite")).expanduser().resolve()
        if not db_path.exists():
            raise FileNotFoundError(f"Guideline DB not found: {db_path}")

        topic_documents = _load_guideline_documents(db_path)
        if not topic_documents:
            raise ValueError(f"Guideline DB contains no documents: {db_path}")

        top_k = int(self.config.get("top_k", 50))
        per_guideline_top_k = int(self.config.get("per_guideline_top_k", min(max(top_k, 4), 8)))
        min_score = float(self.config.get("min_score", 0.25))
        dense_k = int(self.config.get("dense_k", max(top_k * 5, top_k)))
        bm25_k = int(self.config.get("bm25_k", max(top_k * 5, top_k)))
        rrf_k = int(self.config.get("rrf_k", 60))
        claim_mode = bool(self.config.get("claim_mode", True))
        include_parent_context = bool(self.config.get("include_parent_context", True))
        embed_model = str(self.config.get("embed_model", DEFAULT_EMBED_MODEL))
        reranker_model = str(self.config.get("reranker_model", "") or "")
        rerank_top_n = int(self.config.get("rerank_top_n", max(top_k * 5, top_k)))

        print(
            f"[{self.__class__.__name__}] Using per-guideline retrieval "
            f"(documents={len(topic_documents)}, dense_k={dense_k}, bm25_k={bm25_k}, per_guideline_top_k={per_guideline_top_k}, claim_mode={claim_mode})."
        )

        for stmt in state.statements:
            if stmt.guideline_label:
                stmt.retrieval_status = stmt.retrieval_status or "skipped_routed"
                stmt.raw_retrieved_chunk_count = stmt.raw_retrieved_chunk_count or 0
                stmt.usable_retrieved_chunk_count = stmt.usable_retrieved_chunk_count or 0
                stmt.guideline_documents = stmt.guideline_documents or []
                continue

            statement_text = (
                getattr(stmt, "canonical_claim_de", None)
                or getattr(stmt, "normalized_text", None)
                or stmt.text
                or ""
            ).strip()
            if not statement_text:
                stmt.retrieval_status = "no_query"
                stmt.raw_retrieved_chunk_count = 0
                stmt.usable_retrieved_chunk_count = 0
                stmt.guideline_documents = []
                continue

            query_list = list(getattr(stmt, "retrieval_queries", None) or [statement_text])
            document_results: List[GuidelineDocumentResult] = []
            flattened_evidence: List[RAGEvidence] = []
            raw_total = 0
            model = None
            reranker = None

            try:
                model = _load_sentence_transformer(embed_model)
                if reranker_model:
                    from sentence_transformers import CrossEncoder

                    reranker = CrossEncoder(reranker_model)

                for document in topic_documents:
                    _, usable_chunks, raw_chunks = retrieve_chunks(
                        db_path,
                        statement_text,
                        embed_model=embed_model,
                        dense_k=dense_k,
                        bm25_k=bm25_k,
                        final_k=per_guideline_top_k,
                        rrf_k=rrf_k,
                        claim_mode=claim_mode,
                        include_parent_context=include_parent_context,
                        reranker_model=reranker_model,
                        rerank_top_n=rerank_top_n,
                        queries=query_list,
                        topic_flags=getattr(stmt, "topic_flags", None) or [],
                        doc_id=document["document_id"],
                        model=model,
                        reranker=reranker,
                    )
                    filtered_chunks = [chunk for chunk in usable_chunks if float(chunk.score) >= min_score]
                    raw_total += len(raw_chunks)
                    flattened_evidence.extend(filtered_chunks)
                    document_results.append(
                        GuidelineDocumentResult(
                            document_id=document["document_id"],
                            source_path=document["source_path"],
                            title=document.get("title") or None,
                            raw_retrieved_chunk_count=len(raw_chunks),
                            retrieved_chunk_count=len(filtered_chunks),
                            evidence=_sorted_evidence(filtered_chunks),
                        )
                    )
            finally:
                del reranker
                del model
                release_torch_cuda_memory()

            stmt.guideline_documents = document_results
            stmt.evidence = _sorted_evidence(flattened_evidence)
            stmt.raw_retrieved_chunk_count = raw_total
            stmt.usable_retrieved_chunk_count = len(flattened_evidence)
            stmt.retrieval_status = "ok" if flattened_evidence else "no_usable_chunks"

            if self.debug:
                counts = ", ".join(
                    f"{doc.title or doc.document_id}:{doc.retrieved_chunk_count}" for doc in document_results
                )
                print(
                    f"   Statement {stmt.id}: raw={raw_total} usable={len(flattened_evidence)} "
                    f"queries={len(query_list)} docs=[{counts}]."
                )

        return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default="guidelines_vdb.sqlite")
    parser.add_argument("--statement", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--min_score", type=float, default=0.5)
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--doc_id")
    args = parser.parse_args()

    embed_model, chunks, raw_chunks = retrieve_chunks(
        db_path=Path(args.db_path),
        statement=args.statement,
        embed_model=args.embed_model,
        final_k=args.top_k,
        doc_id=args.doc_id,
    )
    chunks = [chunk for chunk in chunks if float(chunk.score) >= args.min_score]
    result: Dict[str, Any] = {
        "statement": args.statement,
        "embed_model": embed_model,
        "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
        "raw_retrieved_chunks": [chunk.model_dump() for chunk in raw_chunks],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
