"""
Design spec for a future deduplication step (no executable code yet).

Purpose
-------
Deduplicate and diversify evidence items collected from multiple queries and/or
sources before reranking and stance. This prevents redundant scoring and avoids
near-identical studies crowding the top of the rankings.

Current status
--------------
- This module is documentation only; there is no PipelineStep implemented here.

Expected inputs (PipelineState prerequisites)
---------------------------------------------
- state.statements: list[Statement]
- Each Statement should provide:
  - stmt.evidence: list[Evidence]

- Evidence objects should provide (best-effort):
  - ev.pubmed_id: str | None
  - ev.url: str | None
  - ev.doi: str | None (optional but valuable)
  - ev.title / ev.article_title: str | None (recommended)
  - ev.abstract: str | None (recommended)
  - ev.pub_type: str | list[str] (optional; can help choose a canonical item)

Expected outputs (fields written back to state)
----------------------------------------------
- stmt.evidence: reduced list with duplicates removed.
- Optional diagnostics:
  - stmt.dedup_log: list of removed items with reasons (pmid, reason, kept_id)
  - ev.duplicate_of: identifier of the canonical item (if you keep trace)

Algorithm sketch
----------------
1) Hard dedup pass
   - Build a canonical key for each evidence item:
     key = pubmed_id if present
           else doi if present
           else normalized_url if present
           else normalized_title if present (fallback)
   - Keep the "best" item per key and drop others.
     "Best" can be chosen by:
     - presence of abstract over summary only
     - richer metadata (title, year, pub_type)
     - higher source weight (if multi-source)

2) Near dedup pass (optional but recommended when unioning many queries)
   - Compute text embeddings on title+abstract (or title+summary).
   - For each item, compare to items already kept:
     if cosine_similarity > threshold (e.g., 0.92..0.97), treat as near-dup.
   - Keep one canonical representative, drop the rest.
   - To scale to large N, use:
     - approximate nearest neighbor index (FAISS), or
     - MMR-style selection with similarity penalty.

3) Optional diversity constraints
   - Enforce caps, e.g.:
     max 2 reviews, max 1 editorial, at least 1 primary study if available.
   - This is not strict dedup, but often implemented alongside it.

Configuration knobs (typical)
-----------------------------
- hard_keys: ["pubmed_id", "doi", "url", "title"]
- normalize_title: bool
- near_dedup: bool
- embedding_model: sentence-transformers model id (or local embedder)
- similarity_threshold: float (e.g., 0.95)
- prefer_fields: ["abstract", "title", "year"] for choosing canonical
- max_per_pub_type: dict or policy function (optional)
- debug: bool

Runtime prerequisites
--------------------
- Hard dedup: no ML dependencies required.
- Near dedup: requires an embedding model (sentence-transformers or similar).

Failure modes and safeguards
----------------------------
- Missing IDs (no PMID/DOI): fallback to title hashing can cause false merges.
- Very short titles/abstracts: similarity is unstable; require minimum length.
- "Same topic" vs "same paper": thresholds must be tuned to avoid over-pruning.
- Transparency: keep a dedup log for debugging and auditability.

Downstream usage
----------------
- Intended to run before reranking and stance scoring.
- Improves evidence diversity and reduces score inflation from duplicates.
"""
