# =============================================================================
# MODULE: deduplicate.py
# =============================================================================
# Purpose
# -------
# Deduplicate and diversify evidence items collected from multiple queries and/or
# multiple sources, prior to reranking and stance. This prevents:
#   - redundant scoring (wasted compute),
#   - multiple near-identical papers crowding the top of rankings,
#   - inflated confidence due to duplicates.
#
# Deduplication is applied at two levels:
#   (1) Hard dedup: exact identity (same PMID/DOI/URL)
#   (2) Near dedup: high textual similarity (same study described differently)
#
# Inputs (PipelineState prerequisites)
# -----------------------------------
# - state.statements: list[Statement]
# - Each Statement must provide:
#     - stmt.evidence: list[Evidence]
#
# - Evidence objects should provide (best-effort):
#     - ev.pubmed_id: str | None
#     - ev.url: str | None
#     - ev.doi: str | None (optional but excellent)
#     - ev.title / ev.article_title: str | None (recommended)
#     - ev.abstract: str | None (recommended)
#     - ev.pub_type: str|list[str] (optional; can help choose canonical item)
#
# Outputs (Fields written back to state)
# -------------------------------------
# - stmt.evidence: reduced list with duplicates removed.
# - Optional diagnostics:
#     - stmt.dedup_log: list of removed items with reasons (pmid, reason, kept_id)
#     - ev.duplicate_of: identifier of the canonical item (if you keep trace)
#
# High-level algorithm
# --------------------
# 1) Hard dedup pass
#    - Build a canonical key for each evidence item:
#        key = pubmed_id if present
#              else doi if present
#              else normalized_url if present
#              else normalized_title if present (fallback)
#    - Keep the “best” item per key and drop others.
#      “Best” can be chosen by:
#        - presence of abstract over summary only
#        - richer metadata (title, year, pub_type)
#        - higher source weight (if multi-source)
#
# 2) Near dedup pass (optional but recommended when unioning many queries)
#    - Compute text embeddings on title+abstract (or title+summary).
#    - For each item, compare to items already kept:
#        if cosine_similarity > threshold (e.g., 0.92..0.97), treat as near-dup
#    - Keep one canonical representative, drop the rest.
#    - To scale to large N, use:
#        - approximate nearest neighbor index (FAISS), or
#        - MMR-style selection with similarity penalty.
#
# 3) (Optional) Diversity constraints
#    - Enforce caps, e.g.:
#        max 2 reviews, max 1 editorial, at least 1 primary study if available.
#    - This is not strict “dedup”, but often implemented alongside dedup.
#
# Configuration knobs (typical)
# -----------------------------
# - hard_keys: ["pubmed_id","doi","url","title"]
# - normalize_title: bool
# - near_dedup: bool
# - embedding_model: sentence-transformers model id (or local embedder)
# - similarity_threshold: float (e.g., 0.95)
# - prefer_fields: ["abstract","title","year"] for choosing canonical
# - max_per_pub_type: dict or policy function (optional)
# - debug: bool
#
# Runtime prerequisites
# --------------------
# - Hard dedup: no ML dependencies required.
# - Near dedup:
#     - requires an embedding model (HF sentence-transformers or similar),
#       plus torch/transformers or a dedicated embedding runtime.
#     - OR use reranker/stance embeddings if already computed (avoid recompute).
#
# Failure modes & safeguards
# --------------------------
# - Missing IDs (no PMID/DOI): fall back to title hashing; may cause false merges.
# - Very short titles/abstracts: similarity becomes unstable; reduce threshold or
#   require minimum text length.
# - “Same topic” vs “same paper”: near-dedup threshold must be tuned; too low
#   removes legitimately distinct studies.
# - Transparency: always keep a dedup log for debugging and auditability.
#
# Downstream usage
# ----------------
# - Dedup runs BEFORE:
#     - reranking (saves compute)
#     - stance (reduces contradictory duplicates)
# - Also improves evidence diversity and prevents score inflation.
# =============================================================================
