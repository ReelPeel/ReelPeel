"""Evidence-only classification of claims against retrieved guideline chunks."""

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.base import PipelineStep
from ..core.models import GuidelineDocumentResult, PipelineState, RAGEvidence, SourceType, Statement


NOT_MENTIONED_LABEL = "wird nicht in Leitlinien genannt"
MATCH_LABEL = "entspricht Leitlinie"
CONTRADICT_LABEL = "widerspricht Leitlinien"
PARTIAL_LABEL = "entspricht teilweise Leitlinien"


def _strip_code_fence(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_classification_response(
    response: str,
    *,
    allowed_labels: List[str],
    available_chunk_ids: List[str],
    citation_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str]]:
    try:
        payload = json.loads(_strip_code_fence(response))
    except Exception as exc:
        raise ValueError("Classifier response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Classifier response must be a JSON object.")

    label = payload.get("label")
    citations = payload.get("citations")
    if label not in allowed_labels:
        raise ValueError(f"Invalid classification label: {label!r}")
    if not isinstance(citations, list) or any(not isinstance(x, str) for x in citations):
        raise ValueError("Classifier citations must be a list of chunk ID strings.")

    aliases = citation_aliases or {}
    citations = [aliases.get(citation, citation) for citation in citations]
    citations = list(dict.fromkeys(citations))
    available = set(available_chunk_ids)
    unknown = [chunk_id for chunk_id in citations if chunk_id not in available]
    if unknown:
        raise ValueError(f"Classifier cited unavailable chunks: {unknown}")
    if label != NOT_MENTIONED_LABEL and not citations:
        raise ValueError("Every mentioned-guideline label requires at least one citation.")

    return label, citations


def parse_unification_response(
    response: str,
    *,
    allowed_labels: List[str],
    available_doc_refs: Dict[str, GuidelineDocumentResult],
) -> Tuple[str, List[str]]:
    try:
        payload = json.loads(_strip_code_fence(response))
    except Exception as exc:
        raise ValueError("Unified classifier response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Unified classifier response must be a JSON object.")

    label = payload.get("label")
    documents = payload.get("documents")
    if label not in allowed_labels:
        raise ValueError(f"Invalid unified label: {label!r}")
    if not isinstance(documents, list) or any(not isinstance(x, str) for x in documents):
        raise ValueError("Unified classifier documents must be a list of document IDs.")

    documents = list(dict.fromkeys(documents))
    unknown = [doc for doc in documents if doc not in available_doc_refs]
    if unknown:
        raise ValueError(f"Unified classifier cited unavailable documents: {unknown}")
    if label != NOT_MENTIONED_LABEL and not documents:
        raise ValueError("Every mentioned-guideline label requires at least one cited guideline document.")
    return label, documents


def _format_chunks(evidence_items: Sequence[RAGEvidence]) -> Tuple[str, List[str], Dict[str, str]]:
    lines = []
    chunk_ids = []
    aliases: Dict[str, str] = {}
    for index, evidence in enumerate(evidence_items, start=1):
        source_type = getattr(evidence, "source_type", None)
        if hasattr(source_type, "value"):
            source_type = source_type.value
        if source_type != SourceType.RAG.value:
            raise ValueError("Guideline classification accepts RAG evidence only.")

        chunk_id = str(evidence.chunk_id)
        reference = f"C{index}"
        chunk_ids.append(chunk_id)
        aliases[reference] = chunk_id
        source = str(evidence.source_path)
        pages = ",".join(str(page) for page in evidence.pages) or "-"
        text = (evidence.abstract or "").strip()
        lines.append(f'<chunk id="{reference}" source="{source}" pages="{pages}">\n{text}\n</chunk>')
    return "\n\n".join(lines), chunk_ids, aliases


def _fallback_label(allowed_labels: List[str]) -> str:
    return NOT_MENTIONED_LABEL if NOT_MENTIONED_LABEL in allowed_labels else allowed_labels[0]


def _definition_block(allowed_labels: List[str], label_definitions: Dict[str, str]) -> str:
    return "\n".join(f"- {label}: {label_definitions.get(label, '')}" for label in allowed_labels)


def _group_stmt_evidence(stmt: Statement) -> List[GuidelineDocumentResult]:
    if stmt.guideline_documents:
        return list(stmt.guideline_documents)

    grouped: Dict[str, GuidelineDocumentResult] = {}
    for evidence in stmt.evidence:
        source_type = getattr(evidence, "source_type", None)
        if hasattr(source_type, "value"):
            source_type = source_type.value
        if source_type != SourceType.RAG.value:
            continue
        doc_id = getattr(evidence, "document_id", None) or str(evidence.chunk_id).split(":", 1)[0]
        if doc_id not in grouped:
            grouped[doc_id] = GuidelineDocumentResult(
                document_id=doc_id,
                source_path=str(evidence.source_path),
                title=getattr(evidence, "document_title", None),
            )
        grouped[doc_id].evidence.append(evidence)
        grouped[doc_id].retrieved_chunk_count += 1
    return sorted(grouped.values(), key=lambda item: (item.source_path, item.document_id))


def _build_document_prompt(
    *,
    topic: str,
    definitions: str,
    stmt: Statement,
    document: GuidelineDocumentResult,
    chunk_block: str,
) -> str:
    title = document.title or document.source_path
    return f'''You are a deterministic guideline comparison system for topic "{topic}".
You are evaluating the claim against ONE guideline document only.
Use ONLY the chunks from this single guideline document below. Do not use medical knowledge, assumptions, or facts not explicitly present in those chunks.
Return exactly one JSON object with this shape:
{{"label": "<one allowed label>", "citations": ["<chunk id>", "..."]}}

Allowed labels and their meanings:
{definitions}

Guideline document: {title}
Claim type: {stmt.claim_type or "unknown"}
Routing hint: {stmt.routing_reason or "none"}
Canonical German claim: {stmt.canonical_claim_de or stmt.translated_text_de or stmt.text}
Canonical English claim: {stmt.canonical_claim_en or stmt.translated_text_en or stmt.text}

Rules:
- Evaluate ONLY whether this single guideline document supports, contradicts, partially supports, or does not mention the claim.
- Cite only short chunk IDs shown below, such as "C1" or "C2".
- Copy each cited short chunk ID exactly; never shorten or expand it.
- Every label except "{NOT_MENTIONED_LABEL}" requires at least one citation.
- Use "{MATCH_LABEL}" only when the claim is contentwise fully consistent with this guideline document in meaning, including central specifications.
- Use "{PARTIAL_LABEL}" when central aspects match this document, but the claim omits important details, or adds unsupported aspects.
- Use "{CONTRADICT_LABEL}" when the claim fully or in central aspects directly contradicts this guideline document.
- Use "{NOT_MENTIONED_LABEL}" only when this document does not address the relevant content aspect in a meaningful way.
- Close recommendation paraphrases may count as direct matches.
- If an important specification is missing, do not infer it from background knowledge; prefer "{PARTIAL_LABEL}" or "{NOT_MENTIONED_LABEL}" depending on whether the central aspect is still covered.
- Do not include reasoning, explanation, markdown, or additional keys.
- Only use the allowed labels.

CLAIM (ORIGINAL):
{stmt.text}

CLAIM (GERMAN):
{getattr(stmt, "translated_text_de", None) or stmt.text}

CLAIM (ENGLISH):
{getattr(stmt, "translated_text_en", None) or stmt.text}

GUIDELINE CHUNKS:
{chunk_block}
'''


def _chunk_preview(text: str, limit: int = 400) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _build_unification_prompt(
    *,
    topic: str,
    definitions: str,
    stmt: Statement,
    doc_refs: Dict[str, GuidelineDocumentResult],
) -> str:
    lines = []
    for ref, document in doc_refs.items():
        lines.append(f"{ref}: {document.title or document.source_path}")
        lines.append(f"  label: {document.label}")
        if document.cited_chunk_ids:
            cited = {e.chunk_id: e for e in document.evidence}
            for chunk_id in document.cited_chunk_ids[:3]:
                evidence = cited.get(chunk_id)
                if evidence is None:
                    continue
                pages = ",".join(str(page) for page in evidence.pages) or "-"
                lines.append(f"  cited chunk {chunk_id} pages={pages}: {_chunk_preview(evidence.abstract)}")
        else:
            lines.append("  cited chunks: none")
    document_block = "\n".join(lines)
    return f'''You are a deterministic guideline aggregation system for topic "{topic}".
Your job is to combine per-guideline labels into ONE final label for the claim.
Use ONLY the document-level labels and cited excerpts below. Do not use outside knowledge.
Return exactly one JSON object with this shape:
{{"label": "<one allowed label>", "documents": ["<document ref>", "..."]}}

Allowed labels and their meanings:
{definitions}

Rules:
- Cite only document refs shown below, such as "D1" or "D2".
- Every label except "{NOT_MENTIONED_LABEL}" requires at least one cited document ref.
- Use "{MATCH_LABEL}" only when the claim is fully supported overall by the guideline set, with no missing central qualification in the documents that address the claim.
- Use "{PARTIAL_LABEL}" when support is incomplete, qualified, or mixed across the guideline set, or when only some addressed aspects match.
- Use "{CONTRADICT_LABEL}" when the guidelines that address the central claim contradict it overall.
- Use "{NOT_MENTIONED_LABEL}" only when the guideline set does not meaningfully address the relevant content aspect.
- Documents labeled "{NOT_MENTIONED_LABEL}" do not by themselves force a partial label.
- If some documents support the claim and others contradict or qualify it, prefer "{PARTIAL_LABEL}" unless the contradiction clearly dominates the central claim.
- Do not include reasoning, explanation, markdown, or additional keys.

CLAIM (ORIGINAL):
{stmt.text}

CLAIM (GERMAN):
{getattr(stmt, "translated_text_de", None) or stmt.text}

CLAIM (ENGLISH):
{getattr(stmt, "translated_text_en", None) or stmt.text}

PER-GUIDELINE DECISIONS:
{document_block}
'''


def _heuristic_unified_label(labels: Sequence[str], allowed_labels: List[str]) -> str:
    normalized = [label for label in labels if label]
    if any(label == MATCH_LABEL for label in normalized):
        return MATCH_LABEL if MATCH_LABEL in allowed_labels else _fallback_label(allowed_labels)

    mentioned = [label for label in normalized if label != NOT_MENTIONED_LABEL]
    if not mentioned:
        return _fallback_label(allowed_labels)

    if CONTRADICT_LABEL in mentioned:
        if PARTIAL_LABEL in mentioned:
            return PARTIAL_LABEL if PARTIAL_LABEL in allowed_labels else CONTRADICT_LABEL
        return CONTRADICT_LABEL if CONTRADICT_LABEL in allowed_labels else _fallback_label(allowed_labels)

    if PARTIAL_LABEL in mentioned:
        return PARTIAL_LABEL if PARTIAL_LABEL in allowed_labels else _fallback_label(allowed_labels)

    return _fallback_label(allowed_labels)


def _documents_for_final_label(documents: Sequence[GuidelineDocumentResult], final_label: str) -> List[GuidelineDocumentResult]:
    if final_label == MATCH_LABEL:
        selected = [document for document in documents if document.label == MATCH_LABEL]
        return selected or list(documents)
    if final_label == NOT_MENTIONED_LABEL:
        return []
    if final_label == CONTRADICT_LABEL:
        selected = [document for document in documents if document.label == CONTRADICT_LABEL]
        return selected or [document for document in documents if document.label != NOT_MENTIONED_LABEL]
    if final_label == PARTIAL_LABEL:
        selected = [document for document in documents if document.label == PARTIAL_LABEL]
        if selected:
            return selected
        return [document for document in documents if document.label in {CONTRADICT_LABEL, MATCH_LABEL}]
    return [document for document in documents if document.label != NOT_MENTIONED_LABEL]


def _union_citations(documents: Sequence[GuidelineDocumentResult]) -> List[str]:
    citations: List[str] = []
    seen = set()
    for document in documents:
        for chunk_id in document.cited_chunk_ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            citations.append(chunk_id)
    return citations


class GuidelineClassificationStep(PipelineStep):
    """Classify against each guideline document, then unify into one final label."""

    def execute(self, state: PipelineState) -> PipelineState:
        topic = str(self.config.get("topic", "")).strip()
        allowed_labels = list(self.config.get("allowed_labels") or [])
        label_definitions: Dict[str, str] = dict(self.config.get("label_definitions") or {})
        model = self.config.get("model")
        max_retries = int(self.config.get("max_retries", 1))

        if not topic:
            raise ValueError("Guideline classifier requires a topic.")
        if NOT_MENTIONED_LABEL not in allowed_labels:
            raise ValueError(f"Allowed labels must contain {NOT_MENTIONED_LABEL!r}.")
        if not model:
            raise ValueError("Guideline classifier requires an LLM model.")

        definitions = _definition_block(allowed_labels, label_definitions)

        for stmt in state.statements:
            if stmt.guideline_label:
                stmt.classification_status = stmt.classification_status or "skipped_pre_labeled"
                continue

            documents = _group_stmt_evidence(stmt)
            stmt.guideline_documents = documents
            if not documents:
                stmt.guideline_label = _fallback_label(allowed_labels)
                stmt.cited_chunk_ids = []
                stmt.classification_status = "fallback_no_evidence"
                stmt.fallback_label_used = True
                continue

            document_fallback_used = False
            for document in documents:
                if not document.evidence:
                    document.label = _fallback_label(allowed_labels)
                    document.cited_chunk_ids = []
                    document.classification_status = "fallback_no_evidence"
                    document.fallback_label_used = True
                    document_fallback_used = True
                    continue

                chunk_block, available_chunk_ids, citation_aliases = _format_chunks(document.evidence)
                prompt = _build_document_prompt(
                    topic=topic,
                    definitions=definitions,
                    stmt=stmt,
                    document=document,
                    chunk_block=chunk_block,
                )
                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        response = self.llm.call(
                            prompt=prompt,
                            model=model,
                            temperature=0.0,
                            max_tokens=int(self.config.get("max_tokens", 256)),
                        )
                        label, citations = parse_classification_response(
                            response,
                            allowed_labels=allowed_labels,
                            available_chunk_ids=available_chunk_ids,
                            citation_aliases=citation_aliases,
                        )
                        document.label = label
                        document.cited_chunk_ids = citations
                        document.classification_status = "ok"
                        document.fallback_label_used = False
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < max_retries:
                            prompt += (
                                "\n\nYour previous response was invalid. Return only the required "
                                f"JSON object. Validation error: {exc}"
                            )
                if last_error is not None:
                    document.label = _fallback_label(allowed_labels)
                    document.cited_chunk_ids = []
                    document.classification_status = "fallback_after_invalid_response"
                    document.fallback_label_used = True
                    document_fallback_used = True

            stmt.guideline_label = _heuristic_unified_label(
                [document.label or NOT_MENTIONED_LABEL for document in documents],
                allowed_labels,
            )
            stmt.cited_chunk_ids = _union_citations(
                _documents_for_final_label(documents, stmt.guideline_label)
            )

            stmt.classification_status = "ok"
            if document_fallback_used:
                stmt.classification_status = "ok_with_fallback"
            stmt.failure_stage = None
            stmt.fallback_label_used = document_fallback_used

        return state
