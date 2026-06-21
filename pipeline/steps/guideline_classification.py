"""Evidence-only classification of claims against retrieved guideline chunks."""

import json
import re
from typing import Dict, List, Optional, Tuple

from ..core.base import PipelineStep
from ..core.models import PipelineState, SourceType, Statement


NOT_MENTIONED_LABEL = "wird nicht in Leitlinien genannt"
NON_RECOMMENDATION_LABEL = "keine Beikostempfehlung"


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
    if label not in {NOT_MENTIONED_LABEL, NON_RECOMMENDATION_LABEL} and not citations:
        raise ValueError("Every mentioned-guideline label requires at least one citation.")

    return label, citations


def _format_chunks(stmt: Statement) -> Tuple[str, List[str], Dict[str, str]]:
    lines = []
    chunk_ids = []
    aliases: Dict[str, str] = {}
    for index, evidence in enumerate(stmt.evidence, start=1):
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
        lines.append(
            f'<chunk id="{reference}" source="{source}" pages="{pages}">\n{text}\n</chunk>'
        )
    return "\n\n".join(lines), chunk_ids, aliases


def _fallback_label(stmt: Statement, allowed_labels: List[str], *, no_evidence: bool) -> str:
    if stmt.claim_type in {"explanatory_non_recommendation", "off_topic"} and NON_RECOMMENDATION_LABEL in allowed_labels:
        return NON_RECOMMENDATION_LABEL
    if no_evidence:
        return NOT_MENTIONED_LABEL if NOT_MENTIONED_LABEL in allowed_labels else allowed_labels[0]
    return NOT_MENTIONED_LABEL if NOT_MENTIONED_LABEL in allowed_labels else allowed_labels[0]


class GuidelineClassificationStep(PipelineStep):
    """Classify using only RAG chunks and return an allowed label plus citations."""

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

        definitions = "\n".join(
            f"- {label}: {label_definitions.get(label, '')}" for label in allowed_labels
        )

        for stmt in state.statements:
            if stmt.guideline_label:
                stmt.classification_status = stmt.classification_status or "skipped_pre_labeled"
                continue

            chunk_block, available_chunk_ids, citation_aliases = _format_chunks(stmt)
            if not available_chunk_ids:
                stmt.guideline_label = _fallback_label(stmt, allowed_labels, no_evidence=True)
                stmt.cited_chunk_ids = []
                stmt.classification_status = "fallback_no_evidence"
                stmt.fallback_label_used = True
                continue

            prompt = f'''You are a deterministic guideline comparison system for topic "{topic}".
Use ONLY the guideline chunks below. Do not use medical knowledge, assumptions, or facts not explicitly present in those chunks.
Return exactly one JSON object with this shape:
{{"label": "<one allowed label>", "citations": ["<chunk id>", "..."]}}

Allowed labels and their meanings:
{definitions}

Claim type: {stmt.claim_type or "unknown"}
Routing hint: {stmt.routing_reason or "none"}
Canonical German claim: {stmt.canonical_claim_de or stmt.translated_text_de or stmt.text}
Canonical English claim: {stmt.canonical_claim_en or stmt.translated_text_en or stmt.text}

Rules:
- Cite only short chunk IDs shown below, such as "C1" or "C2".
- Copy each cited short chunk ID exactly; never shorten or expand it.
- Every label except "{NOT_MENTIONED_LABEL}" and "{NON_RECOMMENDATION_LABEL}" requires at least one citation.
- Use "entspricht Leitlinie" only when the claim is fully consistent with the guideline in meaning, including central qualifications.
- Use "entspricht teilweise Leitlinien" when central aspects match, but the claim is too general, imprecise, omits important details, or adds unsupported aspects.
- Use "widerspricht Leitlinien" when the claim fully or in central aspects directly contradicts the guideline.
- Use "{NOT_MENTIONED_LABEL}" only when the guidelines provide no information on the relevant content aspect and no meaningful link to the guideline content can be established.
- Use "{NON_RECOMMENDATION_LABEL}" only for background or explanatory claims that are not complementary-feeding recommendations.
- Close recommendation paraphrases may count as direct matches.
- If an important specification is missing, do not infer it from background knowledge; prefer "entspricht teilweise Leitlinien" or "{NOT_MENTIONED_LABEL}" depending on whether the central aspect is still covered.
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
                    stmt.guideline_label = label
                    stmt.cited_chunk_ids = citations
                    stmt.classification_status = "ok"
                    stmt.fallback_label_used = False
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
                stmt.guideline_label = _fallback_label(stmt, allowed_labels, no_evidence=False)
                stmt.cited_chunk_ids = []
                stmt.classification_status = "fallback_after_invalid_response"
                stmt.failure_stage = "classification"
                stmt.fallback_label_used = True

        return state
