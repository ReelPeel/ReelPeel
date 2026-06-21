"""Topic-specific claim typing and normalization before guideline retrieval."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from ..core.base import PipelineStep
from ..core.models import PipelineState, Statement


CLAIM_TYPE_RECOMMENDATION = "recommendation_paraphrase"
CLAIM_TYPE_DESCRIPTIVE = "descriptive_but_guideline_relevant"
CLAIM_TYPE_EXPLANATORY = "explanatory_non_recommendation"
CLAIM_TYPE_OFF_TOPIC = "off_topic"

BEIKOST_NON_RECOMMENDATION_LABEL = "keine Beikostempfehlung"


_FOOD_RULES: Sequence[tuple[re.Pattern[str], str, str]] = (
    (re.compile(r"\beier?n?\b", re.I), "Ei", "egg"),
    (re.compile(r"\berdn(ü|u)sse|erdnuss", re.I), "Erdnuss", "peanut"),
    (re.compile(r"\bbaumn(ü|u)sse|n(ü|u)sse\b", re.I), "Baumnüsse", "tree nuts"),
    (re.compile(r"\bsoja\b", re.I), "Soja", "soy"),
    (re.compile(r"\bweizen\b|\bgluten\b", re.I), "Weizen/Gluten", "wheat or gluten"),
    (re.compile(r"\bfisch\b", re.I), "Fisch", "fish"),
    (re.compile(r"\bmeer(es)?fr(ü|u)chte\b", re.I), "Meeresfrüchte", "seafood"),
    (re.compile(r"\bmilchprodukte\b|\bkuhmilch\b|\bmilch\b", re.I), "Kuhmilch", "cow's milk"),
)

_EXPLANATORY_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"treten\s+nicht\s+zwangsl[aä]ufig.*erste[nr]?\s+gabe", re.I),
    re.compile(r"k(ö|o)nnen\s+sich\s+(ü|u)ber\s+zeit\s+bilden", re.I),
    re.compile(r"allergi(en|sche\s+reaktionen?).*(bilden|entwickeln)", re.I),
    re.compile(r"unreife[nr]?\s+nieren", re.I),
    re.compile(r"wasserhaushalt", re.I),
)

_GUIDELINE_RELEVANT_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(allergen(e)?|allergene[nr]?\s+lebensmittel)\b", re.I),
    re.compile(r"\beinf(ü|u)hren|gabe|anbieten|verz(ö|o)gert|fr(ü|u)h|sp[aä]testens\b", re.I),
    re.compile(r"\bbeikost|erstes\s+lebensjahr|s(ä|ae)ugling|baby|babys\b", re.I),
    re.compile(r"\brestriktion|vermeidung|meidung|veraltet|empfehlung\b", re.I),
)

_DESCRIPTIVE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"geh(ö|o)ren\s+zu\s+den\s+h[aä]ufigsten\s+allergenen", re.I),
    re.compile(r"wahl\s+des\s+zeitpunktes", re.I),
)


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _food_mentions(text: str) -> List[tuple[str, str]]:
    matches: List[tuple[str, str]] = []
    for pattern, german, english in _FOOD_RULES:
        if pattern.search(text):
            matches.append((german, english))
    return matches


def _canonical_claim_de(stmt: Statement, claim_type: str, foods: Sequence[tuple[str, str]]) -> str:
    source = (stmt.translated_text_de or stmt.text or "").strip()
    if claim_type == CLAIM_TYPE_EXPLANATORY:
        return source
    if foods:
        joined = ", ".join(food for food, _ in foods)
        if "veraltet" in source.lower():
            return f"In der Beikost gilt die Empfehlung, {joined} im ersten Lebensjahr nicht zu meiden oder zu verzögern."
        if re.search(r"h[aä]ufig", source, re.I):
            return f"Die Leitlinie empfiehlt, {joined} in altersgerechter Form regelmäßig nach Einführung der Beikost anzubieten."
        if re.search(r"h[aä]ufigsten\s+allergenen", source, re.I):
            return f"Die Leitlinie behandelt die Einführung von {joined} in der Beikost zur Allergieprävention."
        if re.search(r"sp[aä]testens|fr(ü|u)h|verz(ö|o)gert", source, re.I):
            return f"Die Leitlinie macht eine Empfehlung zum Zeitpunkt der Einführung von {joined} in der Beikost."
    if re.search(r"allergene", source, re.I):
        return "Die Leitlinie macht eine Empfehlung zur Einführung potenziell allergener Lebensmittel in der Beikost."
    return source


def _canonical_claim_en(stmt: Statement, claim_type: str, foods: Sequence[tuple[str, str]]) -> str:
    source = (stmt.translated_text_en or stmt.text or "").strip()
    if claim_type == CLAIM_TYPE_EXPLANATORY:
        return source
    if foods:
        joined = ", ".join(food for _, food in foods)
        if "outdated" in source.lower() or "veraltet" in (stmt.text or "").lower():
            return f"The guideline recommends not delaying or avoiding {joined} during the first year of life."
        if re.search(r"often|frequently|h[aä]ufig", source, re.I):
            return f"The guideline recommends offering {joined} regularly in an age-appropriate form after complementary feeding starts."
        if re.search(r"most common allergen", source, re.I) or re.search(r"h[aä]ufigsten\s+allergenen", stmt.text or "", re.I):
            return f"The guideline addresses the introduction of {joined} in complementary feeding for allergy prevention."
        if re.search(r"latest|early|delayed|sp[aä]testens|fr(ü|u)h|verz(ö|o)gert", source, re.I):
            return f"The guideline makes a recommendation about the timing of introducing {joined} during complementary feeding."
    if re.search(r"allergen", source, re.I):
        return "The guideline makes a recommendation about introducing potentially allergenic foods during complementary feeding."
    return source


def _keyword_query(stmt: Statement, foods: Sequence[tuple[str, str]]) -> str:
    parts = ["Beikost", "Allergieprävention"]
    if foods:
        parts.extend(food for food, _ in foods)
    text = (stmt.translated_text_de or stmt.text or "")
    if re.search(r"fr(ü|u)h|sp[aä]testens|verz(ö|o)gert|zeitpunkt", text, re.I):
        parts.append("Einführung Zeitpunkt")
    elif re.search(r"h[aä]ufig", text, re.I):
        parts.append("regelmäßige Gabe")
    else:
        parts.append("Einführung Empfehlung")
    return " ".join(parts)


def classify_beikost_claim(stmt: Statement) -> tuple[str, List[str], str | None]:
    text = " ".join(filter(None, [stmt.text, stmt.translated_text_de, stmt.translated_text_en])).strip()
    flags: List[str] = []
    for pattern, german, _ in _FOOD_RULES:
        if pattern.search(text):
            flags.append(f"food:{german.casefold()}")
    if re.search(r"allergen", text, re.I):
        flags.append("allergen")
    if re.search(r"fr(ü|u)h|sp[aä]testens|verz(ö|o)gert|zeitpunkt", text, re.I):
        flags.append("timing")
    if re.search(r"h[aä]ufig|regelm[aä]ßig", text, re.I):
        flags.append("frequency")

    if any(pattern.search(text) for pattern in _EXPLANATORY_PATTERNS):
        return CLAIM_TYPE_EXPLANATORY, _dedupe(flags + ["non_recommendation"]), (
            "Background or mechanism claim about allergy development, not a complementary-feeding recommendation."
        )
    if not any(pattern.search(text) for pattern in _GUIDELINE_RELEVANT_PATTERNS):
        return CLAIM_TYPE_OFF_TOPIC, _dedupe(flags + ["off_topic"]), (
            "Claim is outside complementary-feeding recommendation scope."
        )
    if any(pattern.search(text) for pattern in _DESCRIPTIVE_PATTERNS):
        return CLAIM_TYPE_DESCRIPTIVE, _dedupe(flags + ["descriptive"]), None
    return CLAIM_TYPE_RECOMMENDATION, _dedupe(flags + ["recommendation"]), None


class TopicClaimNormalizationStep(PipelineStep):
    """Route topic-specific non-recommendation claims and build retrieval queries."""

    def execute(self, state: PipelineState) -> PipelineState:
        topic = str(self.config.get("topic", "")).strip()
        for stmt in state.statements:
            original = (stmt.text or "").strip()
            stmt.fallback_label_used = False
            if not original:
                continue

            if topic != "beikost":
                de = (stmt.translated_text_de or original).strip()
                en = (stmt.translated_text_en or original).strip()
                stmt.claim_type = CLAIM_TYPE_RECOMMENDATION
                stmt.canonical_claim_de = de
                stmt.canonical_claim_en = en
                stmt.retrieval_queries = _dedupe([de, en])
                stmt.topic_flags = ["generic"]
                stmt.normalized_text = de
                continue

            claim_type, flags, routing_reason = classify_beikost_claim(stmt)
            foods = _food_mentions(" ".join(filter(None, [stmt.text, stmt.translated_text_de, stmt.translated_text_en])))
            de = _canonical_claim_de(stmt, claim_type, foods)
            en = _canonical_claim_en(stmt, claim_type, foods)
            queries: List[str] = []
            if claim_type not in {CLAIM_TYPE_EXPLANATORY, CLAIM_TYPE_OFF_TOPIC}:
                queries.extend([
                    (stmt.translated_text_de or original).strip(),
                    de,
                    (stmt.translated_text_en or original).strip(),
                    en,
                    _keyword_query(stmt, foods),
                ])
            stmt.claim_type = claim_type
            stmt.topic_flags = flags
            stmt.routing_reason = routing_reason
            stmt.canonical_claim_de = de
            stmt.canonical_claim_en = en
            stmt.retrieval_queries = _dedupe(queries)[:5]
            stmt.normalized_text = de or (stmt.translated_text_de or original)

            if claim_type in {CLAIM_TYPE_EXPLANATORY, CLAIM_TYPE_OFF_TOPIC}:
                stmt.guideline_label = BEIKOST_NON_RECOMMENDATION_LABEL
                stmt.routing_reason = routing_reason or "Claim is not a complementary-feeding recommendation."
                stmt.retrieval_status = "skipped_non_recommendation"
                stmt.classification_status = "skipped_non_recommendation"
                stmt.failure_stage = None
                stmt.fallback_label_used = False
                stmt.cited_chunk_ids = []
                stmt.evidence = []
                stmt.raw_retrieved_chunk_count = 0
                stmt.usable_retrieved_chunk_count = 0
        return state
