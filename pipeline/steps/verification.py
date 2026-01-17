"""
Verification module (steps 6-8): filter evidence, judge truthness, aggregate score.

This file defines the final pipeline steps that take ranked evidence and turn
it into statement-level verdicts and an overall truthiness score.

Included steps:
- FilterEvidenceStep (Step 6):
  Uses an LLM to decide whether each evidence item is relevant to the claim.
  PubMed/Epistemonikos/RAG evidence is formatted into a compact line (including
  relevance and any available stance metadata) and passed to the model.
  Only responses starting with "yes" are treated as relevant.
- TruthnessStep (Step 7):
  Builds an evidence block (plus RAG chunk block) and calls an LLM to produce
  a VERDICT and FINALSCORE. Parses those fields and writes them to Statement.
- ScoringStep (Step 8):
  Computes state.overall_truthiness as a weighted average. Scores below a
  threshold are up-weighted to penalize false or uncertain statements.

Inputs:
- state.statements with stmt.evidence already populated and scored.
- Evidence objects may contain relevance, weight, and stance fields.
- config keys: prompt_template, model, temperature, max_tokens, threshold.

Outputs:
- stmt.evidence filtered for relevance.
- stmt.verdict, stmt.score, stmt.rationale set by TruthnessStep.
- state.overall_truthiness set by ScoringStep.
"""

import re

from ..core.base import PipelineStep
from ..core.models import PipelineState, SourceType, Statement


def _fmt_score(val):
    if val is None:
        return None
    try:
        return f"{float(val):.2f}"
    except Exception:
        return None


def _source_type_value(ev):
    st = getattr(ev, "source_type", None)
    return st.value if hasattr(st, "value") else st


def _format_evidence_line(ev, include_text: bool = True, missing_text: str = "text: [no content]") -> str:
    source_type = _source_type_value(ev)
    if source_type == SourceType.PUBMED.value:
        source_label = "PMID"
    elif source_type == SourceType.EPISTEMONIKOS.value:
        source_label = "EPIST"
    elif source_type == SourceType.RAG.value:
        source_label = "RAG"
    else:
        source_label = "SOURCE"

    parts = [source_label]

    w = _fmt_score(getattr(ev, "weight", None))
    if w is not None:
        parts.append(f"w {w}")

    rel = _fmt_score(getattr(ev, "relevance", None))
    if rel is not None:
        parts.append(f"rel {rel}")

    stance_info = []
    st = getattr(ev, "stance", None)
    if st is not None:
        abs_label = getattr(st, "abstract_label", None)
        abs_s = _fmt_score(getattr(st, "abstract_p_supports", None))
        abs_r = _fmt_score(getattr(st, "abstract_p_refutes", None))
        abs_n = _fmt_score(getattr(st, "abstract_p_neutral", None))
        if abs_label or any(v is not None for v in [abs_s, abs_r, abs_n]):
            probs = []
            if abs_s is not None:
                probs.append(f"S{abs_s}")
            if abs_r is not None:
                probs.append(f"R{abs_r}")
            if abs_n is not None:
                probs.append(f"N{abs_n}")
            label = abs_label.value if abs_label is not None else "NA"
            prob_txt = f" ({' '.join(probs)})" if probs else ""
            stance_info.append(f"abs {label}{prob_txt}")

    if stance_info:
        parts.append(f"stance {'; '.join(stance_info)}")

    if include_text:
        content = (getattr(ev, "abstract", None) or "").strip().replace("\n", " ")
        if content:
            parts.append(f"text: {content}")
        elif missing_text:
            parts.append(missing_text)

    return "- " + " | ".join(parts)


def _format_rag_chunk(chunk, include_text: bool = True) -> str:
    source = "RAG"
    parts = []
    score = _fmt_score(getattr(chunk, "score", None))
    if score is not None:
        parts.append(f"score {score}")
    weight = _fmt_score(getattr(chunk, "weight", None))
    if weight is not None:
        parts.append(f"w {weight}")

    if include_text:
        text = (getattr(chunk, "abstract", "") or "").strip().replace("\n", " ")
        if text:
            parts.append(f"abstract: {text}")
        else:
            parts.append("abstract: [no abstract]")

    return "- " + source + (" | " + " | ".join(parts) if parts else "")


# -------------------------------------------------------------------------
# STEP 6: Filter Evidence (Reduce to relevant)
# -------------------------------------------------------------------------
class FilterEvidenceStep(PipelineStep):
    def execute(self, state: PipelineState) -> PipelineState:

        prompt_tmpl = self.config.get('prompt_template', "")

        for stmt in state.statements:
            # We will rebuild the evidence list with only relevant items
            filtered_evidence = []

            for ev in stmt.evidence:
                # Use abstract if available, otherwise skip filter
                text_to_check = getattr(ev, "abstract", None) or ""
                if not text_to_check.strip():
                    # Keep it if there's no text to check (manual review)
                    filtered_evidence.append(ev)
                    continue

                evidence_line = _format_evidence_line(ev, include_text=True, missing_text=None)
                if self._is_related(stmt.text, evidence_line, prompt_tmpl):
                    filtered_evidence.append(ev)
                else:
                    # Optional: Log dropped evidence
                    pass

            # Update the statement with the filtered list
            stmt.evidence = filtered_evidence

        return state

    def _is_related(self, stmt_text: str, evidence: str, tmpl: str) -> bool:
        try:
            prompt = tmpl.format(statement=stmt_text, evidence=evidence)
            res = self.llm.call(
                prompt=prompt,
                model=self.config.get('model'),
                temperature=self.config.get('temperature', 0),
                # max_tokens=self.config.get('max_tokens', 128),
            )
            self.log_artifact(f"Raw Output for Verification", res)
            return res.startswith("yes")
        except Exception as e:
            print(f"[ERROR] Step 6 Filter failed: {e}")
            return False

        # -------------------------------------------------------------------------


# STEP 7: Statement to Truthness (Fact Check)
# -------------------------------------------------------------------------
class TruthnessStep(PipelineStep):
    def execute(self, state: PipelineState) -> PipelineState:
        prompt_tmpl = self.config.get('prompt_template', "")
        transcript = state.transcript or ""

        for stmt in state.statements:
            stmt.evidence.sort(key=lambda ev: float(getattr(ev, "relevance", 0.0) or 0.0))
            # 1. Build Evidence Block
            pubmed_lines = []
            epistemonikos_lines = []  # TODO: confirm Epistemonikos formatting expectations for this list
            rag_lines = []
            for ev in stmt.evidence:
                source_type = _source_type_value(ev)
                if source_type == SourceType.RAG.value:
                    rag_lines.append(_format_rag_chunk(ev, include_text=True))
                elif source_type == SourceType.EPISTEMONIKOS.value:
                    epistemonikos_lines.append(_format_evidence_line(ev, include_text=True))
                else:
                    pubmed_lines.append(_format_evidence_line(ev, include_text=True))

            pubmed_block = "\n".join(pubmed_lines) or "No PubMed evidence provided."
            epistemonikos_block = "\n".join(epistemonikos_lines) or "No Epistemonikos evidence provided."
            rag_block = "\n".join(rag_lines) or "No RAG evidence provided."

            evidence_block = "\n".join(
                [
                    "PUBMED EVIDENCE:",
                    pubmed_block,
                    "",
                    "EPISTEMONIKOS EVIDENCE:",
                    epistemonikos_block,
                    "",
                    "RAG EVIDENCE:",
                    rag_block,
                ]
            )

            # 2. Call LLM
            try:
                prompt = prompt_tmpl.format(
                    claim_text=stmt.text,
                    evidence_block=evidence_block,
                    rag_chunks=rag_block,
                    transcript=transcript,
                )

                res = self.llm.call(
                    model=self.config.get('model'),
                    temperature=self.config.get('temperature', 0),
                    # max_tokens=self.config.get('max_tokens', 512),
                    prompt=prompt,
                )

                self.log_artifact(f"Raw Output for Statement {stmt.id}", res)

                # 3. Parse Output
                self._parse_verdict(stmt, res)

            except Exception as e:
                stmt.verdict = "error"
                stmt.rationale = f"LLM Call Failed: {e}"
                stmt.score = 0.0

        return state

    def _parse_verdict(self, stmt: Statement, reply: str):
        # Regex to find VERDICT and FINALSCORE
        verdict_match = re.search(r"VERDICT:\s*(true|false|uncertain)", reply, re.I)
        score_match = re.search(r"FINALSCORE:\s*([0-1](?:\.\d+)?)", reply, re.I)

        if verdict_match and score_match:
            stmt.verdict = verdict_match.group(1).lower()
            stmt.score = float(score_match.group(1))
            stmt.rationale = reply  # Store full reasoning
        else:
            stmt.verdict = "uncertain"
            stmt.score = 0.0
            stmt.rationale = f"Unparsable output:\n{reply}"


# -------------------------------------------------------------------------
# STEP 8: Scoring (Weighted Average)
# -------------------------------------------------------------------------
class ScoringStep(PipelineStep):
    def execute(self, state: PipelineState) -> PipelineState:
        threshold = self.config.get("threshold", 0.3)

        scores = [s.score for s in state.statements if s.score is not None]

        if not scores:
            state.overall_truthiness = 0.0
            return state

        # Weight logic: Triple weight if score is below threshold
        weights = [3 if c < threshold else 1 for c in scores]

        weighted_sum = sum(c * w for c, w in zip(scores, weights))
        total_weight = sum(weights)

        if total_weight > 0:
            state.overall_truthiness = round(weighted_sum / total_weight, 2)
        else:
            state.overall_truthiness = 0.0

        return state


# -----------------------------------------------------------------------------
# STEP 8 SCORING – Hinweis zur Semantik (Truthiness vs. Certainty)
# -----------------------------------------------------------------------------
# Problem:
# - Ein einzelner Skalar in [0,1] kann "Richtung" (wahr vs. falsch) und
#   "Sicherheit" (klar vs. unklar) nicht gleichzeitig sauber transportieren.
# - Wenn "uncertain" intern als 0.5 kodiert ist, ist 0.5 zwar neutral,
#   wird aber von vielen Nutzern als "eher schlecht" gelesen (50% = eher falsch).
#
# Empfehlung:
# - Zwei Größen statt einer:
#   1) overall_truthiness: Richtung in [0,1]
#      - 0.0 = absolut falsch
#      - 1.0 = absolut korrekt
#      - 0.5 = neutral / unbestimmt
#   2) overall_certainty: Sicherheit in [0,1]
#      - 0.0 = komplett unklar
#      - 1.0 = sehr klar
#   UI kann dann explizit kommunizieren:
#   - "Gesamt: unklar (certainty 0.12)" statt nur "0.50".
#
# Aggregationsidee:
# - Statements nahe 0.5 sollen die Richtung kaum beeinflussen.
# - Dafür wird als Gewicht die Distanz zur 0.5 verwendet (Unklarheit downweighten).
#
# Definitionen pro Statement-Score s_i in [0,1]:
# - Richtung (zentriert, normiert):
#     d_i = 2 * (s_i - 0.5)          # in [-1, 1]
# - Sicherheit (aus Richtung abgeleitet):
#     c_i = |d_i| = 2 * |s_i - 0.5|  # in [0, 1]
#   Interpretation:
#     c_i ~ 0   => sehr unklar (s_i ~ 0.5)
#     c_i ~ 1   => sehr klar (s_i nahe 0 oder 1)
# - Gewicht (Schärfung optional):
#     w_i = max(eps, c_i ** gamma)
#   eps   > 0 verhindert Division durch 0 / Null-Gewichte.
#   gamma >= 1 schärft (macht mittlere Sicherheiten weniger einflussreich).
#
# Aggregation über alle Statements:
# - Aggregierte Richtung:
#     D = sum(d_i * w_i) / sum(w_i)  # in [-1, 1]
# - Zurück auf [0,1]:
#     overall_truthiness = 0.5 + 0.5 * D
# - Gesamt-Sicherheit (einfacher Startpunkt):
#     overall_certainty = mean(c_i)
#   (alternativ konservativer: weighted mean, harmonic mean, min, etc.)
#
# Edge Case:
# - Wenn alle Statements uncertain sind (alle s_i ~ 0.5), dann c_i ~ 0.
#   => overall_certainty ~ 0 und overall_truthiness bleibt ~ 0.5,
#      aber wird eindeutig als "unklar" ausgewiesen.
#
# Beispiel-Intuition:
# - [0.5, 0.5, 0.5] => truthiness ~ 0.5, certainty 0
# - [0.9, 0.8, 0.5] => truthiness > 0.5, certainty 0.47
# - [0.9, 0.1, 0.5] => truthiness ~ 0.5, certainty 0.53
# -----------------------------------------------------------------------------
