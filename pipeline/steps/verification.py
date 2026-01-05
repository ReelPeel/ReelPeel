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
        source_id = getattr(ev, "pubmed_id", None) or "N/A"
    elif source_type == SourceType.EPISTEMONIKOS.value:
        source_label = "EPIST"
        source_id = getattr(ev, "epistemonikos_id", None) or "N/A"
    elif source_type == SourceType.RAG.value:
        source_label = "RAG"
        source_id = getattr(ev, "chunk_id", None) or "N/A"
    else:
        source_label = "SOURCE"
        source_id = "N/A"

    parts = [f"{source_label} {source_id}"]

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
        content = (
            getattr(ev, "abstract", None)
            or getattr(ev, "text", None)
            or ""
        ).strip().replace("\n", " ")
        if content:
            parts.append(f"text: {content}")
        elif missing_text:
            parts.append(missing_text)

    return "- " + " | ".join(parts)


def _format_rag_chunk(chunk, include_text: bool = True) -> str:
    source = getattr(chunk, "source_path", None) or "unknown"
    pages = getattr(chunk, "pages", None) or []
    if pages:
        source = f"{source} p.{','.join(str(p) for p in pages)}"

    parts = []
    score = _fmt_score(getattr(chunk, "score", None))
    if score is not None:
        parts.append(f"score {score}")
    weight = _fmt_score(getattr(chunk, "weight", None))
    if weight is not None:
        parts.append(f"w {weight}")

    if include_text:
        text = (getattr(chunk, "text", "") or "").strip().replace("\n", " ")
        if text:
            parts.append(f"text: {text}")
        else:
            parts.append("text: [no text]")

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
                if _source_type_value(ev) == SourceType.RAG.value:
                    filtered_evidence.append(ev)
                    continue

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
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 128),
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
            evidence_lines = []
            rag_lines = []
            for ev in stmt.evidence:
                if _source_type_value(ev) == SourceType.RAG.value:
                    rag_lines.append(_format_rag_chunk(ev, include_text=True))
                else:
                    evidence_lines.append(_format_evidence_line(ev, include_text=True))

            evidence_block = "\n".join(evidence_lines) or "No evidence provided."

            rag_block = "\n".join(rag_lines) or "No RAG chunks provided."

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
                    temperature=self.config.get('temperature', 0.1),
                    max_tokens=self.config.get('max_tokens', 512),
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
        threshold = self.config.get("threshold", 0.15)

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
