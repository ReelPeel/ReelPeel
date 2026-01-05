import re

from ..core.base import PipelineStep
from ..core.models import PipelineState, Statement


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
                # Use abstract if available, otherwise fallback (or skip)
                text_to_check = ev.abstract or ev.summary or ""
                if not text_to_check.strip():
                    # Keep it if there's no text to check (manual review)
                    filtered_evidence.append(ev)
                    continue

                if self._is_related(stmt.text, text_to_check, prompt_tmpl):
                    filtered_evidence.append(ev)
                else:
                    # Optional: Log dropped evidence
                    pass

            # Update the statement with the filtered list
            stmt.evidence = filtered_evidence

        return state

    def _is_related(self, stmt_text: str, ev_text: str, tmpl: str) -> bool:
        try:
            prompt = tmpl.format(statement=stmt_text, evidence_summary=ev_text)
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

        def _fmt(val):
            if val is None:
                return None
            try:
                return f"{float(val):.2f}"
            except Exception:
                return None

        for stmt in state.statements:
            # 1. Build Evidence Block
            evidence_lines = []
            for ev in stmt.evidence:
                pmid = ev.pubmed_id or "N/A"
                parts = [f"PMID {pmid}"]

                w = _fmt(getattr(ev, "weight", None))
                if w is not None:
                    parts.append(f"w {w}")

                rel = _fmt(getattr(ev, "relevance", None))
                if rel is not None:
                    parts.append(f"rel {rel}")

                stance_info = []
                st = getattr(ev, "stance", None)
                if st is not None:
                    abs_label = getattr(st, "abstract_label", None)
                    abs_s = _fmt(getattr(st, "abstract_p_supports", None))
                    abs_r = _fmt(getattr(st, "abstract_p_refutes", None))
                    abs_n = _fmt(getattr(st, "abstract_p_neutral", None))
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

                    sum_label = getattr(st, "summary_label", None)
                    sum_s = _fmt(getattr(st, "summary_p_supports", None))
                    sum_r = _fmt(getattr(st, "summary_p_refutes", None))
                    sum_n = _fmt(getattr(st, "summary_p_neutral", None))
                    if sum_label or any(v is not None for v in [sum_s, sum_r, sum_n]):
                        probs = []
                        if sum_s is not None:
                            probs.append(f"S{sum_s}")
                        if sum_r is not None:
                            probs.append(f"R{sum_r}")
                        if sum_n is not None:
                            probs.append(f"N{sum_n}")
                        label = sum_label.value if sum_label is not None else "NA"
                        prob_txt = f" ({' '.join(probs)})" if probs else ""
                        stance_info.append(f"sum {label}{prob_txt}")

                if stance_info:
                    parts.append(f"stance {'; '.join(stance_info)}")

                # Use abstract (preferred) or summary
                content = (ev.abstract or ev.summary or "").strip().replace("\n", " ")
                if content:
                    parts.append(f"text: {content}")

                evidence_lines.append("- " + " | ".join(parts))

            evidence_block = "\n".join(evidence_lines) or "No evidence provided."

            # 2. Call LLM
            try:
                prompt = prompt_tmpl.format(
                    claim_text=stmt.text,
                    evidence_block=evidence_block,
                    transcript=transcript
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
