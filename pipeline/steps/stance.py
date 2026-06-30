"""
Step: Compute stance (NLI) for evidence relative to each claim.

This step runs a natural language inference model to decide whether each piece
of evidence supports, refutes, or is neutral toward the claim. It writes the
probabilities and label into the nested Evidence.stance model.

Inputs:
- state.statements with Statement.text and stmt.evidence populated.
- Evidence fields used: abstract and/or text (configurable).
- Optional evidence titles are scored once as a separate weak candidate.

Config keys:
- model_name: default "cnut1648/biolinkbert-mednli".
- device: "cuda", "cpu", or explicit device string.
- use_fp16: use fp16 weights on CUDA.
- batch_size, max_length: inference batching and truncation.
- evidence_fields: list of evidence fields to score, typically ["abstract", "text"].
- top_m_by_relevance: if set, only score the top-M evidence by relevance.
- section_chunking_enabled: default True. If strict section headers are found,
  sections are scored as individual candidates; otherwise the full text falls
  back to fixed tokenizer chunks with 64-token overlap.
- diagnostic_test_gate_enabled: default True. Downgrades `Supports` to `Neutral`
  for diagnostic/test claims when the evidence describes diagnostic uncertainty
  or a formal-test mismatch with a lay skin/application procedure.
- Titles are scored once per evidence item and downweighted in aggregation.

Outputs:
- ev.stance.abstract_label set to Supports/Refutes/Neutral.
- ev.stance.abstract_p_supports/refutes/neutral set to probabilities.

Label mapping:
- Uses model.config.id2label when available; otherwise assumes
  0=entailment, 1=neutral, 2=contradiction.
- Maps entailment -> Supports, contradiction -> Refutes, neutral -> Neutral.

Runtime notes:
- Requires torch and transformers.
- Models are cached in-process to avoid repeated loads.
- Long sections/texts are chunked to fit max_length.

"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..core.base import PipelineStep
from ..core.models import PipelineState, Stance as EvidenceStance, StanceLabel

_MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[Any, Any]] = {}

_DIAGNOSTIC_CLAIM_RE = re.compile(
    r"\b("
    r"test|tests|tested|testing|"
    r"diagnos(?:e|es|ed|ing|is|tic|tics)|"
    r"detect|detects|detected|detecting|"
    r"screen|screens|screened|screening|"
    r"check|checks|checked|checking|"
    r"identify|identifies|identified|identifying|"
    r"confirm|confirms|confirmed|confirming|"
    r"rule\s*out"
    r")\b",
    re.I,
)
_PROCEDURE_SKIN_CLAIM_RE = re.compile(
    r"\b(apply|applying|rub|rubbing|place|placing|small\s+amount)\b.*\bskin\b"
    r"|\bskin\b.*\b(apply|applying|rub|rubbing|place|placing|small\s+amount)\b",
    re.I,
)
_DIAGNOSTIC_UNCERTAINTY_RE = re.compile(
    r"\b("
    r"gold\s+standard|oral\s+food\s+challenge|"
    r"double[- ]blind|placebo[- ]controlled|"
    r"mainly\s+clinical|primarily\s+clinical|"
    r"not\s+(?:so\s+)?accurate|not\s+reliable|not\s+definitive|"
    r"limited\s+accuracy|poor\s+accuracy|single\s+diagnosis"
    r")\b",
    re.I,
)
_FORMAL_DIAGNOSTIC_TEST_RE = re.compile(
    r"\b("
    r"skin[- ]prick|prick\s+test|patch\s+test|atopy\s+patch|"
    r"sige|specific\s+ige|oral\s+food\s+challenge|food\s+challenge|"
    r"double[- ]blind\s+placebo[- ]controlled"
    r")\b",
    re.I,
)


def _pick_device(device_cfg: Optional[str]) -> str:
    # Mirrors rerank.py behavior
    if device_cfg and str(device_cfg).lower() != "auto":
        return device_cfg
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model_name: str, device: str, use_fp16: bool):
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("Missing dependencies for stance. Install: pip install torch transformers")

    key = (model_name, device, use_fp16)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch_dtype = None
    if use_fp16 and device.startswith("cuda"):
        torch_dtype = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.eval()
    model.to(device)

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _batch(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _count_batch_tokens(inputs: Dict[str, torch.Tensor]) -> int:
    attn = inputs.get("attention_mask")
    if attn is not None:
        return int(attn.sum().item())
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return 0
    return int(input_ids.numel())


def _infer_nli_indices(model) -> Tuple[int, int, int]:
    """
    Returns: (entailment_idx, neutral_idx, contradiction_idx)

    Uses model.config.id2label when available. Falls back to common conventions.
    """
    id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    norm: Dict[int, str] = {}

    for k, v in id2label.items():
        try:
            i = int(k)
        except Exception:
            i = k  # best-effort
        norm[int(i)] = str(v).strip().lower()

    ent = neu = con = None
    for idx, lab in norm.items():
        if "entail" in lab:
            ent = idx
        elif "neutral" in lab:
            neu = idx
        elif "contrad" in lab:
            con = idx

    if ent is not None and neu is not None and con is not None:
        return int(ent), int(neu), int(con)

    # Fallback for typical 3-class NLI heads:
    #   0=entailment, 1=neutral, 2=contradiction
    return 0, 1, 2


class StanceEvidenceStep(PipelineStep):
    """
    Config keys:
      - model_name: str (default: "cnut1648/biolinkbert-mednli")
      - device: "cuda", "cuda:0", "cpu", "auto" (default: auto)
      - use_fp16: bool (default: True)
      - batch_size: int (default: 16)
      - max_length: int (default: 512)

      - evidence_fields: list[str] (default: ["abstract", "text"])
          Which Evidence fields to compute stance on.

      - top_m_by_relevance: int | None (default: None)
          If set, only compute stance for the Top-M evidence items per statement,
          selected by ev.relevance (descending). Others are left as-is.

      - section_chunking_enabled: bool (default: True)
          If true, strict abstract section headers are used as stance candidates.
          If false, use fixed tokenizer chunks over the whole abstract/text.

      - diagnostic_test_gate_enabled: bool (default: True)
          If true, conservative post-processing prevents diagnostic/test evidence
          with uncertainty or method-mismatch cues from supporting a lay claim.

      - Titles are scored once per evidence item and downweighted in aggregation.
      - Long section/text candidates are split into fixed tokenizer chunks with
        64-token overlap.
    """
    
    def execute(self, state: PipelineState) -> PipelineState:
        print("Entering StanceEvidenceStep...")
        model_name = self.config.get("model_name", "cnut1648/biolinkbert-mednli")
        device = _pick_device(self.config.get("device"))
        use_fp16 = bool(self.config.get("use_fp16", True))
        batch_size = int(self.config.get("batch_size", 16))
        max_length = int(self.config.get("max_length", 512))
        print("StanceEvidenceStep Flag1")
        evidence_fields = self.config.get("evidence_fields", ["abstract", "text"])
        if "abstract" not in evidence_fields and "text" not in evidence_fields:
            evidence_fields = ["abstract", "text"]
        top_m = self.config.get("top_m_by_relevance", None)
        default_section_chunking_enabled = True
        section_chunking_cfg = self.config.get("section_chunking_enabled", default_section_chunking_enabled)
        if isinstance(section_chunking_cfg, str):
            section_chunking_enabled = section_chunking_cfg.strip().lower() not in {"0", "false", "no", "off"}
        else:
            section_chunking_enabled = bool(section_chunking_cfg)
        default_diagnostic_test_gate_enabled = True
        diagnostic_test_gate_cfg = self.config.get(
            "diagnostic_test_gate_enabled",
            default_diagnostic_test_gate_enabled,
        )
        if isinstance(diagnostic_test_gate_cfg, str):
            diagnostic_test_gate_enabled = diagnostic_test_gate_cfg.strip().lower() not in {"0", "false", "no", "off"}
        else:
            diagnostic_test_gate_enabled = bool(diagnostic_test_gate_cfg)
        section_headers = [
            "CONCLUSIONS AND CLINICAL RELEVANCE",
            "DATA EXTRACTION AND SYNTHESIS",
            "CONCLUSIONS AND RELEVANCE",
            "DATA COLLECTION AND ANALYSIS",
            "MAIN OUTCOMES AND MEASURES",
            "MAIN OUTCOME AND MEASURES",
            "MAIN OUTCOME AND MEASURE",
            "MATERIALS AND METHODS",
            "BACKGROUND AND OBJECTIVE",
            "PURPOSE OF REVIEW",
            "SELECTION CRITERIA",
            "TRIAL REGISTRATION",
            "RECENT FINDINGS",
            "STUDY SELECTION",
            "SEARCH METHODS",
            "MATERIAL AND METHOD",
            "DATA EXTRACTION",
            "MAIN RESULTS",
            "STUDY DESIGN",
            "DATA SOURCES",
            "INTERPRETATION",
            "INTERVENTIONS",
            "CONCLUSIONS",
            "LIMITATIONS",
            "OBJECTIVES",
            "CONCLUSION",
            "IMPORTANCE",
            "BACKGROUND",
            "OBJECTIVE",
            "RATIONALE",
            "UNLABELLED",
            "EXPOSURES",
            "FINDINGS",
            "SUMMARY",
            "METHODS",
            "RESULTS",
            "FUNDING",
            "DESIGN",
            "AIMS",
            "AIM",
        ]
        section_re = re.compile(
            r"(^|(?<=[.!?])\s+)("
            + "|".join(re.escape(h) for h in section_headers)
            + r")\s*:",
            re.I,
        )

        if torch is None:
            raise RuntimeError("torch/transformers not available for stance.")

        tokenizer, model = _load_model(model_name=model_name, device=device, use_fp16=use_fp16)
        ent_i, neu_i, con_i = _infer_nli_indices(model)

        for stmt in state.statements:
            claim = (stmt.text or "").strip()
            if not claim or not getattr(stmt, "evidence", None):
                continue

            # Select candidates (optional)
            ev_indices = list(range(len(stmt.evidence)))
            if isinstance(top_m, int) and top_m > 0 and len(ev_indices) > top_m:
                ev_indices.sort(
                    key=lambda i: float(getattr(stmt.evidence[i], "relevance", 0.0) or 0.0),
                    reverse=True,
                )
                ev_indices = ev_indices[:top_m]

            premises: List[str] = []
            hypotheses: List[str] = []
            mapping: List[Tuple[int, str, float]] = []

            # Reset per-run fields for selected evidence (nested stance model)
            for i in ev_indices:
                ev = stmt.evidence[i]

                if getattr(ev, "stance", None) is None:
                    ev.stance = EvidenceStance()
                else:
                    # Ensure it's the expected shape; if user loads from dicts, pydantic should coerce
                    ev.stance = ev.stance  # no-op, explicit for clarity

                # Reset only the fields we may overwrite this run
                ev.stance.abstract_label = None
                ev.stance.abstract_p_supports = None
                ev.stance.abstract_p_refutes = None
                ev.stance.abstract_p_neutral = None

                title = (
                    getattr(ev, "title", None)
                    or getattr(ev, "article_title", None)
                    or getattr(ev, "paper_title", None)
                )
                title = str(title).strip() if title else ""

                statement_token_count = len(tokenizer.encode(claim, add_special_tokens=False))
                chunk_size = max_length - (statement_token_count + 16)
                if chunk_size < 1:
                    chunk_size = 1
                chunk_overlap = 64
                if chunk_overlap >= chunk_size:
                    chunk_overlap = max(0, chunk_size - 1)
                chunk_step = max(1, chunk_size - chunk_overlap)
                title_added = False

                for field in evidence_fields:
                    txt = getattr(ev, field, None)
                    if txt:
                        txt = " ".join(str(txt).split())

                    if not txt:
                        continue

                    if title and not title_added:
                        premises.append(title)
                        hypotheses.append(claim)
                        mapping.append((i, "title", 0.5))
                        title_added = True

                    segments: List[Tuple[str, str]] = []
                    section_matches = list(section_re.finditer(txt)) if section_chunking_enabled else []
                    if section_matches:
                        if section_matches[0].start() > 0:
                            preamble = txt[: section_matches[0].start()].strip()
                            if preamble:
                                segments.append(("UNKNOWN", preamble))
                        for idx, match in enumerate(section_matches):
                            label = match.group(2).upper()
                            start_text = match.end()
                            end_text = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(txt)
                            section_text = txt[start_text:end_text].strip()
                            if section_text:
                                segments.append((label, section_text))
                    else:
                        segments.append(("UNKNOWN", txt))

                    for section_label, section_text in segments:
                        token_ids = tokenizer.encode(section_text, add_special_tokens=False)
                        if not token_ids:
                            continue

                        start = 0
                        while start < len(token_ids):
                            chunk_ids = token_ids[start : start + chunk_size]
                            chunk_text = tokenizer.decode(
                                chunk_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            ).strip()
                            if chunk_text:
                                premises.append(chunk_text)
                                hypotheses.append(claim)
                                mapping.append((i, field, 1.0))
                            if start + chunk_size >= len(token_ids):
                                break
                            start += chunk_step

            if not premises:
                continue

            all_probs: List[List[float]] = []
            with torch.no_grad():
                for prem_chunk, hyp_chunk in zip(_batch(premises, batch_size), _batch(hypotheses, batch_size)):
                    inputs = tokenizer(
                        prem_chunk,
                        hyp_chunk,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    self.add_step_tokens(_count_batch_tokens(inputs))
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logits = model(**inputs, return_dict=True).logits.float()
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.extend(probs.detach().cpu().tolist())

            results_by_evidence: Dict[int, List[Tuple[float, float, float, float, StanceLabel]]] = {}

            for probs, (ev_idx, field, weight) in zip(all_probs, mapping):
                p_ent = float(probs[ent_i])  # entailment -> Supports
                p_neu = float(probs[neu_i])  # neutral    -> Neutral
                p_con = float(probs[con_i])  # contradiction -> Refutes

                if p_ent >= 0.70 and p_ent - max(p_con, p_neu) >= 0.15:
                    label = StanceLabel.SUPPORTS
                elif p_con >= 0.60 and p_con - max(p_ent, p_neu) >= 0.15:
                    label = StanceLabel.REFUTES
                else:
                    label = StanceLabel.NEUTRAL

                if field == "title" or field == "abstract" or field == "text":
                    results_by_evidence.setdefault(ev_idx, []).append((p_ent, p_con, p_neu, weight, label))

            # Write aggregated results back into Evidence.stance
            for ev_idx, candidates in results_by_evidence.items():
                if not candidates:
                    continue

                total_weight = sum(weight for p_ent, p_con, p_neu, weight, label in candidates)
                if total_weight <= 0:
                    total_weight = float(len(candidates))

                p_support = sum(
                    p_ent * weight for p_ent, p_con, p_neu, weight, label in candidates
                ) / total_weight
                p_refute = sum(
                    p_con * weight for p_ent, p_con, p_neu, weight, label in candidates
                ) / total_weight
                p_neutral = sum(
                    p_neu * weight for p_ent, p_con, p_neu, weight, label in candidates
                ) / total_weight

                prob_total = p_support + p_refute + p_neutral
                if prob_total > 0:
                    p_support /= prob_total
                    p_refute /= prob_total
                    p_neutral /= prob_total

                if p_support >= 0.70 and p_support - max(p_refute, p_neutral) >= 0.15:
                    overall_label = StanceLabel.SUPPORTS
                elif p_refute >= 0.60 and p_refute - max(p_support, p_neutral) >= 0.15:
                    overall_label = StanceLabel.REFUTES
                else:
                    overall_label = StanceLabel.NEUTRAL

                ev = stmt.evidence[ev_idx]
                if ev.stance is None:
                    ev.stance = EvidenceStance()

                if diagnostic_test_gate_enabled and overall_label == StanceLabel.SUPPORTS:
                    title = (
                        getattr(ev, "title", None)
                        or getattr(ev, "article_title", None)
                        or getattr(ev, "paper_title", None)
                        or ""
                    )
                    evidence_text = " ".join(
                        str(part or "")
                        for part in [title, getattr(ev, "abstract", None), getattr(ev, "text", None)]
                    )
                    is_diagnostic_claim = bool(
                        _DIAGNOSTIC_CLAIM_RE.search(claim)
                        or _PROCEDURE_SKIN_CLAIM_RE.search(claim)
                    )
                    has_diagnostic_uncertainty = bool(_DIAGNOSTIC_UNCERTAINTY_RE.search(evidence_text))
                    has_method_mismatch = bool(
                        _PROCEDURE_SKIN_CLAIM_RE.search(claim)
                        and _FORMAL_DIAGNOSTIC_TEST_RE.search(evidence_text)
                    )
                    if is_diagnostic_claim and (has_diagnostic_uncertainty or has_method_mismatch):
                        overall_label = StanceLabel.NEUTRAL
                        p_neutral = max(p_neutral, p_support)
                        p_support = min(p_support, 0.49)
                        prob_total = p_support + p_refute + p_neutral
                        if prob_total > 0:
                            p_support /= prob_total
                            p_refute /= prob_total
                            p_neutral /= prob_total

                rounded_probs = [
                    round(float(p_support), 2),
                    round(float(p_refute), 2),
                    round(float(p_neutral), 2),
                ]
                round_delta = round(1.0 - sum(rounded_probs), 2)
                if round_delta:
                    strongest_idx = max(range(3), key=lambda idx: rounded_probs[idx])
                    rounded_probs[strongest_idx] = round(
                        rounded_probs[strongest_idx] + round_delta,
                        2,
                    )

                ev.stance.abstract_label = overall_label
                ev.stance.abstract_p_supports = rounded_probs[0]
                ev.stance.abstract_p_refutes = rounded_probs[1]
                ev.stance.abstract_p_neutral = rounded_probs[2]

        return state
