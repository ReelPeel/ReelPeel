"""
Step: Compute stance (NLI) for evidence relative to each claim.

This step runs a natural language inference model to decide whether each piece
of evidence supports, refutes, or is neutral toward the claim. It writes the
probabilities and label into the nested Evidence.stance model.

Inputs:
- state.statements with Statement.text and stmt.evidence populated.
- Evidence fields used: abstract and/or text (configurable).
- Optional evidence titles are prefixed to the passage when available.

Config keys:
- model_name: default "cnut1648/biolinkbert-mednli".
- device: "cuda", "cpu", or explicit device string.
- use_fp16: use fp16 weights on CUDA.
- batch_size, max_length: inference batching and truncation.
- evidence_fields: list of evidence fields to score, typically ["abstract", "text"].
- top_m_by_relevance: if set, only score the top-M evidence by relevance.
- Abstracts are scored in fixed tokenizer chunks with 64-token overlap.
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
- Long inputs are truncated to max_length (no chunking implemented here).

NOTE ON LONG ABSTRACTS
---------------------
This model was trained with max_seq_length=512 tokens. For long abstracts/summaries,
we rely on tokenizer truncation (max_length). This keeps the *leading* portion of the
text and discards the tail. If this becomes an issue, implement chunking (e.g., sliding
window) and aggregate probabilities (max/mean) downstream.

"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..core.base import PipelineStep
from ..core.models import PipelineState, Stance as EvidenceStance, StanceLabel

_MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[Any, Any]] = {}


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

      - Titles are scored once per evidence item and downweighted in aggregation.
      - Abstract/text fields are split into fixed tokenizer chunks with 64-token overlap.
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

                    token_ids = tokenizer.encode(txt, add_special_tokens=False)
                    if not token_ids:
                        continue

                    if title and not title_added:
                        premises.append(title)
                        hypotheses.append(claim)
                        mapping.append((i, "title", 0.5))
                        title_added = True

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

                support_score = max(
                    max(0.0, p_ent - max(p_con, p_neu)) * weight
                    for p_ent, p_con, p_neu, weight, label in candidates
                )
                refute_score = max(
                    max(0.0, p_con - max(p_ent, p_neu)) * weight
                    for p_ent, p_con, p_neu, weight, label in candidates
                )

                if refute_score >= 0.18 and refute_score >= support_score + 0.10:
                    overall_label = StanceLabel.REFUTES
                elif support_score >= 0.25 and support_score >= refute_score + 0.15:
                    overall_label = StanceLabel.SUPPORTS
                else:
                    overall_label = StanceLabel.NEUTRAL

                support_candidates = [
                    p_ent for p_ent, p_con, p_neu, weight, label in candidates
                    if label == StanceLabel.SUPPORTS
                ]
                refute_candidates = [
                    p_con for p_ent, p_con, p_neu, weight, label in candidates
                    if label == StanceLabel.REFUTES
                ]

                p_support = max(support_candidates) if support_candidates else max(
                    p_ent for p_ent, p_con, p_neu, weight, label in candidates
                )
                p_refute = max(refute_candidates) if refute_candidates else max(
                    p_con for p_ent, p_con, p_neu, weight, label in candidates
                )
                p_neutral = max(p_neu for p_ent, p_con, p_neu, weight, label in candidates)

                ev = stmt.evidence[ev_idx]
                if ev.stance is None:
                    ev.stance = EvidenceStance()

                ev.stance.abstract_label = overall_label
                ev.stance.abstract_p_supports = round(float(p_support), 2)
                ev.stance.abstract_p_refutes = round(float(p_refute), 2)
                ev.stance.abstract_p_neutral = round(float(p_neutral), 2)

        return state
