"""
STEP (SCORES): COMPUTE STANCE (NLI) FOR EVIDENCE
------------------------------------------------
- Computes the stance of each evidence item relative to a claim using an NLI model.
- Uses cnut1648/biolinkbert-mednli (BioLinkBERT-large fine-tuned on MedNLI).
- Writes per-field outputs back into Evidence:
    - stance_abstract_label / stance_abstract_p_{supports,refutes,neutral}
    - stance_summary_label  / stance_summary_p_{supports,refutes,neutral}

Label mapping for cnut1648/biolinkbert-mednli:
    id2label = {0: entailment, 1: neutral, 2: contradiction}
We map:
    entailment    -> supports
    contradiction -> refutes
    neutral       -> neutral

NOTE ON LONG ABSTRACTS
---------------------
This model was trained with max_seq_length=512 tokens. For long abstracts/summaries,
we rely on tokenizer truncation (max_length). This keeps the *leading* portion of the
text and discards the tail. If this becomes an issue, implement chunking (e.g., sliding
window) and aggregate probabilities (max/mean) downstream.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

from ..core.base import PipelineStep
from ..core.models import PipelineState

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[Any, Any]] = {}


def _pick_device(device_cfg: Optional[str]) -> str:
    # Mirrors rerank.py behavior
    if device_cfg:
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


def _infer_nli_indices(model) -> Tuple[int, int, int]:
    """
    Returns: (entailment_idx, neutral_idx, contradiction_idx)

    Uses model.config.id2label when available. Falls back to common conventions.
    """
    id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    norm = {}
    for k, v in id2label.items():
        try:
            i = int(k)
        except Exception:
            i = k
        norm[i] = str(v).strip().lower()

    # Try semantic match first
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

    # Fallbacks (best-effort): 3-class NLI is almost always in 0..2
    # Common training scripts for MedNLI/MNLI use:
    #   0=entailment, 1=neutral, 2=contradiction  (as on the HF card for this model)
    return 0, 1, 2


class StanceEvidenceStep(PipelineStep):
    """
    Config keys:
      - model_name: str (default: "cnut1648/biolinkbert-mednli")
      - device: "cuda", "cuda:0", "cpu", "auto" (default: auto)
      - use_fp16: bool (default: True)
      - batch_size: int (default: 16)
      - max_length: int (default: 512)

      - evidence_fields: list[str] (default: ["abstract","summary"])
          Which Evidence fields to compute stance on.

      - top_m_by_relevance: int | None (default: None)
          If set, only compute stance for the Top-M evidence items per statement,
          selected by ev.relevance (descending). Others are left as-is.

      - threshold_decisive: float (default: 0.0)
          If max(p_supports, p_refutes) < threshold, force label to "neutral".
    """

    def execute(self, state: PipelineState) -> PipelineState:
        model_name = self.config.get("model_name", "cnut1648/biolinkbert-mednli")
        device = _pick_device(self.config.get("device"))
        use_fp16 = bool(self.config.get("use_fp16", True))
        batch_size = int(self.config.get("batch_size", 16))
        max_length = int(self.config.get("max_length", 512))

        evidence_fields = self.config.get("evidence_fields", ["abstract", "summary"])
        top_m = self.config.get("top_m_by_relevance", None)
        threshold_decisive = float(self.config.get("threshold_decisive", 0.0))

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
            mapping: List[Tuple[int, str]] = []

            # Reset per-run fields for selected evidence
            for i in ev_indices:
                ev = stmt.evidence[i]
                if hasattr(ev, "stance_abstract_label"):
                    ev.stance_abstract_label = None
                    ev.stance_abstract_p_supports = None
                    ev.stance_abstract_p_refutes = None
                    ev.stance_abstract_p_neutral = None
                if hasattr(ev, "stance_summary_label"):
                    ev.stance_summary_label = None
                    ev.stance_summary_p_supports = None
                    ev.stance_summary_p_refutes = None
                    ev.stance_summary_p_neutral = None

                for field in evidence_fields:
                    txt = getattr(ev, field, None)
                    title = (
                        getattr(ev, "title", None)
                        or getattr(ev, "article_title", None)
                        or getattr(ev, "paper_title", None)
                    )
                    if title:
                        title = str(title).strip()
                        txt = f"{title}\n\n{txt}"
                    if txt:
                        txt = str(txt).strip()
                    if not txt:
                        continue

                    premises.append(txt)
                    hypotheses.append(claim)
                    mapping.append((i, field))

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
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logits = model(**inputs, return_dict=True).logits.float()
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.extend(probs.detach().cpu().tolist())

            # Write results back
            for probs, (ev_idx, field) in zip(all_probs, mapping):
                p_ent = float(probs[ent_i])
                p_neu = float(probs[neu_i])
                p_con = float(probs[con_i])

                # Decide label
                if max(p_ent, p_con) < threshold_decisive:
                    label = "neutral"
                else:
                    label = "supports" if p_ent >= p_con else "refutes"

                ev = stmt.evidence[ev_idx]

                if field == "abstract":
                    ev.stance_abstract_label = label
                    ev.stance_abstract_p_supports = p_ent
                    ev.stance_abstract_p_refutes = p_con
                    ev.stance_abstract_p_neutral = p_neu
                elif field == "summary":
                    ev.stance_summary_label = label
                    ev.stance_summary_p_supports = p_ent
                    ev.stance_summary_p_refutes = p_con
                    ev.stance_summary_p_neutral = p_neu
                else:
                    # If you add more fields, extend here.
                    pass

        return state
