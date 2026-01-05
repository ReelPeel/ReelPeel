"""
STEP (SCORES): RERANK EVIDENCE WITH BGE RERANKER v2 m3
-----------------------------------------------------
- Scores each evidence item for *relevance to the claim* using BAAI/bge-reranker-v2-m3.
- Computes ev.relevance_abstract (claim vs abstract).
- Also writes ev.relevance as a combined score for downstream sorting.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..core.base import PipelineStep
from ..core.models import PipelineState

_MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[Any, Any]] = {}


def _pick_device(device_cfg: Optional[str]) -> str:
    # keep your existing behavior; if you want least-used GPU, use device="auto" and your autopick logic
    if device_cfg:
        return device_cfg
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model_name: str, device: str, use_fp16: bool):
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("Missing dependencies for reranking. Install: pip install torch transformers")

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


def _combine_scores(a: Optional[float], s: Optional[float], strategy: str) -> Optional[float]:
    vals = [v for v in [a, s] if v is not None]
    if not vals:
        return None

    strategy = (strategy or "max").lower()
    if strategy == "max":
        return max(vals)
    if strategy == "mean":
        return sum(vals) / len(vals)
    if strategy == "abstract_first":
        return a if a is not None else s
    if strategy == "summary_first":
        return s if s is not None else a

    # default fallback
    return max(vals)


class RerankEvidenceStep(PipelineStep):
    """
    Config keys:
      - model_name: str (default: "BAAI/bge-reranker-v2-m3")
      - device: "cuda", "cuda:0", "cpu", "auto" (default: auto)
      - use_fp16: bool (default: True)
      - normalize: bool (default: True) -> sigmoid to [0,1]
      - batch_size: int (default: 16)
      - max_length: int (default: 512)

      - score_fields: list[str] (default: ["abstract"])
          Which evidence fields to score individually.

      - combine_strategy: str (default: "max")
          How to compute ev.relevance from the per-field scores:
            - "max": max(abstract, summary)
            - "mean": mean of available
            - "abstract_first": abstract if exists else summary
            - "summary_first": summary if exists else abstract

      - empty_relevance: float (default: 0.0)
    """

    def execute(self, state: PipelineState) -> PipelineState:
        model_name = self.config.get("model_name", "BAAI/bge-reranker-v2-m3")
        device = _pick_device(self.config.get("device"))
        use_fp16 = bool(self.config.get("use_fp16", True))
        normalize = bool(self.config.get("normalize", True))
        batch_size = int(self.config.get("batch_size", 16))
        max_length = int(self.config.get("max_length", 512))

        score_fields = self.config.get("score_fields", ["abstract"])
        combine_strategy = self.config.get("combine_strategy", "max")
        empty_relevance = float(self.config.get("empty_relevance", 0.0))

        if torch is None:
            raise RuntimeError("torch/transformers not available for reranking.")

        tokenizer, model = _load_model(model_name=model_name, device=device, use_fp16=use_fp16)


        for stmt in state.statements:
            claim = (stmt.text or "").strip()
            if not claim or not getattr(stmt, "evidence", None):
                continue

            # Build (claim, passage) pairs for *each field*, and map back to (ev_idx, field)
            pairs: List[List[str]] = []
            mapping: List[Tuple[int, str]] = []

            for i, ev in enumerate(stmt.evidence):
                # reset per-field relevance each run (optional)
                if hasattr(ev, "relevance_abstract"):
                    ev.relevance_abstract = None
                if hasattr(ev, "relevance_summary"):
                    ev.relevance_summary = None

                for field in score_fields:
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

                    pairs.append([claim, txt])
                    mapping.append((i, field))

                # if neither field exists, set combined relevance fallback now
                if not any(getattr(ev, f, None) for f in score_fields):
                    ev.relevance = float(empty_relevance)

            if not pairs:
                continue

            # Score in batches
            all_scores: List[float] = []
            with torch.no_grad():
                for chunk in _batch(pairs, batch_size=batch_size):
                    inputs = tokenizer(
                        chunk,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=max_length,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    logits = model(**inputs, return_dict=True).logits.view(-1).float()
                    scores = torch.sigmoid(logits) if normalize else logits
                    all_scores.extend(scores.detach().cpu().tolist())

            # Write per-field scores back
            for score, (ev_idx, field) in zip(all_scores, mapping):
                ev = stmt.evidence[ev_idx]
                fs = round(float(score), 2)

                if field == "abstract" and hasattr(ev, "relevance_abstract"):
                    ev.relevance_abstract = fs
                elif field == "summary" and hasattr(ev, "relevance_summary"):
                    ev.relevance_summary = fs
                else:
                    # If you ever add more fields, you can extend mapping logic here.
                    # For now, silently ignore unknown field targets.
                    pass

            # Compute combined relevance for each evidence
            for ev in stmt.evidence:
                a = getattr(ev, "relevance_abstract", None)
                s = getattr(ev, "relevance_summary", None)
                combined = _combine_scores(a, s, combine_strategy)

                if combined is None:
                    # fallback if nothing was scored
                    combined = float(ev.relevance) if ev.relevance is not None else float(empty_relevance)

                ev.relevance = float(combined)

        return state
