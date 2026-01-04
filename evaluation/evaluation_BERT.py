#!/usr/bin/env python3
"""
Train DeBERTa (3-class) on train_txt and evaluate on test_txt.

Input .txt formats supported:
  A) JSON list:         [ {...}, {...} ]
  B) JSONL:             {"id":..., "statement":..., "label":...}\n...
  C) Python-literal:    [{'id':..., 'statement':..., 'label':...}, ...]

Outputs (written to --output_dir):
  - BERT_pipeline.json       (per-test-example predictions)
  - Bert_eval_results.json   (accuracy, per-class recall, confusion matrix)

Labels:
  0 = true, 1 = uncertain, 2 = false
"""

import argparse
import ast
import inspect
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

VALID_LABELS = {0, 1, 2}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_parse_jsonl(text: str) -> Optional[List[Dict[str, Any]]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    out = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        out.append(obj)
    return out


def _try_parse_pythonish(text: str) -> Optional[Any]:
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def load_records_from_text_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Input file is empty: {path}")

    obj = _try_parse_json(text)
    if obj is None:
        obj = _try_parse_jsonl(text)
    if obj is None:
        obj = _try_parse_pythonish(text)

    if obj is None:
        raise ValueError(
            f"Could not parse {path}. Provide JSON list, JSONL, or Python-literal list."
        )

    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        obj = obj["data"]

    if not isinstance(obj, list):
        raise ValueError(f"Parsed content is not a list in: {path}")

    for i, r in enumerate(obj):
        if not isinstance(r, dict):
            raise ValueError(f"{path}: record {i} is not a dict.")
        if "statement" not in r:
            raise ValueError(f"{path}: record {i} missing 'statement'.")
        if "id" not in r:
            raise ValueError(f"{path}: record {i} missing 'id'.")
        # label may be missing; such rows are skipped in scoring.
    return obj


def build_dataset(records: List[Dict[str, Any]]) -> Dataset:
    ds = Dataset.from_list(records)
    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    else:
        ds = ds.add_column("labels", [None] * len(ds))
    return ds


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def tok(batch):
        return tokenizer(batch["statement"], truncation=True, max_length=max_length)

    return ds.map(tok, batched=True)


def compute_metrics_from_arrays(
    y_true: List[int], y_pred: List[int], n_total: int
) -> Dict[str, Any]:
    n_scored = len(y_true)
    n_skipped = n_total - n_scored

    if n_scored == 0:
        return {
            "n_total_in_file": int(n_total),
            "n_scored": 0,
            "n_skipped": int(n_skipped),
            "accuracy": None,
            "per_class_recall": {str(c): None for c in sorted(VALID_LABELS)},
            "confusion_matrix": {},
        }

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    accuracy = float((y_true_arr == y_pred_arr).mean())

    per_class_recall: Dict[str, Any] = {}
    for cls in sorted(VALID_LABELS):
        mask = (y_true_arr == cls)
        denom = int(mask.sum())
        per_class_recall[str(cls)] = None if denom == 0 else float(((y_pred_arr == cls) & mask).sum() / denom)

    confusion: Dict[str, Dict[str, int]] = {}
    for t, p in zip(y_true_arr.tolist(), y_pred_arr.tolist()):
        ts, ps = str(t), str(p)
        confusion.setdefault(ts, {})
        confusion[ts][ps] = confusion[ts].get(ps, 0) + 1

    return {
        "n_total_in_file": int(n_total),
        "n_scored": int(n_scored),
        "n_skipped": int(n_skipped),
        "accuracy": accuracy,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion,
    }


def make_trainer(
    model,
    training_args,
    tokenizer,
    data_collator,
    train_ds: Optional[Dataset] = None,
    eval_ds: Optional[Dataset] = None,
) -> Trainer:
    # Transformers is deprecating `tokenizer=` in Trainer; keep compatibility with both signatures.
    kwargs = dict(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_txt", type=str, required=True, help="Training file (.txt).")
    parser.add_argument("--test_txt", type=str, required=True, help="Test file (.txt) for accuracy reporting.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for JSON results/checkpoints.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/deberta-v3-base",
        help="HF model name or local checkpoint path.",
    )

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    # Optional internal validation split from TRAIN ONLY (not the test set)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio from train set.")
    parser.add_argument("--no_eval_during_train", action="store_true", help="Disable eval during training.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Load train/test
    train_records = load_records_from_text_file(args.train_txt)
    test_records = load_records_from_text_file(args.test_txt)

    train_ds = build_dataset(train_records)
    test_ds = build_dataset(test_records)

    # Tokenizer: use_fast=False avoids the SentencePiece byte-fallback fast-tokenizer warning and keeps behavior stable.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    train_tok = tokenize_dataset(train_ds, tokenizer, args.max_length)
    test_tok = tokenize_dataset(test_ds, tokenizer, args.max_length)

    # Model: 3-class head; use_safetensors=True avoids the .bin loading path.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=3,
        use_safetensors=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Keep only valid-labeled rows for training/eval split
    train_valid_idx = [
        i for i, lbl in enumerate(train_tok["labels"])
        if isinstance(lbl, (int, np.integer)) and int(lbl) in VALID_LABELS
    ]
    if len(train_valid_idx) < 10:
        raise ValueError("Not enough valid-labeled rows in train file to train a classifier head.")

    train_tok_valid = train_tok.select(train_valid_idx)

    eval_ds = None
    if not args.no_eval_during_train and args.val_ratio > 0:
        split = train_tok_valid.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_tok_use = split["train"]
        eval_ds = split["test"]
    else:
        train_tok_use = train_tok_valid

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "deberta_runs"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=25,
        report_to="none",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        # if eval_ds is None, Trainer will not evaluate; this remains safe
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=2,
        load_best_model_at_end=(eval_ds is not None),
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
    )

    trainer = make_trainer(
        model=model,
        training_args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_ds=train_tok_use,
        eval_ds=eval_ds,
    )

    # Train
    trainer.train()

    # Predict on TEST
    pred_out = trainer.predict(test_tok)
    logits = pred_out.predictions
    y_pred_all = np.argmax(logits, axis=1).astype(int).tolist()

    # Write BERT_pipeline.json (TEST SET)
    pipeline_rows: List[Dict[str, Any]] = []
    y_true_scored: List[int] = []
    y_pred_scored: List[int] = []

    true_labels = test_tok["labels"]
    ids = test_tok["id"]

    for i in range(len(test_records)):
        true_lbl = true_labels[i]
        pred_lbl = int(y_pred_all[i])

        pipeline_rows.append(
            {"id": ids[i], "true_label": true_lbl, "predicted_label": pred_lbl}
        )

        if isinstance(true_lbl, (int, np.integer)) and int(true_lbl) in VALID_LABELS:
            y_true_scored.append(int(true_lbl))
            y_pred_scored.append(pred_lbl)

    pipeline_path = os.path.join(args.output_dir, "BERT_pipeline.json")
    with open(pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_rows, f, ensure_ascii=False, indent=2)

    # Write Bert_eval_results.json (TEST SET)
    eval_results = compute_metrics_from_arrays(
        y_true=y_true_scored,
        y_pred=y_pred_scored,
        n_total=len(test_records),
    )
    eval_path = os.path.join(args.output_dir, "Bert_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {pipeline_path}")
    print(f"Wrote: {eval_path}")
    print(f"Test accuracy: {eval_results['accuracy']}")


if __name__ == "__main__":
    main()
