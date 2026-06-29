# Technical Notes: Stance Step

## Current Issue

The current stance step uses `cnut1648/biolinkbert-mednli` on the full abstract
against the claim. This is fast, but it can misclassify topical similarity as
`Supports`, especially when the abstract is only indirectly related or when the
relevant conclusion appears late in the abstract.

Raising `max_length` to 8k/16k is not a useful fix: BioLinkBERT/BERT-style models
are effectively limited to about 512 tokens. Longer abstracts should be chunked
instead.

## Findings From Offline Mocks

The stored test runs were inspected from `offline_mock/**/process.json`.

- 22 process files inspected.
- 89 statement instances, 23 unique statement texts.
- Statements are short: about 9-19 rough tokens, median about 12.
- 228 evidence abstract instances, 77 unique abstracts.
- Titles are moderate: about 3-28 rough tokens, median about 13.
- Abstracts are mixed length: median about 318 rough tokens, p90 about 451,
  p95 about 566, max about 1309.

This means most statement-title-abstract pairs fit near the 512-token model
limit, but long reviews/systematic reviews will be truncated today. Chunking
mainly protects the long-tail cases and late abstract conclusions.

## Proposed Deterministic Approach

No additional LLM step is needed.

Pipeline:

1. Normalize whitespace.
2. Classify the title against the statement once, if a title exists.
3. Tokenize the abstract with the same tokenizer used by the stance model.
4. Build fixed-length token chunks with overlap.
5. Run BioLinkBERT NLI per chunk.
6. Aggregate title and chunk scores into one overall stance per evidence item.

Recommended chunking:

- Keep `max_length=512`.
- Compute `statement_token_count` with the BioLinkBERT tokenizer.
- Use `chunk_size = 512 - (statement_token_count + 16)`.
- Tokenize the abstract without special tokens.
- Use fixed token chunks with `64` token overlap.
- Do not use section labels.
- Do not use sentence splitting.

Rationale:

- The offline statements are short, so the dynamic chunk size is usually large.
- A `64` token overlap preserves local scientific context across boundaries.
- The p90 abstract length is about 451 rough tokens, so overlap is useful but
  should not be too large.
- Fixed token windows are deterministic and avoid fragile sentence regex logic.

## Title Handling

Classify the title separately against the statement exactly once.

Reasoning:

- Titles are short enough in the offline data to score separately.
- Titles often contain the population/intervention/outcome and help topical
  grounding.
- The title is weaker evidence than the abstract and gets signal weight `0.5`
  during aggregation.
- The title is not part of the abstract chunks and not part of the overlap.

## Label Decision

Avoid simple majority voting. Medical stance should be conservative.

Per chunk:

- `Supports` only if support probability is high and clearly above both
  refute and neutral.
- `Refutes` can use a slightly lower threshold than support.
- Otherwise label as `Neutral`.

Suggested thresholds:

- `Supports`: `p_support >= 0.70` and margin `>= 0.15`
- `Refutes`: `p_refute >= 0.60` and margin `>= 0.15`
- Else: `Neutral`

Overall evidence stance should aggregate weighted confidence, not label counts.
Support should require a stronger signal than refutation, because falsely
endorsing medical claims is the higher-risk error.

Current overall decision:

- Compute per candidate:
  - `support_signal = max(0, p_support - max(p_refute, p_neutral))`
  - `refute_signal = max(0, p_refute - max(p_support, p_neutral))`
- Weight title signals with `0.5`.
- Weight abstract chunk signals with `1.0`.
- Aggregate via max weighted signal.
- `Refutes` if `refute_score >= 0.18` and
  `refute_score >= support_score + 0.10`.
- `Supports` if `support_score >= 0.25` and
  `support_score >= refute_score + 0.15`.
- Else `Neutral`.

Only the existing overall fields are stored:

```json
{
  "abstract_label": "Supports|Refutes|Neutral",
  "abstract_p_supports": 0.72,
  "abstract_p_refutes": 0.08,
  "abstract_p_neutral": 0.20
}
```

## Offline Evaluation After Chunking

Tested with `factchecker_t26` on `offline_mock/**/process.json`.

Aggregate over 22 process files / 228 evidence items:

- `Supports -> Supports`: 185
- `Refutes -> Refutes`: 35
- `Supports -> Refutes`: 4
- `Supports -> Neutral`: 2
- `Refutes -> Supports`: 2

Comparison against the available evidence summaries in
`offline_mock/DT0UIgzDZ79/evidence_summaries`:

- 10 evidence summaries checked.
- 1 of 10 stance labels changed.
- New counts: 8 `Supports`, 2 `Refutes`, 0 `Neutral`.
- Several obvious summary/stance conflicts remain, especially for indirect
  diagnostic evidence such as egg allergy skin testing.

Conclusion: fixed-token chunking reduces truncation risk but does not by itself
solve the main stance problem. The remaining issue is mostly model semantics:
BioLinkBERT-MedNLI often treats topical or indirect biomedical evidence as a
directional stance.
