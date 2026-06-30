# Technical Notes: Stance Step

## Current Best Approach

Current best approach for the live system:

1. Keep BioLinkBERT-MedNLI for the real-time stance step.
2. Keep `max_length=512`; do not try 8k/16k context on BioLinkBERT.
3. Use section-aware chunking:
   - If strict abstract section headers are detected, use each section as its
     own stance candidate.
   - If a section is longer than the available model budget, split only that
     section with 64-token overlap.
   - If no section headers are detected, fall back to fixed token chunks over
     the whole abstract.
4. Score the title once against the statement, but treat it as weak evidence
   with weight `0.5`.
5. Aggregate candidate probabilities with a simple weighted mean:
   - title candidate weight `0.5`
   - every abstract/section chunk weight `1.0`
   - no section-specific weighting
   - normalize the stored `abstract_p_*` values so they sum to `1.00`
6. Apply the diagnostic/test gate by default:
   - diagnostic/test claims with method-mismatch or diagnostic-uncertainty cues
     cannot remain `Supports`; they are downgraded to `Neutral`.
7. Always pass the same `evidence.stance` object to `/evidence_summary` that is
   displayed in the UI.

The biggest practical improvement came from step 6, not from changing the
stance aggregation. The old browser summary route did not send
`evidence.stance`, so the summary prompt often received `Unknown` stance and
re-derived the paper direction from the abstract. That made the displayed
stance and generated summary diverge.

## Original Issue

The original stance step used `cnut1648/biolinkbert-mednli` on the full
abstract against the claim. This was fast, but it could misclassify topical
similarity as `Supports`, especially when the abstract was only indirectly
related or when the relevant conclusion appeared late in the abstract.

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

## Deterministic Stance Approach

No additional LLM step is needed.

Pipeline:

1. Normalize whitespace.
2. Classify the title against the statement once, if a title exists.
3. Detect strict abstract section headers with a whitelist regex.
4. If sections exist, use each section as an individual chunk candidate.
5. If no sections exist, tokenize the full abstract with the same tokenizer used
   by the stance model and build fixed-length token chunks with overlap.
6. Run BioLinkBERT NLI per title/chunk candidate.
7. Aggregate title and chunk scores into one overall stance per evidence item.

Recommended chunking:

- Keep `max_length=512`.
- Compute `statement_token_count` with the BioLinkBERT tokenizer.
- Use `chunk_size = 512 - (statement_token_count + 16)`.
- Tokenize abstracts/sections without special tokens.
- Use `64` token overlap when splitting long text.
- Use sections when strict headers are present.
- Do not use sentence splitting.

Rationale:

- The offline statements are short, so the dynamic chunk size is usually large.
- A `64` token overlap preserves local scientific context across boundaries.
- The p90 abstract length is about 451 rough tokens, so overlap is useful but
  should not be too large.
- Section-aware chunks reduce some false support by keeping result/conclusion
  blocks separate from background/method context.
- Fixed token fallback is deterministic and avoids fragile sentence regex logic.

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

Overall evidence stance now aggregates the candidate probabilities directly.
This intentionally favors a simple, explainable output distribution over the
previous max-signal aggregation.

Current overall decision:

- For every title/chunk candidate, keep the raw BioLinkBERT probabilities:
  `p_support`, `p_refute`, `p_neutral`.
- Weight title candidates with `0.5`.
- Weight every abstract/section chunk with `1.0`.
- Compute weighted means across candidates for all three classes.
- Normalize the three means so they sum to `1.0`.
- Decide the final label from the normalized aggregate:
  - `Supports`: `p_support >= 0.70` and margin `>= 0.15`
  - `Refutes`: `p_refute >= 0.60` and margin `>= 0.15`
  - Else: `Neutral`
- Store the normalized probabilities rounded to 2 decimals, with rounding
  adjusted so the stored values sum to `1.00`.

Only the existing overall fields are stored:

```json
{
  "abstract_label": "Supports|Refutes|Neutral",
  "abstract_p_supports": 0.72,
  "abstract_p_refutes": 0.08,
  "abstract_p_neutral": 0.20
}
```

## Stance Probability Problem And Fix

Previous stored stance probabilities were schema-compatible but semantically
misleading in multi-candidate stance runs.

The stance step scores multiple candidates per evidence item:

- optional title candidate
- section chunks, if section headers are detected
- fixed-token abstract chunks as fallback

The previous export stored independent maxima:

- `abstract_p_supports`: highest support probability from support candidates,
  otherwise highest raw support probability
- `abstract_p_refutes`: highest refute probability from refute candidates,
  otherwise highest raw refute probability
- `abstract_p_neutral`: highest raw neutral probability

These values could come from different candidates. Therefore they were not a
normalized probability distribution and should not have been interpreted as one
model output.

Observed problem in the current offline mock:

```text
Statement:
Eggs are good for brain development due to their high protein content.

Evidence:
[Egg components involved in cognitive function].
PMID: 39279756
Relevance: 0.76
```

Candidate-level behavior:

| Candidate | Label | Supports | Refutes | Neutral |
| --- | ---: | ---: | ---: | ---: |
| Title | Neutral | 0.0056 | 0.0003 | 0.9941 |
| Abstract chunk | Supports | 0.9993 | 0.0003 | 0.0005 |

Previous stored output became:

```json
{
  "abstract_label": "Supports",
  "abstract_p_supports": 1.0,
  "abstract_p_refutes": 0.0,
  "abstract_p_neutral": 0.99
}
```

This looks contradictory because `Supports` and `Neutral` are both near 1.0,
but they come from different candidates. The label is driven by the abstract
chunk; the high neutral value comes from the title.

Testing "no title in stance" reduced this specific title-driven artifact but
did not solve the general issue:

| Offline file | Current labels | No-title labels | Label changes | High support + high neutral cases |
| --- | --- | --- | ---: | ---: |
| `DT0UIgzDZ79/process.json` | 8 Supports, 1 Refutes, 2 Neutral | same | 0 | 3 -> 2 |
| `DT0UbkjDZZj/process.json` | 5 Supports, 1 Refutes, 2 Neutral | same | 0 | 3 -> 3 |

Implemented fix:

- Keep title candidates, but only with weight `0.5`.
- Keep all abstract/section chunks with weight `1.0`.
- Store weighted mean probabilities instead of independent maxima.
- Normalize and round stored probabilities so
  `abstract_p_supports + abstract_p_refutes + abstract_p_neutral == 1.00`.

Current result for PMID `39279756` after the fix:

```json
{
  "abstract_label": "Neutral",
  "abstract_p_supports": 0.67,
  "abstract_p_refutes": 0.0,
  "abstract_p_neutral": 0.33
}
```

This is less assertive than the previous `Supports` label because the neutral
title candidate now contributes to the weighted mean instead of being stored as
an independent max value.

## Diagnostic/Test Gate

This gate is active by default in the live stance step:

```python
diagnostic_test_gate_enabled = True
```

It is also set explicitly in the current video/audio pipeline configs. The gate
is a conservative post-processing rule after BioLinkBERT aggregation.

Why it exists:

- BioLinkBERT-MedNLI can map topical diagnostic similarity to `Supports`.
- The critical example is the egg-skin claim:
  `Applying a small amount of egg to a child's skin can test for egg allergy.`
- Papers about formal allergy diagnosis, skin prick tests, patch tests, sIgE,
  or oral food challenge are topically related but do not necessarily support
  the lay procedure "apply egg to skin".

Rule:

- Only runs when the aggregated stance is `Supports`.
- Detects diagnostic/test claims using terms such as `test`, `diagnose`,
  `detect`, `screen`, `check`, `confirm`, `rule out`.
- Also detects lay skin procedure claims such as `apply/rub/place` plus `skin`.
- Downgrades `Supports -> Neutral` when evidence text contains diagnostic
  uncertainty cues:
  - `gold standard`
  - `oral food challenge`
  - `double-blind`, `placebo-controlled`
  - `mainly clinical`, `primarily clinical`
  - `not accurate`, `not reliable`, `not definitive`
  - `limited accuracy`, `poor accuracy`, `single diagnosis`
- Also downgrades when the claim describes a simple skin application while the
  evidence describes formal diagnostic tests:
  - `skin prick`, `prick test`
  - `patch test`, `atopy patch`
  - `sIgE`, `specific IgE`
  - `oral food challenge`

The downgrade target is `Neutral`, not `Refutes`, because many diagnostic
papers do not say the test is useless; they say it is indirect, limited, or not
the gold standard for individual diagnosis.

When the gate fires, the stored label is `Neutral`, and the stored support score
is capped so downstream prompts do not see a neutral label with a high support
probability.

## Offline Evaluation After Fixed-Token Chunking

Tested with `factchecker_t26` on `offline_mock/**/process.json`.

Environment:

- Conda env: `factchecker_t26`
- Python: `3.9`
- Torch: `2.6.0+cu124`
- Transformers: `4.55.2`
- CUDA available during test: yes

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

Conclusion: fixed-token chunking reduces truncation risk but does not solve the
main stance problem by itself. The remaining issue is mostly model semantics:
BioLinkBERT-MedNLI often treats topical or indirect biomedical evidence as a
directional stance.

## Offline Evaluation After Section-Aware Chunking

Tested with `factchecker_t26` using
`pipeline/test_configs/kai_offline_mock_evaluate_section_stance.py`.

Report written to:

```text
offline_mock/section_stance_eval.json
```

Section-aware rule:

- If strict section headers are detected, each section is used as an individual
  stance candidate.
- If a section is too long, only that section is split with 64-token overlap.
- If no section header is detected, fall back to fixed-token chunking over the
  full abstract and mark it as `UNKNOWN`.
- No section weighting is applied.

Aggregate over 22 process files / 228 evidence items:

- Old stored stance: 191 `Supports`, 37 `Refutes`
- Fixed-token stance: 187 `Supports`, 39 `Refutes`, 2 `Neutral`
- Section-aware stance: 174 `Supports`, 37 `Refutes`, 17 `Neutral`

Transitions from fixed-token to section-aware:

- `Supports -> Supports`: 173
- `Refutes -> Refutes`: 37
- `Supports -> Neutral`: 14
- `Neutral -> Supports`: 1
- `Neutral -> Neutral`: 1
- `Refutes -> Neutral`: 2

Section extraction:

- 119 evidence items with recognized sections.
- 109 evidence items fell back to `UNKNOWN`.
- The known false-positive risk `these conclusions:` is not matched as a
  section header and stays `UNKNOWN`.

Summary conflict heuristic:

- Fixed-token conflicts: 112
- Section-aware conflicts: 107

Conclusion: section-aware chunking makes the stance output more conservative
and increases `Neutral`, but only modestly reduces summary conflicts. The main
remaining issue still appears to be BioLinkBERT-MedNLI treating indirect or
topical biomedical evidence as directional stance.

## Evaluation Run History

All runs below use the same stored offline data:

- `offline_mock/**/process.json`
- 22 process files
- 228 evidence items
- 77 unique abstracts observed in earlier length analysis

| Run | Description | Supports | Refutes | Neutral | Summary conflicts |
| --- | --- | ---: | ---: | ---: | ---: |
| Stored baseline | Existing stance labels in offline outputs | 191 | 37 | 0 | not measured globally |
| Fixed-token chunking | Title once + full abstract token chunks with 64-token overlap | 187 | 39 | 2 | 112 |
| Section-aware chunking | Sections replace full-abstract chunks when headers exist | 174 | 37 | 17 | 107 |
| 3-sentence sliding windows | Section text split into windows of 3 sentences, stride 1 | 138 | 46 | 44 | 116 |

Current conclusion from these runs:

- Section-aware chunking is the best stance-side default tested so far because
  it is more conservative and increases `Neutral` for weak/indirect abstracts.
- The improvement in stance-summary agreement is modest by itself.
- 3-sentence sliding windows are more conservative, but they performed worse
  against evidence summaries and create many more candidates per evidence item.
- The much larger mismatch reduction came from fixing the summary request so it
  receives the same stance that the UI displays.

Observed transitions:

- Stored baseline -> fixed-token:
  - `Supports -> Supports`: 185
  - `Refutes -> Refutes`: 35
  - `Supports -> Refutes`: 4
  - `Supports -> Neutral`: 2
  - `Refutes -> Supports`: 2
- Fixed-token -> section-aware:
  - `Supports -> Supports`: 173
  - `Refutes -> Refutes`: 37
  - `Supports -> Neutral`: 14
  - `Neutral -> Supports`: 1
  - `Neutral -> Neutral`: 1
  - `Refutes -> Neutral`: 2

Evidence summary comparison:

- Detailed evidence summaries are available for `offline_mock/DT0UIgzDZ79`.
- Fixed-token changed only 1 of 10 labels compared with the stored stance for
  those summary-backed evidence items.
- Section-aware evaluation reduces global heuristic summary conflicts from 112
  to 107, so the improvement is small.
- The most important remaining conflict type is still `Supports` assigned to
  indirect or topical evidence that the generated evidence summary describes as
  unclear, indirect, or contradictory.

Section extraction findings:

- 119 evidence items had recognized section headers.
- 109 evidence items fell back to `UNKNOWN`.
- Frequent extracted headers: `RESULTS`, `METHODS`, `BACKGROUND`,
  `CONCLUSIONS`, `OBJECTIVE`, `CONCLUSION`, `SUMMARY`,
  `PURPOSE OF REVIEW`, `RECENT FINDINGS`.
- The strict header regex avoids the known false positive `these conclusions:`;
  PMID `18162844` remains `UNKNOWN`.

3-sentence sliding-window test:

- Tested windows of about 3 sentences with stride 1.
- Average sentence chunks per evidence item: 7.11.
- Max sentence chunks per evidence item: 25.
- Counts: 138 `Supports`, 46 `Refutes`, 44 `Neutral`.
- Summary conflicts: 116.
- Decision: not used. It is slower and more conservative, but the mismatch is
  worse than section-aware chunking.

Reproduction commands:

```bash
python -m py_compile pipeline/steps/stance.py
conda run -n factchecker_t26 python -m py_compile pipeline/test_configs/kai_offline_mock_evaluate_section_stance.py
conda run -n factchecker_t26 python pipeline/test_configs/kai_offline_mock_evaluate_section_stance.py
```

Generated report:

```text
offline_mock/section_stance_eval.json
```

## Offline Evaluation After Diagnostic/Test Gate

Tested with:

```bash
conda run -n factchecker_t26 python pipeline/test_configs/kai_offline_mock_evaluate_diagnostic_stance_gates.py
```

Source sweep:

```text
offline_mock/sweep_section_aware_stance
```

Report:

```text
offline_mock/diagnostic_stance_gate_eval.json
```

Results over 122 section-aware evidence-summary items:

| Variant | Supports | Refutes | Neutral | Mismatch | Changed labels | Egg-skin mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current section-aware | 95 | 18 | 9 | 17 | 0 | 13 |
| Rule 1 only: diagnostic uncertainty/method mismatch | 82 | 18 | 22 | 4 | 13 | 0 |
| Rule 2 diagnostic only: `Supports` + rel `<0.80` | 82 | 18 | 22 | 4 | 13 | 0 |
| Rule 1 + Rule 2 diagnostic | 82 | 18 | 22 | 4 | 13 | 0 |
| Rule 2 global: all `Supports` + rel `<0.80` | 64 | 18 | 40 | 14 | 31 | 0 |
| Rule 1 + Rule 2 global | 64 | 18 | 40 | 14 | 31 | 0 |

Decision:

- Use Rule 1 as the live default.
- Do not use Rule 2 global; it is too aggressive and neutralizes many more
  supports while performing worse on mismatches.
- Keep Rule 2 diagnostic-only as an offline finding for now. It matched Rule 1
  on this sweep but is less semantically specific.

## Aggregation Variants Tested

All variants below were tested on the sweep data with the same BioLinkBERT
candidate scores. They only change the final aggregation rule.

Sweep source:

```text
offline_mock/sweep_20260626_072546
```

Against the old evidence summaries, before the summary route included stance:

| Aggregation | Supports | Refutes | Neutral | Mismatch with Evidence Summary |
| --- | ---: | ---: | ---: | ---: |
| Current section-aware aggregation | 166 | 35 | 17 | 99 |
| `refute=max`, `support=mean(top2)` | 156 | 44 | 18 | 99 |
| Support only from Results/Conclusion sections | 160 | 40 | 18 | 100 |

Against the newly generated summaries that received `evidence.stance` in the
request payload:

| Aggregation | Supports | Refutes | Neutral | Mismatch with Evidence Summary |
| --- | ---: | ---: | ---: | ---: |
| Current section-aware aggregation | 166 | 35 | 17 | 21 |
| `refute=max`, `support=mean(top2)` | 156 | 44 | 18 | 29 |
| Support only from Results/Conclusion sections | 160 | 40 | 18 | 24 |

Conclusion: the alternative aggregation rules did not improve the mismatch.
The current section-aware aggregation is the best of these tested options.

## Evidence Summary Stance Payload Fix

Bug found:

- Backend `/evidence_summary` could use an optional `evidence.stance`.
- Browser summary requests did not send `evidence.stance`.
- Therefore the summary prompt often received `Unknown` stance while the UI
  displayed a concrete stance label from the stance algorithm.

Fix:

- `browser-extension/content.js` now sends `stance: evidence?.stance || null`
  in the summary payload.
- `app/main.py` normalizes stance via `_extract_stance_label(...)`, accepting
  either a stance object or string.

New sweep generated with the fixed summary route:

```text
offline_mock/sweep_20260626_072546/run_*/DT*/evidence_summaries_with_stance/
offline_mock/sweep_20260626_072546/run_*/DT*/manifest_with_stance_summary.json
```

Reproduction command:

```bash
python pipeline/test_configs/kai_offline_mock_run_sweep_evidence_summaries_with_stance.py --force
```

Generation result:

| Metric | Count |
| --- | ---: |
| Process files | 20 |
| Evidence summaries regenerated | 218 |
| Failed requests | 0 |

Direct mismatch comparison using the stored stance that was sent in the payload:

| Summary source | Mismatch with stored stance |
| --- | ---: |
| Old summaries without stance payload | 103 / 218 |
| New summaries with stance payload | 9 / 218 |

Interpretation:

- `9 / 218` is the cleanest measure of the summary-route fix, because it
  compares the generated summary against the exact stance sent to the summary
  prompt.
- `21 / 218` is the mismatch when those same new summaries are compared against
  a freshly recomputed current section-aware stance. That number is higher
  because the recomputed stance is not always identical to the older stored
  stance in the sweep `process.json`.
- For live usage, the important invariant is: the summary must be generated
  from the same `evidence.stance` object that the UI displays.

## Deep Research Notes

The tested chunking variants are useful for truncation and conservatism, and
the summary payload fix solves most UI-visible stance/summary contradictions.
The remaining stance-side issue is that `cnut1648/biolinkbert-mednli` is not
fully calibrated for this task: it can still map biomedical topical relatedness
to `Supports`.

Promising next directions to research:

- Replace or complement MedNLI stance with a model trained/evaluated on
  scientific claim verification or biomedical evidence inference.
- Add a directness/relevance gate before stance: same population, intervention,
  comparator, outcome, and direction.
- Treat `Neutral` as the default unless evidence directly matches the claim's
  key qualifiers.
- Evaluate whether title-only stance should be removed entirely or used only as
  retrieval/debug metadata, because titles can amplify topical false support.
- Build a small labeled set from the current evidence summaries and manually
  adjudicate whether each paper `Supports`, `Refutes`, or is `Neutral/Indirect`.
