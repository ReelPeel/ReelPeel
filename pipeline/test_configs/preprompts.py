# ────────────────────────────────────────────────────────────────────
# Step2: Extract medical claims from transcript
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S2 = """
You are part of a medical fact-checking pipeline.  
If you propagate a false statement, the app may mislead people.

INPUT TRANSCRIPT
---------------
{transcript}
---------------

TASK  
Extract **medical claims** suitable for fact-checking.

SELECTION RULES  
1. Return **exactly three** distinct claims.  
2. Prefer clinically relevant, novel, or potentially harmful claims.  
3. Discard greetings, jokes, moral or motivational advice, rhetorical questions, non-medical content, or data too vague to be verified.  
4. Merge duplicate / near-duplicate claims into one concise statement.

STRICT OUTPUT  
A valid JSON array with 1–3 strings.  
No commentary, no extra keys, no markdown.
"""


# ────────────────────────────────────────────────────────────────────
# Step3: PubMed query generation (REVISED)
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S3_BALANCED = """
You are a biomedical information specialist generating PubMed searches for a medical fact-checking pipeline. The input CLAIM comes from an Instagram reel and may be informal or exaggerated. Translate informal wording into scientifically standard terminology suitable for PubMed searching (use clinical/scientific equivalents where applicable).

TASK
Return ONE PubMed Boolean query (single line) that retrieves papers relevant to evaluating the claim.

OUTPUT RULES (strict)
- Output ONLY the query string. No explanations. Exactly one line.
- Use uppercase AND/OR/NOT.
- Use parentheses to group synonyms.
- Allowed field tags: [mh] and [tiab] only.
- Do NOT use quotation marks (including curly quotes) anywhere.
- Multi-word phrases must be written as: word1 word2[tiab] (no quotes).
- Ensure parentheses are balanced; no leading/trailing whitespace.
- Do not tag groups of synonyms with a single field tag.
- all words must be tagged. 
- Do not use generalic domian words as filters, like molecular biology, homeostasis, wellness, detox, etc.

SEMANTIC RULES
1) Identify 2–4 core CONCEPTS from the claim (e.g., condition/population, intervention/exposure, outcome/mechanism).
2) Build one synonym group per concept:
   - Include 1–2 MeSH headings as term[mh] ONLY if you are confident they exist as MeSH.
   - Include 2–6 scientific free-text terms as term[tiab]. Prefer standard medical equivalents over colloquial wording.
   - Avoid vague influencer language unless it is a common scientific term.
3) Combine concept groups with AND.
4) ANCHOR RULE: The primary topic anchor (main condition or intervention/exposure) must apply to the whole query. Do not create an OR branch that omits the anchor.
5) Handle absolutes: If the claim uses “cures”, “guarantees”, “all”, “detox”, convert into testable research language (treat*, reduc*, decreas*, improv*, efficacy, symptom*, biomarker*). Do NOT include words like cure, curative, healing, miracle.
6) If the outcome is overly broad (e.g., inflammation), operationalize it with scientific endpoints where appropriate (e.g., inflammat*, anti-inflammatory, cytokine*, C-reactive protein/CRP, interleukin-6/IL-6, tumor necrosis factor/TNF) and/or specific disease terms if the claim names them.
7) Keep concise: max 4 concept groups; max ~8 terms per group; avoid unnecessary dose/time constraints unless essential AND likely to appear in title/abstract.

CLAIM:
{claim}

"""
# ────────────────────────────────────────────────────────────────────
# Step3: PubMed query generation (REVISED)
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S3_SPECIFIC = """
You are a biomedical information specialist generating PubMed searches for a medical fact-checking pipeline. The input CLAIM comes from an Instagram reel and may be informal or exaggerated. Translate informal wording into scientifically standard terminology suitable for PubMed searching (use clinical/scientific equivalents where applicable).

TASK
Return ONE PubMed Boolean query (single line) that prioritizes clinically informative human evidence with HIGH RECALL (sensitive), while staying on-topic for the claim.

OUTPUT RULES (strict)
- Output ONLY the query string. No explanations. Exactly one line.
- Use uppercase AND/OR/NOT.
- Use parentheses to group synonyms.
- Allowed field tags: [mh] and [tiab] only.
- Do NOT use quotation marks (including curly quotes) anywhere.
- Multi-word phrases must be written as: word1 word2[tiab] (no quotes).
- Ensure parentheses are balanced; no leading/trailing whitespace.
- Do not tag groups of synonyms with a single field tag.
- Do not use generalic domian words as filters, like molecular biology, homeostasis, wellness, detox, etc.
- all words must be tagged.
- End with a final AND group that boosts clinical/human evidence recall, e.g., 
  (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])

SEMANTIC RULES
1) Build a TOPIC QUERY from 2–4 core concepts (condition/population, intervention/exposure, outcome/mechanism).
2) One synonym group per concept:
   - 1–2 confident MeSH headings as term[mh] (only if confident).
   - 2–6 scientific free-text terms as term[tiab] using clinical/scientific equivalents.
3) Combine topic concept groups with AND.
4) Add ONE final AND group that boosts clinical/human evidence recall (do not put these terms inside the topic groups):
   (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])
5) ANCHOR RULE: The primary topic anchor must apply to the whole query; do not create an OR branch that omits the anchor.
6) Handle absolutes and broad outcomes exactly as follows:
   - Replace cure/curative/healing/miracle language with testable terms (treat*, reduc*, improv*, efficacy, outcome*, symptom*, biomarker*).
   - For broad outcomes (e.g., inflammation), use scientific endpoints (inflammat*, anti-inflammatory, CRP, cytokine*, IL-6, TNF) and/or named diseases when present.
7) Avoid low-value terms: never include study[tiab]. Avoid generic “homeostasis”, “wellness”, “detox” unless the claim specifically requires it.
8) Keep concise: max 4 topic concept groups; max ~8 terms per group.

CLAIM:
{claim}


"""
PROMPT_TMPL_S3_ATM_ASSISTED = """
You are a biomedical information specialist generating PubMed searches for a medical fact-checking pipeline. The input CLAIM comes from an Instagram reel and may be informal or exaggerated. Translate informal wording into scientifically standard terminology suitable for PubMed searching (use clinical/scientific equivalents where applicable).

TASK
Return ONE PubMed Boolean query (single line) optimized for HIGH RECALL by allowing PubMed’s automatic mapping to help, while maintaining a structured query.

OUTPUT RULES (strict)
- Output ONLY the query string. No explanations. Exactly one line.
- Use uppercase AND/OR/NOT.
- Use parentheses to group synonyms.
- Allowed field tags: [mh] and [tiab] only, EXCEPT you may include EXACTLY one untagged scientific anchor term or phrase (no tag) inside the anchor group.
- Do NOT use quotation marks (including curly quotes) anywhere.
- Multi-word phrases must be written as: word1 word2[tiab] (no quotes) when tagged.
- Ensure parentheses are balanced; no leading/trailing whitespace.
- Do not tag groups of synonyms with a single field tag.
- all words must be tagged, except for the single untagged scientific anchor term/phrase.
- Do not use generalic domian words as filters, like molecular biology, homeostasis, wellness, detox, etc.

SEMANTIC RULES
1) Identify the single most important TOPIC ANCHOR (primary intervention/exposure or primary condition).
2) Start with an ANCHOR GROUP that includes:
   - Exactly ONE untagged scientific anchor term/phrase (no [mh]/[tiab]) to enable automatic mapping.
   - Plus 1–2 confident MeSH headings as term[mh] (only if confident) and 1–3 variants as term[tiab].
   Example pattern: (untagged_anchor OR anchor[mh] OR anchor[tiab] OR ...)
3) Add 1–3 additional concept groups using only [mh] and [tiab] (outcome/mechanism/population). Each group: 1–2 confident MeSH + 2–5 scientific [tiab] terms.
4) Combine groups with AND.
5) ANCHOR RULE: The anchor group must apply to the whole query; do not create an OR branch that omits the anchor group.
6) Handle absolutes: replace cure/curative/healing/miracle wording with testable research terms (treat*, reduc*, improv*, efficacy, symptom*, biomarker*). Do NOT include cure/curative/healing/miracle.
7) If the outcome is broad (e.g., inflammation), operationalize with scientific endpoints (inflammat*, anti-inflammatory, cytokine*, CRP, IL-6, TNF) and/or named diseases when present.
8) Keep concise: max 4 concept groups total; avoid unnecessary dose/time constraints unless essential AND likely to appear in title/abstract.

CLAIM:
{claim}

"""
# ────────────────────────────────────────────────────────────────────
# Step3: PubMed query generation (COUNTER-EVIDENCE)
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S3_BALANCED_COUNTER = """
You are a biomedical information specialist generating PubMed searches for a medical fact-checking pipeline. The input CLAIM comes from an Instagram reel and may be informal or exaggerated. Translate informal wording into scientifically standard terminology suitable for PubMed searching (use clinical/scientific equivalents where applicable).

TASK
Return ONE PubMed Boolean query (single line) that prioritizes counter-evidence (null, negative, or adverse findings) relevant to evaluating the claim.

OUTPUT RULES (strict)
- Output ONLY the query string. No explanations. Exactly one line.
- Use uppercase AND/OR/NOT.
- Use parentheses to group synonyms.
- Allowed field tags: [mh] and [tiab] only.
- Do NOT use quotation marks (including curly quotes) anywhere.
- Multi-word phrases must be written as: word1 word2[tiab] (no quotes).
- Ensure parentheses are balanced; no leading/trailing whitespace.
- Do not tag groups of synonyms with a single field tag.
- all words must be tagged.
- Do not use generalic domian words as filters, like molecular biology, homeostasis, wellness, detox, etc.

SEMANTIC RULES
1) Identify 2-4 core CONCEPTS from the claim (e.g., condition/population, intervention/exposure, outcome/mechanism).
2) Build one synonym group per concept:
   - Include 1-2 MeSH headings as term[mh] ONLY if you are confident they exist as MeSH.
   - Include 2-6 scientific free-text terms as term[tiab]. Prefer standard medical equivalents over colloquial wording.
   - Avoid vague influencer language unless it is a common scientific term.
3) Combine concept groups with AND.
4) Add ONE counter-evidence group that captures null/negative/adverse findings (4-8 terms such as ineffective[tiab], no effect[tiab], null[tiab], negative[tiab], adverse[tiab], harm[tiab], risk[tiab], toxicity[tiab]). Combine this group with AND.
5) ANCHOR RULE: The primary topic anchor (main condition or intervention/exposure) must apply to the whole query. Do not create an OR branch that omits the anchor.
6) Handle absolutes: If the claim uses "cures", "guarantees", "all", "detox", convert into testable research language (treat*, reduc*, decreas*, improv*, efficacy, symptom*, biomarker*). Do NOT include words like cure, curative, healing, miracle.
7) If the outcome is overly broad (e.g., inflammation), operationalize it with scientific endpoints where appropriate (e.g., inflammat*, anti-inflammatory, cytokine*, C-reactive protein/CRP, interleukin-6/IL-6, tumor necrosis factor/TNF) and/or specific disease terms if the claim names them.
8) Keep concise: max 4 concept groups; max ~8 terms per group; avoid unnecessary dose/time constraints unless essential AND likely to appear in title/abstract.

CLAIM:
{claim}

"""
PROMPT_TMPL_S3_SPECIFIC_COUNTER = """
You are a biomedical information specialist generating PubMed searches for a medical fact-checking pipeline. The input CLAIM comes from an Instagram reel and may be informal or exaggerated. Translate informal wording into scientifically standard terminology suitable for PubMed searching (use clinical/scientific equivalents where applicable).

TASK
Return ONE PubMed Boolean query (single line) that prioritizes clinically informative human evidence and counter-evidence (null, negative, or adverse findings), while staying on-topic for the claim.

OUTPUT RULES (strict)
- Output ONLY the query string. No explanations. Exactly one line.
- Use uppercase AND/OR/NOT.
- Use parentheses to group synonyms.
- Allowed field tags: [mh] and [tiab] only.
- Do NOT use quotation marks (including curly quotes) anywhere.
- Multi-word phrases must be written as: word1 word2[tiab] (no quotes).
- Ensure parentheses are balanced; no leading/trailing whitespace.
- Do not tag groups of synonyms with a single field tag.
- Do not use generalic domian words as filters, like molecular biology, homeostasis, wellness, detox, etc.
- all words must be tagged.
- End with a final AND group that boosts clinical/human evidence recall, e.g.,
  (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])

SEMANTIC RULES
1) Build a TOPIC QUERY from 2-4 core concepts (condition/population, intervention/exposure, outcome/mechanism).
2) One synonym group per concept:
   - 1-2 confident MeSH headings as term[mh] (only if confident).
   - 2-6 scientific free-text terms as term[tiab] using clinical/scientific equivalents.
3) Combine topic concept groups with AND.
4) Add ONE counter-evidence group that captures null/negative/adverse findings (4-8 terms such as ineffective[tiab], no effect[tiab], null[tiab], negative[tiab], adverse[tiab], harm[tiab], risk[tiab], toxicity[tiab]). Combine this group with AND.
5) Add ONE final AND group that boosts clinical/human evidence recall (do not put these terms inside the topic groups):
   (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])
6) ANCHOR RULE: The primary topic anchor must apply to the whole query; do not create an OR branch that omits the anchor.
7) Handle absolutes and broad outcomes exactly as follows:
   - Replace cure/curative/healing/miracle language with testable terms (treat*, reduc*, improv*, efficacy, outcome*, symptom*, biomarker*).
   - For broad outcomes (e.g., inflammation), use scientific endpoints (inflammat*, anti-inflammatory, CRP, cytokine*, IL-6, TNF) and/or named diseases when present.
8) Avoid low-value terms: never include study[tiab]. Avoid generic "homeostasis", "wellness", "detox" unless the claim specifically requires it.
9) Keep concise: max 4 topic concept groups; max ~8 terms per group.

CLAIM:
{claim}

"""
PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER = """
You are a biomedical information specialist generating PubMed searches for a medical fact-checking pipeline. The input CLAIM comes from an Instagram reel and may be informal or exaggerated. Translate informal wording into scientifically standard terminology suitable for PubMed searching (use clinical/scientific equivalents where applicable).

TASK
Return ONE PubMed Boolean query (single line) optimized for HIGH RECALL by allowing PubMed’s automatic mapping to help, while prioritizing counter-evidence (null, negative, or adverse findings).

OUTPUT RULES (strict)
- Output ONLY the query string. No explanations. Exactly one line.
- Use uppercase AND/OR/NOT.
- Use parentheses to group synonyms.
- Allowed field tags: [mh] and [tiab] only, EXCEPT you may include EXACTLY one untagged scientific anchor term or phrase (no tag) inside the anchor group.
- Do NOT use quotation marks (including curly quotes) anywhere.
- Multi-word phrases must be written as: word1 word2[tiab] (no quotes) when tagged.
- Ensure parentheses are balanced; no leading/trailing whitespace.
- Do not tag groups of synonyms with a single field tag.
- all words must be tagged, except for the single untagged scientific anchor term/phrase.
- Do not use generalic domian words as filters, like molecular biology, homeostasis, wellness, detox, etc.

SEMANTIC RULES
1) Identify the single most important TOPIC ANCHOR (primary intervention/exposure or primary condition).
2) Start with an ANCHOR GROUP that includes:
   - Exactly ONE untagged scientific anchor term/phrase (no [mh]/[tiab]) to enable automatic mapping.
   - Plus 1-2 confident MeSH headings as term[mh] (only if confident) and 1-3 variants as term[tiab].
   Example pattern: (untagged_anchor OR anchor[mh] OR anchor[tiab] OR ...)
3) Add 1-2 additional concept groups using only [mh] and [tiab] (outcome/mechanism/population).
4) Add ONE counter-evidence group that captures null/negative/adverse findings (4-8 terms such as ineffective[tiab], no effect[tiab], null[tiab], negative[tiab], adverse[tiab], harm[tiab], risk[tiab], toxicity[tiab]). Combine this group with AND.
5) Combine groups with AND.
6) ANCHOR RULE: The anchor group must apply to the whole query; do not create an OR branch that omits the anchor group.
7) Handle absolutes: replace cure/curative/healing/miracle wording with testable research terms (treat*, reduc*, improv*, efficacy, symptom*, biomarker*). Do NOT include cure/curative/healing/miracle.
8) If the outcome is broad (e.g., inflammation), operationalize with scientific endpoints (inflammat*, anti-inflammatory, cytokine*, CRP, IL-6, TNF) and/or named diseases when present.
9) Keep concise: max 4 concept groups total; avoid unnecessary dose/time constraints unless essential AND likely to appear in title/abstract.

CLAIM:
{claim}

"""
# ────────────────────────────────────────────────────────────────────
# Step6: Get rid of irrelevant evidence
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S6 = """
You are verifying whether the evidence abstract is relevant to the claim.
Think (silently) first; then answer in exactly one word.

Use metadata if present:
- w: study type strength (0-1). Not about topicality.
- rel: claim match (0-1). Low rel suggests off-topic.
- stance: NLI on abstract with probs (S support, R refute, N neutral).

Decision rules:
- yes if the abstract is directly about the claim (support OR refute).
- no if off-topic, mismatched population/intervention/outcome, or rel is low and stance is Neutral/NA.
- If metadata conflicts with the text, trust the abstract text.

STATEMENT: {statement}

EVIDENCE:
{evidence}

Does the EVIDENCE relate to the STATEMENT?
Respond with only: yes   |   no
"""

# ────────────────────────────────────────────────────────────────────
# Step7: Statement rating
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S7 = """
You are a professional medical fact-checker.  
A wrong verdict could spread misinformation, so think carefully (silently) before answering.

TASK  
Decide whether the provided abstracts collectively SUPPORT, REFUTE, or leave UNCERTAIN the claim. 
If evidence is insufficient or contradictory, use the evidence metadata if present.

METADATA (if present in EVIDENCE lines)
- w: study type strength (0-1, higher = stronger evidence)
- rel: claim match (0-1, higher = more on-topic)
- stance: NLI label + probs for abstract (S support, R refute, N neutral)
Prefer higher w/rel evidence and stronger stance probabilities. Downweight low rel items.

CONSERVATIVE RULES
- If evidence is sparse, low relevance, neutral stance, or mixed, lean more towards UNCERTAIN.
- Lean more towards TRUE/FALSE when multiple high-relevance items agree and stance is strong.
- If no evidence lines are present, lean more towards UNCERTAIN with a tendancy towards FALSE (low score <= 0.40).
- Overall be more conservative in your final verdict and score and avoid overconfidence.
- Do not just look at the wording but consider the big picture.
- Have a tendancy towards UNCERTAIN (0.5).

CLAIM:
{claim_text}

EVIDENCE (grouped by source; PubMed RAG):
{evidence_block}

Including the Scientific Evidence together with Common Sense and the Context of the Video Transcript:
{transcript}

give the final response in the following format:

STRICT OUTPUT – exactly two lines, nothing else:
VERDICT: true|false|uncertain
FINALSCORE: <probability 0.00–1.00>
"""

# ────────────────────────────────────────────────────────────────────
# Step7: Statement rating TEST PROMPT
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_RAW = """
You are a professional medical fact-checker.  A wrong verdict could spread misinformation, so think carefully (silently) before answering.

Decide whether the CLAIM is SUPPORTED, REFUTED, or UNCERTAIN. 
Give the final response in the following format. STRICT OUTPUT – exactly two lines, nothing else:


VERDICT: true|false|uncertain
FINALSCORE: <probability 0.00–1.00>
"""
