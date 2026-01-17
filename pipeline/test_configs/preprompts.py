
def get_prompt_s3_by_name(name: str) -> str:
    key = name.strip().lower()
    try:
        return PROMPT_TMPL_S3_BY_NAME[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown S3 prompt template: {name}. Available: {', '.join(PROMPT_TMPL_S3_BY_NAME)}"
        ) from exc


# ────────────────────────────────────────────────────────────────────
# Step 2: Extract medical claims from transcript
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S2 = """
You are part of a medical fact-checking pipeline.  
If you propagate a false statement, the system may mislead people.

INPUT TRANSCRIPT
---------------
{transcript}
---------------

TASK  
Extract **medical claims** suitable for fact-checking.

GOAL
Extract all distinct, checkable medical claims that appear in the transcript. The number of returned claims must adapt to the transcript content.

SELECTION RULES
1. Return as many distinct medical claims as are present, up to a maximum of **8**. If more than 8 are present, return the **8 most clinically important and/or potentially harmful**.
2. A "medical claim" is an assertion about health, disease, symptoms, diagnosis, treatment, prevention, risk, prognosis, nutrition/supplements, physiology, medical tests, medication safety, or health outcomes.
3. Aim for **high recall**: if a statement is plausibly a medical claim and is checkable, include it rather than omitting it.
4. Prefer **specific, testable assertions** over vague advice. Keep the speaker's implied certainty (do not add extra hedging or certainty).
5. Split compound statements into separate claims when they can be checked independently (e.g., "X reduces Y and Z" -> two claims). Merge duplicates / near-duplicates into one concise claim.
6. Preserve key qualifiers when present (population, dosage, timing, comparator, directionality like increases vs decreases). Rewrite each claim to be **self-contained** and **context-independent**, while preserving meaning.
7. Exclude: greetings, jokes, moral/motivational advice, non-medical content, pure opinions, and statements too vague or non-falsifiable to verify.
8. If there are **no** medical claims, return an empty array: []

STRICT OUTPUT  
A valid JSON array of 1–5 strings.  
No commentary, no extra keys, no markdown.
"""


# ────────────────────────────────────────────────────────────────────
# Step 3: Generate PubMed searches from medical claims
# ────────────────────────────────────────────────────────────────────
# Common introductory context (shared by all templates)
PROMPT_S3_INTRO = (
    "You are a biomedical information specialist generating PubMed searches for a medical fact-checking pipeline. "
    "The input CLAIM comes from an Instagram reel and may be informal or exaggerated. "
    "Translate informal wording into scientifically standard terminology suitable for PubMed searching "
    "(use clinical/scientific equivalents where applicable)."
)

# Common output formatting rules (most rules are shared; placeholders will insert variant-specific text)
PROMPT_S3_OUTPUT_RULES = (
    "OUTPUT RULES (strict)\n"
    "- Output ONLY the query string. No explanations. Exactly one line.\n"
    "- Use uppercase AND/OR/NOT.\n"
    "- Use parentheses to group synonyms.\n"
    "- Allowed field tags: [mh] and [tiab] only{ANCHOR_EXCEPT}.\n"
    "- Do NOT use quotation marks (including curly quotes) anywhere.\n"
    "- Multi-word phrases must be written as: word1 word2[tiab] (no quotes){ANCHOR_PHRASE}.\n"
    "- Ensure parentheses are balanced; no leading/trailing whitespace.\n"
    "- Do not tag groups of synonyms with a single field tag.\n"
    "- All words must be tagged{ANCHOR_EXCEPT2}.\n"
    "- Do not use generic domain words as filters (e.g., molecular biology, homeostasis, wellness, detox).\n"
)

# Additional output rule for Specific variants (appended to OUTPUT_RULES)
SPECIFIC_RECALL_RULE = (
    "- End with a final AND group that boosts clinical/human evidence recall, e.g., "
    "(humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] "
    "OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
)

# Base semantic rules (for Balanced variant; others will modify or extend this)
PROMPT_S3_SEMANTIC_BASE = (
    "SEMANTIC RULES\n"
    "1) Identify 2–4 core CONCEPTS from the claim (e.g., condition/population, intervention/exposure, outcome/mechanism).\n"
    "2) Build one synonym group per concept:\n"
    "   - Include 1–2 MeSH headings as term[mh] (only if confident they exist as MeSH terms).\n"
    "   - Include 2–6 scientific free-text terms as term[tiab], preferring standard medical terms over colloquial language.\n"
    "   - Avoid vague or influencer wording unless it is also a common scientific term.\n"
    "3) Combine concept groups with AND.\n"
    "4) ANCHOR RULE: The primary topic anchor (main condition or intervention/exposure) must apply to the whole query. Do not create an OR branch that omits this anchor.\n"
    "5) Handle absolutes: If the claim uses “cures”, “guarantees”, “all”, “detox”, etc., convert that into testable research language (e.g., treat*, reduc*, decreas*, improv*, efficacy, symptom*, biomarker*). Do NOT include words like \"cure\", \"curative\", \"healing\", or \"miracle\".\n"
    "6) If an outcome is overly broad (e.g., “inflammation”), operationalize it with more specific scientific endpoints or markers (e.g., inflammat*, anti-inflammatory, C-reactive protein/CRP, interleukin-6/IL-6, tumor necrosis factor/TNF), and/or include specific diseases if mentioned.\n"
    "7) Keep the query concise: max 4 concept groups total; ~8 terms per group. Avoid adding unnecessary dose/time details unless absolutely essential and likely to appear in titles/abstracts.\n"
)

# Counter-evidence rule (inserted into semantic rules for counter variants)
COUNTER_EVIDENCE_RULE = (
    "{NUM}) Add ONE counter-evidence concept group capturing **null/negative/adverse findings** relevant to the claim – e.g., ineffective[tiab], \"no effect\"[tiab], null[tiab], negative[tiab], adverse[tiab], harm[tiab], risk[tiab], toxicity[tiab] (4–8 such terms). "
    "Combine this group with AND.\n"
)

# Extra semantic rules for Specific (high-recall) variants
SPECIFIC_EXTRA_RULES = (
    "{NUM}) Add ONE final AND group to boost clinical/human evidence recall (do not mix these with the topic concept groups):\n"
    "    (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])\n"
    "{NEXT}) Avoid low-value terms: never include broad terms like \"study\"[tiab] as a search term. Also avoid generic concepts such as “homeostasis”, “wellness”, or “detox” unless the claim explicitly requires them.\n"
)

# Adjusted semantic rules template for ATM variants (different structure for anchor handling)
ATM_SEMANTIC_BASE = (
    "SEMANTIC RULES\n"
    "1) Identify the single most important **TOPIC ANCHOR** (the primary intervention/exposure or condition).\n"
    "2) Start with an **ANCHOR GROUP** that includes:\n"
    "   - **One untagged** scientific anchor term/phrase (no [mh] or [tiab]) to leverage automatic term mapping.\n"
    "   - Plus 1–2 MeSH terms for the anchor (term[mh], if confident) and 1–3 synonym variants (term[tiab]).\n"
    "   *Example:* `(untagged_anchor OR anchor[mh] OR anchor[tiab] OR ...)`\n"
    "3) Add {EXTRA_GROUPS} additional concept group(s) for other aspects (outcome, mechanism, population), using only [mh] and [tiab] tags. Each such group: 1–2 MeSH terms (if confident) and 2–5 scientific terms [tiab].\n"
    "4) Combine all groups with AND.\n"
    "5) ANCHOR RULE: The anchor group must apply to the entire query. Do not allow any OR branch that omits the anchor.\n"
    "6) Handle absolutes: replace any “cure/curative/healing/miracle” language with testable terms (treat*, reduc*, improv*, efficacy, symptom*, biomarker*). Do NOT use the words “cure”, “healing”, etc.\n"
    "7) If an outcome is broad (e.g., inflammation), refine it with specific scientific endpoints or markers (inflammat*, anti-inflammatory, CRP, IL-6, TNF) or include specific diseases when relevant.\n"
    "8) Keep the query concise: aim for ≤4 total concept groups (including the anchor group). Avoid unnecessary dosage/time specifics unless crucial.\n"
)

# ==== Assemble final prompt templates by inserting variant-specific parts ====

# =============================================================================
# 1. BALANCED
PROMPT_TMPL_S3_BALANCED = (
    f"""{PROMPT_S3_INTRO}

TASK  
Return ONE PubMed Boolean query (single line) that retrieves papers relevant to evaluating the claim.

{PROMPT_S3_OUTPUT_RULES.format(ANCHOR_EXCEPT='', ANCHOR_PHRASE='', ANCHOR_EXCEPT2='')}
{PROMPT_S3_SEMANTIC_BASE}CLAIM:\n{{claim}}\n
"""
)

# =============================================================================
# 2. SPECIFIC (high recall, human evidence focus)
PROMPT_TMPL_S3_SPECIFIC = (
    f"""{PROMPT_S3_INTRO}

TASK  
Return ONE PubMed Boolean query (single line) that prioritizes clinically informative human evidence with **HIGH recall**, while staying on-topic for the claim.

{PROMPT_S3_OUTPUT_RULES.format(ANCHOR_EXCEPT='', ANCHOR_PHRASE='', ANCHOR_EXCEPT2='')}\
{SPECIFIC_RECALL_RULE}\n
{PROMPT_S3_SEMANTIC_BASE}"""
    # Insert extra semantic rules for recall and avoiding low-value terms, with proper numbering
    + SPECIFIC_EXTRA_RULES.format(
        NUM='8' if '7)' in PROMPT_S3_SEMANTIC_BASE else '8',
        NEXT='9',
    )
    + f"CLAIM:\n{{claim}}\n"
)

# =============================================================================
# 3. ATM_ASSISTED (allows automatic term mapping via one untagged term)
PROMPT_TMPL_S3_ATM_ASSISTED = (
    f"""{PROMPT_S3_INTRO}

TASK  
Return ONE PubMed Boolean query (single line) optimized for **HIGH recall** by allowing PubMed’s automatic term mapping (ATM) to assist, while maintaining a structured query.

{PROMPT_S3_OUTPUT_RULES.format(ANCHOR_EXCEPT=', EXCEPT you may include **one untagged** term inside the anchor group', 
                                ANCHOR_PHRASE=' when tagged', 
                                ANCHOR_EXCEPT2=', except for that one untagged anchor term/phrase')}
{ATM_SEMANTIC_BASE.format(EXTRA_GROUPS='1–3')}CLAIM:\n{{claim}}\n"""
)

# =============================================================================
# 4. BALANCED_COUNTER (focus on counter-evidence)
PROMPT_TMPL_S3_BALANCED_COUNTER = (
    f"""{PROMPT_S3_INTRO}

TASK  
Return ONE PubMed Boolean query (single line) that **prioritizes counter-evidence** (null, negative, or adverse findings) relevant to evaluating the claim.

{PROMPT_S3_OUTPUT_RULES.format(ANCHOR_EXCEPT='', ANCHOR_PHRASE='', ANCHOR_EXCEPT2='')}
"""
    # Insert counter-evidence semantic rule at position 4, then the rest of base rules (with renumbering)
    + COUNTER_EVIDENCE_RULE.format(NUM='4')
    + PROMPT_S3_SEMANTIC_BASE.replace("4) ANCHOR RULE", "5) ANCHOR RULE")
    .replace("5) Handle absolutes", "6) Handle absolutes")
    .replace("6) If an outcome", "7) If an outcome")
    .replace("7) Keep the query concise", "8) Keep the query concise")
    + f"CLAIM:\n{{claim}}\n"
)

# =============================================================================
# 5. SPECIFIC_COUNTER (high recall + counter-evidence)
PROMPT_TMPL_S3_SPECIFIC_COUNTER = (
    f"""{PROMPT_S3_INTRO}

TASK  
Return ONE PubMed Boolean query (single line) that prioritizes clinically informative human evidence **and counter-evidence** (null/negative findings), while staying on-topic for the claim.

{PROMPT_S3_OUTPUT_RULES.format(ANCHOR_EXCEPT='', ANCHOR_PHRASE='', ANCHOR_EXCEPT2='')}\
{SPECIFIC_RECALL_RULE}\n
"""
    # Semantic rules: insert counter-evidence rule and recall/low-value extras with proper numbering
    + COUNTER_EVIDENCE_RULE.format(NUM='4')
    + PROMPT_S3_SEMANTIC_BASE.replace("4) ANCHOR RULE", "5) ANCHOR RULE")
    .replace("5) Handle absolutes", "6) Handle absolutes")
    .replace("6) If an outcome", "7) If an outcome")
    .replace("7) Keep the query concise", "9) Keep the query concise")
    + SPECIFIC_EXTRA_RULES.format(
        NUM='8',
        NEXT='9' if '9)' not in PROMPT_S3_SEMANTIC_BASE else '10',
    )
    + f"CLAIM:\n{{claim}}\n"
)

# =============================================================================
# 6. ATM_ASSISTED_COUNTER (ATM + counter-evidence)
PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER = (
    f"""{PROMPT_S3_INTRO}

TASK  
Return ONE PubMed Boolean query (single line) optimized for **HIGH recall** by using automatic term mapping, while **prioritizing counter-evidence** (null/negative findings).

{PROMPT_S3_OUTPUT_RULES.format(ANCHOR_EXCEPT=', EXCEPT you may include **one untagged** term inside the anchor group', 
                                ANCHOR_PHRASE=' when tagged', 
                                ANCHOR_EXCEPT2=', except for that one untagged anchor term/phrase')}
"""
    # Semantic rules for ATM with counter: insert counter rule and adjust group count
    + COUNTER_EVIDENCE_RULE.format(NUM='4')
    + ATM_SEMANTIC_BASE.format(EXTRA_GROUPS='1–2')
    .replace("4) Combine all groups", "5) Combine all groups")
    .replace("5) ANCHOR RULE", "6) ANCHOR RULE")
    .replace("6) Handle absolutes", "7) Handle absolutes")
    .replace("7) If an outcome", "8) If an outcome")
    .replace("8) Keep the query concise", "9) Keep the query concise")
    + f"CLAIM:\n{{claim}}\n"
)

PROMPT_TMPL_S3_BY_NAME = {
    "balanced": PROMPT_TMPL_S3_BALANCED,
    "specific": PROMPT_TMPL_S3_SPECIFIC,
    "atm_assisted": PROMPT_TMPL_S3_ATM_ASSISTED,
    "balanced_counter": PROMPT_TMPL_S3_BALANCED_COUNTER,
    "specific_counter": PROMPT_TMPL_S3_SPECIFIC_COUNTER,
    "atm_assisted_counter": PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER,
}



# ────────────────────────────────────────────────────────────────────
# Step 6: Filter irrelevant evidence 
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S6 = """
You are verifying whether the following evidence abstract is relevant to the given claim.
Think carefully (silently) before answering with one word.

Use metadata if present:  
- **w**: study strength (0–1 scale for evidence quality; not a topical relevance measure)  
- **rel**: relevance to claim (0–1, higher = more on-topic)  
- **stance**: NLI stance on the abstract (S = supports, R = refutes, N = neutral)  

Decision rules:  
- Respond "yes" if the abstract is directly about the claim (clearly supports or refutes it).  
- Respond "no" if it is off-topic, has a mismatched population/intervention/outcome, **or** if `rel` is low **and** stance is Neutral/NA.  
- If metadata seems to conflict with the abstract text, rely on the **abstract’s content**.

STATEMENT: {statement}

EVIDENCE:
{evidence}

Does the EVIDENCE relate to the STATEMENT?  
Respond with only: yes | no
"""

# ────────────────────────────────────────────────────────────────────
# Step 7: Final verdict and truthfulness scoring
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S7 = """
You are a professional medical fact-checker.  
A wrong verdict could spread misinformation, so analyze the evidence carefully before finalizing your answer.

TASK:  
Determine whether the provided evidence **collectively** SUPPORTS the claim, REFUTES the claim, or leaves the claim UNCERTAIN.

- If the evidence is contradictory or insufficient, you may use the metadata (relevance, study weight, stance) to inform your decision.

METADATA (in the EVIDENCE lines, if present):  
- **w** (weight): study type strength (0–1, higher = stronger evidence quality)  
- **rel**: relevance to the claim (0–1, higher = more on-topic)  
- **stance**: NLI stance for each abstract — S (supports), R (refutes), or N (neutral), sometimes with a probability score.

Give more weight to evidence with higher **w** and **rel**, and with strong support/refute **stance**. Downweight evidence that has low relevance or only neutral/unclear findings.

CONSERVATIVE RULES:  
- If the evidence is sparse, low-quality, low-relevance, or shows mixed/neutral findings, **choose UNCERTAIN**.  
- Only conclude **TRUE** or **FALSE** if there are multiple high-quality, highly relevant pieces of evidence that **consistently** support or refute the claim.  
- If no evidence is provided, default to **UNCERTAIN** (with a slight lean toward false due to lack of supporting evidence).  
- Avoid overconfidence. When in doubt, it’s safer to be UNCERTAIN. Consider the overall evidence **in totality**, not just individual phrases.

SCORE:  
The **FINALSCORE** is the probability that the claim is true (0.00 = certainly false, 1.00 = certainly true).  
- For an **UNCERTAIN** verdict, use a score around 0.50 (indicating doubt).  
- If the claim is supported or refuted by moderate evidence, choose an intermediate confidence (e.g. 0.60–0.80 range).  
- If the evidence is very strong and unidirectional, you can go up to ~0.90.  
- Do **not** use extreme values like 1.00 or 0.00, since conclusions should be cautious.

CLAIM:  
{claim_text}

EVIDENCE (each source grouped separately, including PubMed abstracts and any RAG findings):  
{evidence_block}

Now provide the final **verdict** and **score** in the exact format below, with no additional commentary or explanation:

STRICT OUTPUT – exactly two lines:  
VERDICT: true|false|uncertain  
FINALSCORE: <probability 0.00–1.00>
"""

# ───────────────────────── Direct Prompt ─────────────────────────────────

PROMPT_TMPL_RAW = """
You are a professional medical fact-checker.  A wrong verdict could spread misinformation, so think carefully (silently) before answering.

Decide whether the CLAIM is SUPPORTED, REFUTED, or UNCERTAIN. 
Give the final response in the following format. STRICT OUTPUT – exactly two lines, nothing else:


VERDICT: true|false|uncertain
FINALSCORE: <probability 0.00–1.00>
"""
