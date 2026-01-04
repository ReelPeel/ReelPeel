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
PROMPT_TMPL_S3_NARROW_QUERY = """
You are a biomedical librarian inside a fact‐checking app. You get a CLAIM about health or medicine extracted from a instragram video transcript.
**Think quickly (silently)** about the PICO concepts in the CLAIM, identify appropriate MeSH headings (in quotes with [MeSH]) and text‐word synonyms (with [tiab]), and produce ONE PubMed Boolean query string that:
Example:
(
  "Smoking"[MeSH] 
  OR smoking[tiab] 
  OR smokers[tiab] 
  OR "tobacco use"[tiab]
)
AND
(
  "Lung Function Tests"[MeSH] 
  OR "pulmonary function"[tiab] 
  OR "lung function"[tiab] 
  OR FEV1[tiab] 
  OR FVC[tiab]
)
AND
(
  "Inflammation"[MeSH] 
  OR inflammation[tiab] 
  OR "mucus hypersecretion"[tiab] 
  OR mucus[tiab]
)
AND
(
  "Proanthocyanidins"[MeSH] 
  OR proanthocyanidin*[tiab] 
  OR OPC[tiab] 
  OR "Traumotein"[tiab] 
  OR "pine bark extract"[tiab] 
  OR Pycnogenol[tiab]
)


• Is exactly one line (no line breaks). 
• Is very precise and specific to the CLAIM.  
• Uses uppercase AND/OR to combine concepts.  
• Wraps MeSH terms in quotes followed by [MeSH] (e.g., "Smoking"[MeSH]).  
• Marks synonym or free‐text terms with [tiab] (e.g., smoking[tiab]).  
• Groups synonyms with parentheses; groups PICO domains by combining with AND.  
• Does not include field tags other than [MeSH] and [tiab].  
• Does not include quotation marks around free‐text terms (other than MeSH).  
• Begins with a letter (A–Z or a–z) and contains no leading/trailing spaces.  
• Contains no line breaks or extra whitespace.


CLAIM:
{claim}
"""

# ────────────────────────────────────────────────────────────────────
# Step3: PubMed query generation (REVISED)
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S3_BROAD_QUERY = """
You are a biomedical librarian inside a fact‐checking app. You get a CLAIM about health or medicine extracted from a instragram video transcript.
**Think quickly (silently)** about the PICO concepts in the CLAIM, identify appropriate MeSH headings (in quotes with [MeSH]) and text‐word synonyms (with [tiab]), and produce ONE PubMed Boolean query string that:
Example:
(
  "Smoking"[MeSH] 
  OR smoking[tiab] 
  OR smokers[tiab] 
  OR "tobacco use"[tiab]
)
AND
(
  "Lung Function Tests"[MeSH] 
  OR "pulmonary function"[tiab] 
  OR "lung function"[tiab] 
  OR FEV1[tiab] 
  OR FVC[tiab]
)
AND
(
  "Inflammation"[MeSH] 
  OR inflammation[tiab] 
  OR "mucus hypersecretion"[tiab] 
  OR mucus[tiab]
)
AND
(
  "Proanthocyanidins"[MeSH] 
  OR proanthocyanidin*[tiab] 
  OR OPC[tiab] 
  OR "Traumotein"[tiab] 
  OR "pine bark extract"[tiab] 
  OR Pycnogenol[tiab]
)


• Is exactly one line (no line breaks). 
• Is broad on the topic but still relevant to the CLAIM. 
• Uses uppercase AND/OR to combine concepts.  
• Wraps MeSH terms in quotes followed by [MeSH] (e.g., "Smoking"[MeSH]).  
• Marks synonym or free‐text terms with [tiab] (e.g., smoking[tiab]).  
• Groups synonyms with parentheses; groups PICO domains by combining with AND.  
• Does not include field tags other than [MeSH] and [tiab].  
• Does not include quotation marks around free‐text terms (other than MeSH).  
• Begins with a letter (A–Z or a–z) and contains no leading/trailing spaces.  
• Contains no line breaks or extra whitespace.


CLAIM:
{claim}
"""

PROMPT_TMPL_S3_SYNONYMS_QUERY = """
You are a biomedical librarian inside a fact‐checking app. You get a CLAIM about health or medicine extracted from a instragram video transcript.
**Think quickly (silently)** about the PICO concepts in the CLAIM, identify appropriate MeSH headings (in quotes with [MeSH]) and text‐word synonyms (with [tiab]), and produce ONE PubMed Boolean query string that:
Example:
(
  "Smoking"[MeSH] 
  OR smoking[tiab] 
  OR smokers[tiab] 
  OR "tobacco use"[tiab]
)
AND
(
  "Lung Function Tests"[MeSH] 
  OR "pulmonary function"[tiab] 
  OR "lung function"[tiab] 
  OR FEV1[tiab] 
  OR FVC[tiab]
)
AND
(
  "Inflammation"[MeSH] 
  OR inflammation[tiab] 
  OR "mucus hypersecretion"[tiab] 
  OR mucus[tiab]
)
AND
(
  "Proanthocyanidins"[MeSH] 
  OR proanthocyanidin*[tiab] 
  OR OPC[tiab] 
  OR "Traumotein"[tiab] 
  OR "pine bark extract"[tiab] 
  OR Pycnogenol[tiab]
)


• Is exactly one line (no line breaks). 
• Use a variety of synonyms to cover the CLAIM comprehensively.
• Uses uppercase AND/OR to combine concepts.  
• Wraps MeSH terms in quotes followed by [MeSH] (e.g., "Smoking"[MeSH]).  
• Marks synonym or free‐text terms with [tiab] (e.g., smoking[tiab]).  
• Groups synonyms with parentheses; groups PICO domains by combining with AND.  
• Does not include field tags other than [MeSH] and [tiab].  
• Does not include quotation marks around free‐text terms (other than MeSH).  
• Begins with a letter (A–Z or a–z) and contains no leading/trailing spaces.  
• Contains no line breaks or extra whitespace.


CLAIM:
{claim}
"""

# ────────────────────────────────────────────────────────────────────
# Step5: Summarise PubMed search results
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S5 = """
You are summarising evidence for a medical fact-checker.  
Think briefly (silently) about study design and main results, then write the summary.

GUIDELINES  
• ≤ 300 words, plain text.  
• Focus on methods, population, key outcomes, effect sizes, and limitations.  
• Do NOT add interpretation beyond the abstract itself.

TEXT TO SUMMARISE
-----------------
{abstract}
-----------------
"""


# ────────────────────────────────────────────────────────────────────
# Step6: Get rid of irrelevant evidence
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S6 = """
You are verifying whether the summary actually addresses the claim.  
Think (silently) first; then answer in exactly one word.

STATEMENT: {statement}

EVIDENCE SUMMARY: {evidence_summary}

Does the summary *directly relate to or support* the statement?
Respond with only: yes   |   no
"""

# ────────────────────────────────────────────────────────────────────
# Step7: Statement rating
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_S7 = """
You are a professional medical fact-checker.  
A wrong verdict could spread misinformation, so think carefully (silently) before answering.

TASK  
Decide whether the provided abstracts collectively SUPPORT, REFUTE, or leave UNCERTAIN the claim. If evidence is insufficient or contradictory, use the "weights" to guide your decision in the "EVIDENCE" if available.

CLAIM:
{claim_text}

EVIDENCE:
{evidence_block}

Inclunding the Scientific Evidence together with Common Sense and the Context of the Video Transcript:
{transcript}

give the final response in the following format:

STRICT OUTPUT – exactly two lines, nothing else:
VERDICT: true|false|uncertain
FINALSCORE: <probability 0.00–1.00>
"""

# ────────────────────────────────────────────────────────────────────
# Step7: Statement rating
# ────────────────────────────────────────────────────────────────────
PROMPT_TMPL_RAW = """
You are a professional medical fact-checker.  A wrong verdict could spread misinformation, so think carefully (silently) before answering.

Decide whether the CLAIM is SUPPORTED, REFUTED, or UNCERTAIN. 
Give the final response in the following format. STRICT OUTPUT – exactly two lines, nothing else:


VERDICT: true|false|uncertain
FINALSCORE: <probability 0.00–1.00>
"""