from copy import deepcopy

from pipeline.test_configs.preprompts import (
    PROMPT_TMPL_S3_ATM_ASSISTED,
    PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER,
    PROMPT_TMPL_S3_BALANCED,
    PROMPT_TMPL_S3_BALANCED_COUNTER,
    PROMPT_TMPL_S3_SPECIFIC,
    PROMPT_TMPL_S3_SPECIFIC_COUNTER,
    PROMPT_TMPL_S6,
    PROMPT_TMPL_S7,
    PROMPT_TMPL_S7_RAW,
    PROMPT_TMPL_S7_ASYMMETRIC,
    PROMPT_TMPL_S7_ASYMMETRIC_RAW
)

# Full pipeline reference:
# 1) Mock Input (Simulating Whisper)
# 2) Extraction
# 3) Generate Query
# 4) Fetch Links
# 5) Abstract Evidence
# 5.1) Weight Evidence
# 5.99) Scores (Rerank/Stance)
# 6) Filter Irrelevant Evidence
# 7) Determine Truthness
# 8) Final Score

BASE_TEMPERATURE = 0.0
BASE_MODEL = "gemma3:27b"
BASE_WEIGHT = 0.4
BASE_MIN_RELEVANCE = 0.65
BASE_STANCE_NEUTRAL = 0.3
BASE_RETMAX = 10
BASE_THRESHOLD_SCORE = 0.3      # Increased weighting for statements with low score for overall score

# -----------------------------------------------------------------------------
# HINWEISE FÜR evaluation.py
#
# 1) Welche Configs werden ausgeführt?
#    evaluation.py sammelt nur Top-Level Dicts mit "name" + "steps".
#    - Dicts mit "type": "module" (z.B. BASE_RESEARCH_MODULE) werden ignoriert
#      und nie direkt ausgeführt.
#    - Dicts ohne "name" oder ohne "steps" werden ebenfalls ignoriert.
#    => Ausgeführt werden nur die Configs wie RAW_PIPELINE_CONFIG, PUBMED_*_CONFIG.
#
# 2) Multiprocessing/Parallelisierung:
#    evaluation.py entscheidet anhand der verfügbaren GPUs:
#    - Wenn mehrere GPUs + mehrere Configs: ProcessPool (1 Worker pro GPU)
#    - Sonst: sequentiell
#
# 3) Multi‑GPU Ollama (optional):
#    Mit diesen Variablen startet evaluation.py mehrere Ollama‑Instanzen
#    (sofern keine externe LLM_BASE_URL gesetzt ist):
#
#    export CUDA_VISIBLE_DEVICES=0,1,2,3
#    export OLLAMA_MULTI_INSTANCE=1
#    export OLLAMA_MULTI_HOST=127.0.0.1
#    export OLLAMA_MULTI_BASE_PORT=11435
#    python evaluation.py
#
#    Wirkung:
#    - CUDA_VISIBLE_DEVICES: welche GPUs für Worker + Ollama verwendet werden
#    - OLLAMA_MULTI_INSTANCE=1: erlaubt Multi‑Ollama‑Startup pro GPU
#    - OLLAMA_MULTI_HOST/BASE_PORT: startende Instanzen auf
#      http://127.0.0.1:11435, :11436, :11437, ...
#    - Jeder Worker bekommt eine andere LLM_BASE_URL per Prozess‑Env.
# -----------------------------------------------------------------------------




# ============================================================================================================================================================
# ============================================================================================================================================================
#                                                            Base Modules
# ============================================================================================================================================================
# ============================================================================================================================================================

# STEP 3-5.1: The Research Module
BASE_RESEARCH_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! PubMed Research Engine",
        "steps": [
            # Step 3: generate_query
            {
                "type": "generate_query",
                "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_BALANCED, "temperature": BASE_TEMPERATURE},
            },
            # Step 4: fetch_links
            {
                "type": "fetch_links",
                "settings": {"retmax": BASE_RETMAX},
            },
            # Step 5: abstract_evidence
            {
                "type": "abstract_evidence",
                "settings": {},
            },
            # Step 5.1: weight_evidence
            {
                "type": "weight_evidence",
                "settings": {"default_weight": BASE_WEIGHT},
            },
        ],
    },
}

# Step 5.99: Scores Module
BASE_SCORES_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! Scores Engine",
        "steps": [
            # Step 5.99: rerank_evidence
            {
                "type": "rerank_evidence",
                "settings": {"model_name": "BAAI/bge-reranker-v2-m3", "use_fp16": True, "normalize": True, "batch_size": 16, "max_length": 4096, "score_fields": ["abstract"], "empty_relevance": 0.0, "min_relevance": BASE_MIN_RELEVANCE},
            },
        ],
    },
}

# STEP 6-8: The Verification Module
BASE_VERIFICATION_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! Verification Engine",
        "steps": [
            # Step 6: filter_evidence
            {
                "type": "filter_evidence",
                "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE},
            },
            # Step 7: truthness
            {
                "type": "truthness",
                "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE},
            },
            # Step 8: scoring
            {
                "type": "scoring",
                "settings": {"threshold": BASE_THRESHOLD_SCORE},
            },
        ],
    },
}






# ============================================================================================================================================================
# ============================================================================================================================================================
#                                                            Pipeline Configurations
# ============================================================================================================================================================
# ============================================================================================================================================================






RAW_PIPELINE_CONFIG = {
    "name": "Raw_Eval_Pipeline",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7_RAW, "temperature": BASE_TEMPERATURE}},
    ],
}

RAW_ASYMMETRIC_PIPELINE_CONFIG = {
    "name": "Raw_Asymmetric_Eval_Pipeline",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7_ASYMMETRIC_RAW, "temperature": BASE_TEMPERATURE}},
    ],
}


RAW_MEDITRON_PIPELINE_CONFIG = {
    "name": "Raw_Meditron_Eval_Pipeline",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": "hf.co/mradermacher/Meditron3-70B-GGUF:latest", "prompt_template": PROMPT_TMPL_S7_ASYMMETRIC_RAW, "temperature": BASE_TEMPERATURE}},
    ],
}

RAW_MEDITRON_ASYMMETRIC_PIPELINE_CONFIG = {
    "name": "Raw_Meditron_Asymmetric_Eval_Pipeline",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": "hf.co/mradermacher/Meditron3-70B-GGUF:latest", "prompt_template": PROMPT_TMPL_S7_ASYMMETRIC, "temperature": BASE_TEMPERATURE}},
    ],
}




# PubMed single-query variants (BASE_RESEARCH_MODULE prompt swap)
def _research_module_with_prompt(prompt_template) -> dict:
    module = deepcopy(BASE_RESEARCH_MODULE)
    module["settings"]["steps"][0]["settings"]["prompt_template"] = prompt_template
    return module


PUBMED_ATM_ASSISTED_CONFIG = {
    "name": "PubMed_1Query_ATM_Assisted",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (ATM Assisted)
        _research_module_with_prompt(PROMPT_TMPL_S3_ATM_ASSISTED),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
    ],
}

PUBMED_ATM_ASSISTED_COUNTER_CONFIG = {
    "name": "PubMed_1Query_ATM_Assisted_Counter",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (ATM Assisted Counter)
        _research_module_with_prompt(PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
    ],
}

PUBMED_BALANCED_CONFIG = {
    "name": "PubMed_1Query_Balanced",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (Balanced)
        _research_module_with_prompt(PROMPT_TMPL_S3_BALANCED),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
    ],
}

PUBMED_BALANCED_COUNTER_CONFIG = {
    "name": "PubMed_1Query_Balanced_Counter",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (Balanced Counter)
        _research_module_with_prompt(PROMPT_TMPL_S3_BALANCED_COUNTER),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
    ],
}

PUBMED_SPECIFIC_CONFIG = {
    "name": "PubMed_1Query_Specific",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (Specific)
        _research_module_with_prompt(PROMPT_TMPL_S3_SPECIFIC),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
    ],
}

PUBMED_SPECIFIC_COUNTER_CONFIG = {
    "name": "PubMed_1Query_Specific_Counter",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (Specific Counter)
        _research_module_with_prompt(PROMPT_TMPL_S3_SPECIFIC_COUNTER),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
    ],
}








PUBMED_SPECIFIC_COUNTER_ASYMMETRIC_CONFIG = {
    "name": "PubMed_1Query_Specific_Counter_Asymmetric",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (Specific Counter)
        _research_module_with_prompt(PROMPT_TMPL_S3_SPECIFIC_COUNTER),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7_ASYMMETRIC, "temperature": BASE_TEMPERATURE}},
    ],
}


PUBMED_SPECIFIC_COUNTER_MEDITRON_CONFIG = {
    "name": "PubMed_1Query_Specific_Counter_Meditron",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (Specific Counter)
        _research_module_with_prompt(PROMPT_TMPL_S3_SPECIFIC_COUNTER),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": "hf.co/mradermacher/Meditron3-70B-GGUF:latest", "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
    ],
}

PUBMED_SPECIFIC_COUNTER_MEDITRON_ASYMMETRIC_CONFIG = {
    "name": "PubMed_1Query_Specific_Counter_Meditron_Asymmetric",
    "debug": True,
    "steps": [
        # Step 1: mock_statements
        {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
        # Step 3-5.1: BASE_RESEARCH_MODULE (Specific Counter)
        _research_module_with_prompt(PROMPT_TMPL_S3_SPECIFIC_COUNTER),
        # Step 6: filter_evidence
        {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
        # Step 7: truthness
        {"type": "truthness", "settings": {"model": "hf.co/mradermacher/Meditron3-70B-GGUF:latest", "prompt_template": PROMPT_TMPL_S7_ASYMMETRIC, "temperature": BASE_TEMPERATURE}},
    ],
}








# # Config for PubMed search without weights
# PUBMED_PIPELINE_CONFIG_NO_WEIGHTS = {
#     "name": "PubMed_Eval_Pipeline_No_Weights",
#     "debug": True,
#     "steps": [
#         # Step 1: mock_statements
#         {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
#         # Step 3: generate_query (Balanced)
#         {"type": "generate_query", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_BALANCED, "temperature": BASE_TEMPERATURE}},
#         # Step 4: fetch_links
#         {"type": "fetch_links", "settings": {"retmax": BASE_RETMAX}},
#         # Step 5: abstract_evidence
#         {"type": "abstract_evidence", "settings": {}},
#         # Step 6: filter_evidence
#         {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
#         # Step 7: truthness
#         {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
#     ],
# }

















# # Single-pass, low-latency PubMed search (ATM-assisted query)
# PUBMED_LOW_LATENCY_CONFIG = {
#     "name": "PubMed_Low_Latency",
#     "debug": True,
#     "steps": [
#         # Step 1: mock_statements
#         {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
#         # Step 3: generate_query (ATM Assisted)
#         {"type": "generate_query", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED, "temperature": BASE_TEMPERATURE}},
#         # Step 4: fetch_links
#         {"type": "fetch_links", "settings": {"retmax": BASE_RETMAX}},
#         # Step 5: abstract_evidence
#         {"type": "abstract_evidence", "settings": {}},
#         # Step 6: filter_evidence
#         {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
#         # Step 7: truthness
#         {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
#     ],
# }

















# # Dual-query PubMed search (balanced + specific) without rerank
# PUBMED_DUAL_QUERY_CONFIG = {
#     "name": "PubMed_DualQuery_Balanced_Specific",
#     "debug": True,
#     "steps": [
#         # Step 1: mock_statements
#         {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
#         # Step 3: generate_query (Balanced)
#         {"type": "generate_query", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_BALANCED, "temperature": BASE_TEMPERATURE}},
#         # Step 3: generate_query (Specific)
#         {"type": "generate_query", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_SPECIFIC, "temperature": BASE_TEMPERATURE}},
#         # Step 4: fetch_links
#         {"type": "fetch_links", "settings": {"retmax": BASE_RETMAX}},
#         # Step 5: abstract_evidence
#         {"type": "abstract_evidence", "settings": {}},
#         # Step 6: filter_evidence
#         {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
#         # Step 7: truthness
#         {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
#     ],
# }















# # Multi-query PubMed search with filtering
# PUBMED_MULTI_QUERY_CONFIG = {
#     "name": "PubMed_MultiQuery_Filter",
#     "debug": True,
#     "steps": [
#         # Step 1: mock_statements
#         {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
#         # Step 3: generate_query (Balanced)
#         {"type": "generate_query", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_BALANCED, "temperature": BASE_TEMPERATURE}},
#         # Step 3: generate_query (Specific)
#         {"type": "generate_query", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_SPECIFIC, "temperature": BASE_TEMPERATURE}},
#         # Step 3: generate_query (ATM Assisted)
#         {"type": "generate_query", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED, "temperature": BASE_TEMPERATURE}},
#         # Step 4: fetch_links
#         {"type": "fetch_links", "settings": {"retmax": BASE_RETMAX}},
#         # Step 5: abstract_evidence
#         {"type": "abstract_evidence", "settings": {}},
#         # Step 6: filter_evidence
#         {"type": "filter_evidence", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6, "temperature": BASE_TEMPERATURE}},
#         # Step 7: truthness
#         {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
#     ],
# }

















# # Multi-query PubMed with rerank + verification
# PUBMED_MULTI_QUERY_RERANK_CONFIG = {
#     "name": "PubMed_MultiQuery_Rerank_Stance",
#     "debug": True,
#     "steps": [
#         # Step 1: mock_statements
#         {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
#         # Step 3-5.1: BASE_RESEARCH_MODULE
#         BASE_RESEARCH_MODULE,
#         # Step 5.99: BASE_SCORES_MODULE
#         BASE_SCORES_MODULE,
#         # Step 6-8: BASE_VERIFICATION_MODULE
#         BASE_VERIFICATION_MODULE,
#     ],
# }



















# # Variants
# RAW_PIPELINE_CONFIG_REASONING = deepcopy(RAW_PIPELINE_CONFIG)
# RAW_PIPELINE_CONFIG_REASONING["name"] = "Raw_Eval_Pipeline_Reasoning"
# RAW_PIPELINE_CONFIG_REASONING["steps"][-1]["settings"]["model"] = "deepseek-r1:32b"

# PUBMED_MULTI_QUERY_RERANK_MEDITRON_CONFIG = deepcopy(PUBMED_MULTI_QUERY_RERANK_CONFIG)
# PUBMED_MULTI_QUERY_RERANK_MEDITRON_CONFIG["name"] = "PubMed_MultiQuery_Rerank_Meditron"
# PUBMED_MULTI_QUERY_RERANK_MEDITRON_CONFIG["steps"][3]["settings"]["steps"][1]["settings"]["model"] = "meditron3-70b"

# PUBMED_MULTI_QUERY_RERANK_GEMMA12B_CONFIG = deepcopy(PUBMED_MULTI_QUERY_RERANK_CONFIG)
# PUBMED_MULTI_QUERY_RERANK_GEMMA12B_CONFIG["name"] = "PubMed_MultiQuery_Rerank_Gemma3_12B"
# PUBMED_MULTI_QUERY_RERANK_GEMMA12B_CONFIG["steps"][1]["settings"]["steps"][0]["settings"]["model"] = "gemma3:12b"
# PUBMED_MULTI_QUERY_RERANK_GEMMA12B_CONFIG["steps"][3]["settings"]["steps"][0]["settings"]["model"] = "gemma3:12b"
# PUBMED_MULTI_QUERY_RERANK_GEMMA12B_CONFIG["steps"][3]["settings"]["steps"][1]["settings"]["model"] = "gemma3:12b"

# PUBMED_MULTI_QUERY_RERANK_MINREL_0_6_CONFIG = deepcopy(PUBMED_MULTI_QUERY_RERANK_CONFIG)
# PUBMED_MULTI_QUERY_RERANK_MINREL_0_6_CONFIG["name"] = "PubMed_MultiQuery_Rerank_MinRel_0.6"
# PUBMED_MULTI_QUERY_RERANK_MINREL_0_6_CONFIG["steps"][2]["settings"]["steps"][0]["settings"]["min_relevance"] = 0.6






# # PubMed + guideline RAG hybrid
# RAG_HYBRID_CONFIG = {
#     "name": "Guideline_RAG_Hybrid",
#     "debug": True,
#     "steps": [
#         # Step 1: mock_statements
#         {"type": "mock_statements", "settings": {"statements": [{"id": 1, "text": "PLATZHALTER"}]}},
#         # Step 3-5.1: BASE_RESEARCH_MODULE
#         BASE_RESEARCH_MODULE,
#         # Step 5.2: retrieve_guideline_facts
#         {"type": "retrieve_guideline_facts", "settings": {"db_path": "pipeline/RAG_vdb/guidelines_vdb.sqlite", "top_k": 2, "min_score": 0.25}},
#         # Step 5.99: rerank_evidence
#         {"type": "rerank_evidence", "settings": {"min_relevance": BASE_MIN_RELEVANCE, "score_fields": ["abstract"]}},
#         # Step 5.99: stance_evidence
#         {"type": "stance_evidence", "settings": {"evidence_fields": ["abstract"], "threshold_decisive": BASE_STANCE_NEUTRAL}},
#         # Step 7: truthness
#         {"type": "truthness", "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S7, "temperature": BASE_TEMPERATURE}},
#     ],
# }