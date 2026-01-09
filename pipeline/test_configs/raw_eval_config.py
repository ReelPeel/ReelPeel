from pipeline.test_configs.preprompts import (
    PROMPT_TMPL_RAW,
    PROMPT_TMPL_S3_ATM_ASSISTED,
    PROMPT_TMPL_S3_BALANCED,
    PROMPT_TMPL_S3_SPECIFIC,
    PROMPT_TMPL_S6,
    PROMPT_TMPL_S7,
)
from .test_extraction import (
    RESEARCH_MODULE as MQ_RESEARCH_MODULE,
    SCORES_MODULE as MQ_SCORES_MODULE,
    VERIFICATION_MODULE as MQ_VERIFICATION_MODULE,
)
from .test_extraction import RESEARCH_MODULE

RAW_PIPELINE_CONFIG = {
    "name": "Raw_Eval_Pipeline",
    "debug": True,
    "steps": [

        # STEP 1: Mock Input (Simulating Whisper)
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {
                        "id": 1, 
                        "text": "Drinking 3 liters of water daily cures kidney stones."
                    },
                ]
            }
        },
        # Step 7: Determine Truthness
            {
                "type": "truthness",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_RAW
                }
            },

    ]
}
# Config for pubmed search
PUBMED_PIPELINE_CONFIG = {
    "name": "PubMed_Eval_Pipeline",
    "debug": True,
    "steps": [
        
        # STEP 1: Mock Input (Simulating Whisper)
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {
                        "id": 1, 
                        "text": "Drinking 3 liters of water daily cures kidney stones."
                    },
                ]
            }
        },

        # STEP 3-5.1: The Research Module
        RESEARCH_MODULE,
        # Step 6: Filter Irrelevant Evidence
            {
                "type": "filter_evidence",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_S6
                }
            },
        # Step 7: Determine Truthness
            {
                "type": "truthness",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_S7
                }
            },
    ]
}

# Config for pubmed search without weights
PUBMED_PIPELINE_CONFIG_NO_WEIGHTS = {
    "name": "PubMed_Eval_Pipeline_No_Weights",
    "debug": True,
    "steps": [
        
        # STEP 1: Mock Input (Simulating Whisper)
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {
                        "id": 1, 
                        "text": "Drinking 3 liters of water daily cures kidney stones."
                    },
                ]
            }
        },

        # STEP 3-5.1: The Research Module
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_S3_BALANCED
                }
            },
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": 5}
            },
            {
                "type": "summarize_evidence",  # Step 5
                "settings": {}
            },
        # Step 6: Filter Irrelevant Evidence
            {
                "type": "filter_evidence",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_S6
                }
            },
        # Step 7: Determine Truthness
            {
                "type": "truthness",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_S7
                }
            },
    ]
}

# Single-pass, low-latency PubMed search (ATM-assisted query)
PUBMED_LOW_LATENCY_CONFIG = {
    "name": "PubMed_Low_Latency",
    "debug": True,
    "steps": [
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {"id": 1, "text": "Drinking 3 liters of water daily cures kidney stones."}
                ]
            },
        },
        {
            "type": "generate_query",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED,
                "temperature": 0.0,
            },
        },
        {"type": "fetch_links", "settings": {"retmax": 3}},
        {"type": "summarize_evidence", "settings": {}},
        {
            "type": "filter_evidence",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S6,
                "temperature": 0.0,
            },
        },
        {
            "type": "truthness",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S7,
                "temperature": 0.0,
            },
        },
    ],
}

# Dual-query PubMed search (balanced + specific) without rerank
PUBMED_DUAL_QUERY_CONFIG = {
    "name": "PubMed_DualQuery_Balanced_Specific",
    "debug": True,
    "steps": [
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {"id": 1, "text": "Drinking 3 liters of water daily cures kidney stones."}
                ]
            },
        },
        {
            "type": "generate_query",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S3_BALANCED,
                "temperature": 0.0,
            },
        },
        {
            "type": "generate_query",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S3_SPECIFIC,
                "temperature": 0.0,
            },
        },
        {"type": "fetch_links", "settings": {"retmax": 5}},
        {"type": "summarize_evidence", "settings": {}},
        {
            "type": "filter_evidence",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S6,
                "temperature": 0.0,
            },
        },
        {
            "type": "truthness",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S7,
                "temperature": 0.0,
            },
        },
    ],
}

# Multi-query PubMed search with filtering
PUBMED_MULTI_QUERY_CONFIG = {
    "name": "PubMed_MultiQuery_Filter",
    "debug": True,
    "steps": [
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {"id": 1, "text": "Drinking 3 liters of water daily cures kidney stones."}
                ]
            },
        },
        MQ_RESEARCH_MODULE,
        {
            "type": "filter_evidence",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S6,
                "temperature": 0.0,
            },
        },
        {
            "type": "truthness",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S7,
                "temperature": 0.0,
            },
        },
    ],
}

# Multi-query PubMed with rerank + stance + verification
PUBMED_MULTI_QUERY_RERANK_CONFIG = {
    "name": "PubMed_MultiQuery_Rerank_Stance",
    "debug": True,
    "steps": [
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {"id": 1, "text": "Drinking 3 liters of water daily cures kidney stones."}
                ]
            },
        },
        MQ_RESEARCH_MODULE,
        MQ_SCORES_MODULE,
        MQ_VERIFICATION_MODULE,
    ],
}

# PubMed + guideline RAG hybrid
RAG_HYBRID_CONFIG = {
    "name": "Guideline_RAG_Hybrid",
    "debug": True,
    "steps": [
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {"id": 1, "text": "Healthy term infants should receive 400 IU of vitamin D daily."}
                ]
            },
        },
        RESEARCH_MODULE,
        {
            "type": "retrieve_guideline_facts",
            "settings": {
                "db_path": "pipeline/RAG_vdb/guidelines_vdb.sqlite",
                "top_k": 2,
                "min_score": 0.25,
            },
        },
        {"type": "rerank_evidence", "settings": {}},
        {"type": "stance_evidence", "settings": {}},
        {
            "type": "truthness",
            "settings": {
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S7,
            },
        },
    ],
}
