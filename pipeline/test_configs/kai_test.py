from pipeline.test_configs.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S3, PROMPT_TMPL_S6, PROMPT_TMPL_S7

# 1. Define the Research Module (Steps 3, 4, 5, 5.1)
RESEARCH_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! PubMed Research Engine",
        "steps": [
            {
                "type": "generate_query",  # Step 3
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S3,
                    "temperature": 0.0,
                    # "max_tokens": 512 # Currently hardcoded in individual step
                }
            },
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": 3}
            },
            {
                "type": "summarize_evidence",  # Step 5
                "settings": {}
            },
            {
                "type": "weight_evidence",  # Step 5.1
                "settings": {"default_weight": 0.5}
            }
        ]
    }
}


SCORES_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! Scores Engine",
        "steps": [
            {
                "type": "rerank_evidence",
                "settings": {
                    "model_name": "BAAI/bge-reranker-v2-m3", # BAAI/bge-reranker-v2-gemma bigger but slower
                    "use_fp16": True,
                    "normalize": True,
                    "batch_size": 16,
                    "max_length": 4096,
                    "score_fields": ["abstract", "summary"],
                    "empty_relevance": 0.0,
                },
            },
            {
                "type": "stance_evidence",
                "settings": {
                    "model_name": "cnut1648/biolinkbert-mednli",
                    "use_fp16": True,
                    "batch_size": 16,
                    "max_length": 512,
                    "evidence_fields": ["abstract", "summary"],

                    # optional: only compute stance on Top-M evidence per statement (by ev.relevance)
                    # "top_m_by_relevance": 5,

                    # optional: if both support/refute are weak, force "neutral"
                    "threshold_decisive": 0.2,
                }
                },
            # {"type": "similarity_penalty", "settings": {...}},
        ]
    }
}


VERIFICATION_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! Verification Engine",
        "debug": True,
        "steps": [
            # Step 6: Filter Irrelevant Evidence
            {
                "type": "filter_evidence",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S6,
                    "temperature": 0.0,
                    # "max_tokens": 512 # Currently hardcoded in individual step
                }
            },
            # Step 7: Determine Truthness
            {
                "type": "truthness",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S7,
                    "temperature": 0.0,
                    # "max_tokens": 512 # Currently hardcoded in individual step
                }
            },
            # Step 8: Final Score
            {
                "type": "scoring",
                "settings": {
                    "threshold": 0.15
                }
            }
        ]
    }
}

# 2. Define the Full Pipeline Config
FULL_PIPELINE_CONFIG = {
    "name": "MODULE! Full_End_to_End_Run",
    "debug": True,
    "steps": [
        # STEP 1: Mock Input (Simulating Whisper)
        {
            "type": "mock_transcript",
            "settings": {
                "transcript_text": (
                    "Fasting for 72 hours triggers autophagy and renews the immune system. "
                    "Also, drinking celery juice every morning cures all inflammation."
                )
            }
        },

        # STEP 2: Extraction
        {
            "type": "extraction",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S2,
                "temperature": 0.0,
                # "max_tokens": 512 # Currently hardcoded in individual step
            }
        },

        # STEP 3-5.1: The Research Module
        RESEARCH_MODULE,
        
        # Step 5.99: Scores Module
        SCORES_MODULE,
        
        # STEP 6-8: The Verification Module
        VERIFICATION_MODULE
    ]
}
