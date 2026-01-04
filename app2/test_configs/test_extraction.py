from app2.core.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S3_NARROW_QUERY, PROMPT_TMPL_S5, PROMPT_TMPL_S6, PROMPT_TMPL_S7

# 1. Define the Research Module (Steps 3, 4, 5, 5.1)
RESEARCH_MODULE = {
    "type": "module",
    "settings": {
        "name": "PubMed Research Engine",
        "steps": [
            {
                "type": "generate_query",  # Step 3
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S3_NARROW_QUERY
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
            {
                "type": "weight_evidence",  # Step 5.1
                "settings": {"default_weight": 0.5}
            }
        ]
    }
}

VERIFICATION_MODULE = {
    "type": "module",
    "settings": {
        "name": "Verification Engine",
        "debug": True,
        "steps": [
            # Step 6: Filter Irrelevant Evidence
            {
                "type": "filter_evidence",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S6
                }
            },
            # Step 7: Determine Truthness
            {
                "type": "truthness",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S7
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
    "name": "Full_End_to_End_Run",
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
                "prompt_template": PROMPT_TMPL_S2
            }
        },

        # STEP 3-5.1: The Research Module
        RESEARCH_MODULE,
        # STEP 6-8: The Verification Module
        VERIFICATION_MODULE
    ]
}
