# 2. Define the Full Pipeline Config
from pipeline.test_configs.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S3_SPECIFIC, PROMPT_TMPL_S3_BALANCED, PROMPT_TMPL_S3_ATM_ASSISTED, PROMPT_TMPL_S7

FULL_PIPELINE_CONFIG = {
    "name": "Full_End_to_End_Run",
    "debug": True,
    "llm_settings" : {
        "base_url": "http://localhost:11434/v1",
        "api_key" : "ollama",
    },
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
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S2
            }
        },
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S3_BALANCED
                }
            },
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC
                }
            },
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED
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
            },
        {
                "type": "truthness",
                "settings": {
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