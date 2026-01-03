from app.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S3, PROMPT_TMPL_S5

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
                    "prompt_template": PROMPT_TMPL_S3
                }
            },
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": 3}
            },
            {
                "type": "summarize_evidence",  # Step 5
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S5
                }
            },
            {
                "type": "weight_evidence",  # Step 5.1
                "settings": {"default_weight": 0.5}
            }
        ]
    }
}

# 2. Define the Full Pipeline Config
FULL_PIPELINE_CONFIG = {
    "name": "Full_End_to_End_Run",
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
        RESEARCH_MODULE
    ]
}