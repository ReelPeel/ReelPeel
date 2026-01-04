from pipeline.test_configs.preprompts import PROMPT_TMPL_RAW, PROMPT_TMPL_S3_BALANCED, PROMPT_TMPL_S6, PROMPT_TMPL_S7
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
