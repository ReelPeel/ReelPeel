from app2.core.preprompts import PROMPT_TMPL_S7

RAW_PIPELINE_CONFIG = {
    "name": "Full_End_to_End_Run",
    "debug": True,
    "steps": [
        # Step 7: Determine Truthness
            {
                "type": "truthness",
                "settings": {
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S7
                }
            },

    ]
}