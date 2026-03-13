from pipeline.test_configs.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S7
from pipeline.test_configs.test_extraction import RESEARCH_MODULE, VERIFICATION_MODULE
from pipeline.test_configs.preprompts import PROMPT_TMPL_S3_SPECIFIC, PROMPT_TMPL_S3_BALANCED, PROMPT_TMPL_S3_ATM_ASSISTED

BASE_TEMPERATURE = 0.0

AUDIO_PIPELINE_CONFIG = {
    "name": "Audio_To_Transcript_Run",
    "debug": True,
    "steps": [
        {
            "type": "audio_to_transcript",
            "settings": {
                "audio_path": "audios/audio.wav",
                "whisper_model": "large-v3",
                "translate_non_english": True,
            },
        },
        {
            "type": "extraction",
            "settings": {
                "model": "gemma3:12b",
                "prompt_template": PROMPT_TMPL_S2,
                "temperature": BASE_TEMPERATURE,
            },
        },
        {
            "type": "generate_query",  # Step 3 (multi-prompt)
            "settings": {
                "model": "gemma3:12b",
                "temperature": BASE_TEMPERATURE,
                "prompt_templates": [
                    {"name": "balanced", "template": PROMPT_TMPL_S3_BALANCED},
                    {"name": "specific", "template": PROMPT_TMPL_S3_SPECIFIC},
                    {"name": "atm_assisted", "template": PROMPT_TMPL_S3_ATM_ASSISTED},
                ],
                "parallel": {"enabled": True},
            },
        },
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": 5}
            },
            {
                "type": "abstract_evidence",  # Step 5
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
                    "prompt_template": PROMPT_TMPL_S7,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            # Step 8: Final Score
            {
                "type": "scoring",
                "settings": {
                    "threshold": 0.15
                }
            }
    ],
}


