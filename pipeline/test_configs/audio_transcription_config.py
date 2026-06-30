import copy

from pipeline.test_configs.preprompts import (
    PROMPT_TMPL_S2,
    PROMPT_TMPL_S3_SPECIFIC,
    PROMPT_TMPL_S3_SPECIFIC_COUNTER,
    PROMPT_TMPL_S7_ACTIONABLE_ADVICE,
)
from pipeline.test_configs.test_extraction import RESEARCH_MODULE, VERIFICATION_MODULE
from pipeline.test_configs.kai_test import SCORES_MODULE

BASE_TEMPERATURE = 0.0
SCORES_MIN_RELEVANCE = 0.7
BASE_MODEL = "gemma3:27b"
WHISPER_MODEL = "turbo"
STEP_3_MODEL = "gemma3:12b"
RETMAX = 10
STEP_7_MODEL = "gemma3:27b"
STEP_7_PROMPT = PROMPT_TMPL_S7_ACTIONABLE_ADVICE

SCORES_MODULE_MIN_REL = copy.deepcopy(SCORES_MODULE)
SCORES_MODULE_MIN_REL["settings"]["steps"][0]["settings"]["min_relevance"] = SCORES_MIN_RELEVANCE
SCORES_MODULE_MIN_REL["settings"]["steps"][1]["settings"]["section_chunking_enabled"] = True
SCORES_MODULE_MIN_REL["settings"]["steps"][1]["settings"]["diagnostic_test_gate_enabled"] = True

AUDIO_PIPELINE_CONFIG = {
    "name": "Audio_To_Transcript_Run",
    "debug": True,
    "steps": [
        {
            "type": "audio_to_transcript",
            "settings": {
                "audio_path": "audios/audio.wav",
                "whisper_model": WHISPER_MODEL,
                "translate_non_english": True,
            },
        },
        {
            "type": "extraction",
            "settings": {
                "model": BASE_MODEL,
                "prompt_template": PROMPT_TMPL_S2,
                "temperature": BASE_TEMPERATURE,
            },
        },
        {
            "type": "generate_query",  # Step 3 (multi-prompt)
            "settings": {
                "model": STEP_3_MODEL,
                "temperature": BASE_TEMPERATURE,
                "prompt_templates": [
                    {"name": "specific", "template": PROMPT_TMPL_S3_SPECIFIC},
                    {"name": "specific_counter", "template": PROMPT_TMPL_S3_SPECIFIC_COUNTER},
                ],
                "parallel": {"enabled": True},
                "prefetch_links": {"enabled": True, "retmax": RETMAX, "prefetch_abstracts": True},
            },
        },
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": RETMAX}
            },
            {
                "type": "abstract_evidence",  # Step 5
                "settings": {}
            },
            {
                "type": "weight_evidence",  # Step 5.1
                "settings": {"default_weight": 0.15}
            },
            SCORES_MODULE_MIN_REL,
        {
                "type": "truthness",
                "settings": {
                    "model": STEP_7_MODEL,
                    "prompt_template": STEP_7_PROMPT,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            # Step 8: Final Score
            {
                "type": "scoring",
                "settings": {
                    "threshold": 0.4
                }
            }
    ],
}
