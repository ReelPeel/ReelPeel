import copy

from pipeline.test_configs.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER, PROMPT_TMPL_S3_BALANCED_COUNTER, PROMPT_TMPL_S3_HIGHLY_SPECIFIC, PROMPT_TMPL_S3_HIGHLY_SPECIFIC_COUNTER, PROMPT_TMPL_S3_SPECIFIC_COUNTER, PROMPT_TMPL_S6, PROMPT_TMPL_S7, PROMPT_TMPL_S7_ACTIONABLE_ADVICE, PROMPT_TMPL_S7_ACTIONABLE_ADVICE_V2, PROMPT_TMPL_S7_METRICS
from pipeline.test_configs.test_extraction import RESEARCH_MODULE, VERIFICATION_MODULE
from pipeline.test_configs.kai_test import SCORES_MODULE
from pipeline.test_configs.preprompts import PROMPT_TMPL_S3_SPECIFIC, PROMPT_TMPL_S3_BALANCED, PROMPT_TMPL_S3_ATM_ASSISTED

 

BASE_TEMPERATURE = 0.1
SCORES_MIN_RELEVANCE = 0.1
BASE_MODEL="gemma3:27b" 
# gemma3:27b / 12b
# hf.co/mradermacher/medgemma-27b-text-it-GGUF:Q4_K_M
# hf.co/mradermacher/DeepSeek-R1-Distill-Qwen-32B-Medical-GGUF:Q6_K

WHISPER_MODEL = "tiny.en"
# large-v3
# turbo
# tiny

STEP_3_MODEL = "gemma3:12b" # Query Generation Model
RETMAX = 10  # Number of articles per query

FILTER_VERIFICATION_MODULE_ENABLED = False

STEP_7_MODEL = "gemma3:27b" # Final Truthness Model
STEP_7_PROMPT = PROMPT_TMPL_S7_ACTIONABLE_ADVICE
# PROMPT_TMPL_S7
# PROMPT_TMPL_S7_METRICS
# PROMPT_TMPL_S7_ACTIONABLE_ADVICE
# PROMPT_TMPL_S7_ACTIONABLE_ADVICE_V2
INCLUDE_EVIDENCE_TEXT = True



SCORES_MODULE_MIN_REL = copy.deepcopy(SCORES_MODULE)
SCORES_MODULE_MIN_REL["settings"]["steps"][0]["settings"]["min_relevance"] = SCORES_MIN_RELEVANCE




# -----------------------------------------------------------------------------
# Test config
# -----------------------------------------------------------------------------

VIDEO_PIPELINE_CONFIG = {
    "name": "Video_To_Audio_Run",
    "debug": True,
    "steps": [
        {
            "type": "video_to_audio",
            "settings": {
                "video_path": "Jana.mp4",
            },
        },
        {
            "type": "audio_to_transcript",
            "settings": {
                "whisper_model": WHISPER_MODEL,
                "translate_non_english": True,
            },
        },
        # {
        #     "type": "mock_transcript",
        #     "settings": {
        #         "transcript_text": (
        #             "Okay, these are three health hacks I wish I knew earlier. Save this for your read. Vitamin C every day basically stops you from catching colds, especially in the winter. There is something doctors won't tell you, but if you ever feel bloated or toxic, a detox tea can clear out your liver in like 24 hours. You'll thank me the next day. Also, this might sound boring, but washing your hands for 25 seconds makes all the difference. So try out these hacks and comment down below if they helped you."
        #         )
        #         # "transcript_text": (    #ORIGINALES TRANSCRIPT MIT FEHLERN (VITAMIN C -> COMING TO SEA)
        #         # "Okay, these are three health hacks I wish I knew earlier. Save this for your read. Vitamin C every day basically stops you from catching colds, especially in the winter. There is something doctors won't tell you, but if you ever feel bloated or toxic, a detox tea can clear out your liver in like 24 hours. You'll thank me the next day. Also, this might sound boring, but washing your hands for 25 seconds makes all the difference. So try out these hacks and comment down below if they helped you.",
        #         # )
        #     }
        # },
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
                    # {"name": "balanced", "template": PROMPT_TMPL_S3_BALANCED},
                    # {"name": "balanced_counter", "template": PROMPT_TMPL_S3_BALANCED_COUNTER},
                    {"name": "specific", "template": PROMPT_TMPL_S3_SPECIFIC},
                    {"name": "specific_counter", "template": PROMPT_TMPL_S3_SPECIFIC_COUNTER},
                    # {"name": "atm_assisted", "template": PROMPT_TMPL_S3_ATM_ASSISTED},
                    # {"name": "atm_assisted_counter", "template": PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER},
                ],
                "parallel": {"enabled": True},
                "prefetch_links": {"enabled": True, "retmax": RETMAX},
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
        *(
            [{
                "type": "filter_evidence",
                "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6},
            }]
            if FILTER_VERIFICATION_MODULE_ENABLED
            else []
        ),
        {
                "type": "truthness",
                "settings": {
                    "model": STEP_7_MODEL,
                    "prompt_template": STEP_7_PROMPT,
                    "include_evidence_text": INCLUDE_EVIDENCE_TEXT,
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



# {
        #     "type": "retrieve_guideline_facts",
        #     "settings": {
        #         "db_path": "pipeline/RAG_vdb/guidelines_vdb.sqlite",
        #         "top_k": 5,
        #         "min_score": 0.25,
        #     },
        # },





# -----------------------------------------------------------------------------
# APP Config
# -----------------------------------------------------------------------------

VIDEO_URL_PIPELINE_CONFIG = {
    "name": "Video_URL_End_to_End_Run",
    "debug": True,
    "steps": [
        {
            "type": "download_reel",
            "settings": {
                "video_url": "",
                "output_dir": "temp",
            },
        },
        {
            "type": "video_to_audio",
            "settings": {},
        },
        {
            "type": "audio_to_transcript",
            "settings": {
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
                    # {"name": "balanced", "template": PROMPT_TMPL_S3_BALANCED},
                    # {"name": "balanced_counter", "template": PROMPT_TMPL_S3_BALANCED_COUNTER},
                    {"name": "specific", "template": PROMPT_TMPL_S3_SPECIFIC},
                    {"name": "specific_counter", "template": PROMPT_TMPL_S3_SPECIFIC_COUNTER},
                    # {"name": "atm_assisted", "template": PROMPT_TMPL_S3_ATM_ASSISTED},
                    # {"name": "atm_assisted_counter", "template": PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER},
                ],
                "parallel": {"enabled": True},
                "prefetch_links": {"enabled": True, "retmax": RETMAX},
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
        *(
            [{
                "type": "filter_evidence",
                "settings": {"model": BASE_MODEL, "prompt_template": PROMPT_TMPL_S6},
            }]
            if FILTER_VERIFICATION_MODULE_ENABLED
            else []
        ),
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
