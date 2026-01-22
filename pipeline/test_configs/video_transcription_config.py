import copy

from pipeline.test_configs.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER, PROMPT_TMPL_S3_BALANCED_COUNTER, PROMPT_TMPL_S3_HIGHLY_SPECIFIC, PROMPT_TMPL_S3_HIGHLY_SPECIFIC_COUNTER, PROMPT_TMPL_S3_SPECIFIC_COUNTER, PROMPT_TMPL_S7, PROMPT_TMPL_S7_ACTIONABLE_ADVICE, PROMPT_TMPL_S7_ACTIONABLE_ADVICE_V2, PROMPT_TMPL_S7_METRICS
from pipeline.test_configs.test_extraction import RESEARCH_MODULE, VERIFICATION_MODULE
from pipeline.test_configs.kai_test import SCORES_MODULE
from pipeline.test_configs.preprompts import PROMPT_TMPL_S3_SPECIFIC, PROMPT_TMPL_S3_BALANCED, PROMPT_TMPL_S3_ATM_ASSISTED

BASE_TEMPERATURE = 0.3
SCORES_MIN_RELEVANCE = 0.7
BASE_MODEL="gemma3:27b" 
# gemma3:27b
# hf.co/mradermacher/medgemma-27b-text-it-GGUF:Q4_K_M
# hf.co/mradermacher/DeepSeek-R1-Distill-Qwen-32B-Medical-GGUF:Q6_K

STEP_7_MODEL = BASE_MODEL
STEP_7_PROMPT = PROMPT_TMPL_S7_ACTIONABLE_ADVICE_V2
# PROMPT_TMPL_S7
# PROMPT_TMPL_S7_METRICS
# PROMPT_TMPL_S7_ACTIONABLE_ADVICE
# PROMPT_TMPL_S7_ACTIONABLE_ADVICE_V2
INCLUDE_EVIDENCE_TEXT = True



SCORES_MODULE_MIN_REL = copy.deepcopy(SCORES_MODULE)
SCORES_MODULE_MIN_REL["settings"]["steps"][0]["settings"]["min_relevance"] = SCORES_MIN_RELEVANCE

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
                "whisper_model": "large-v3",
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
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_BALANCED,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_BALANCED_COUNTER,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            
        # {
        #         "type": "generate_query",  # Step 3
        #         "settings": {
        #             "model": BASE_MODEL,
        #             "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED,
        #             "temperature": BASE_TEMPERATURE,
        #         }
        #     },
        # {
        #         "type": "generate_query",  # Step 3
        #         "settings": {
        #             "model": BASE_MODEL,
        #             "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER,
        #             "temperature": BASE_TEMPERATURE,
        #         }
        #     },
       
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC,
                    "temperature": BASE_TEMPERATURE,
                }
            },
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC_COUNTER,
                    "temperature": BASE_TEMPERATURE,
                }
            },
        
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": 20}
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
                    "include_evidence_text": INCLUDE_EVIDENCE_TEXT,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            # Step 8: Final Score
            {
                "type": "scoring",
                "settings": {
                    "threshold": 0.3
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
                "whisper_model": "large-v3",
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
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_BALANCED,
                    "temperature": BASE_TEMPERATURE,
                }
            },
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC,
                    "temperature": BASE_TEMPERATURE,
                }
            },
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC_COUNTER,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_BALANCED_COUNTER,
                    "temperature": BASE_TEMPERATURE,
                }
            },
       
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": 20}
            },
            {
                "type": "abstract_evidence",  # Step 5
                "settings": {}
            },
            {
                "type": "weight_evidence",  # Step 5.1
                "settings": {"default_weight": 0.15}
            },
            SCORES_MIN_RELEVANCE,
        {
                "type": "truthness",
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S7,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            # Step 8: Final Score
            {
                "type": "scoring",
                "settings": {
                    "threshold": 0.3
                }
            }
    ],
}
