from pipeline.test_configs.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S3_SPECIFIC, PROMPT_TMPL_S3_SPECIFIC_COUNTER
from pipeline.test_configs.test_extraction import RESEARCH_MODULE
from pipeline.test_configs.kai_test import SCORES_MODULE


VIDEO_PIPELINE_CONFIG = {
    "name": "Video_To_Audio_Run",
    "debug": True,
    "steps": [
        {
            "type": "video_to_audio",
            "settings": {
                "video_path": "downloads/downloads_vitamind/C4dki_DoayB.mp4",
            },
        },
        {
            "type": "audio_to_transcript",
            "settings": {
                "whisper_model": "turbo",
                "translate_non_english": True,
            },
        },
        {
            "type": "extraction",
            "settings": {
                "model": "gemma3:27b",
                "prompt_template": PROMPT_TMPL_S2,
                "temperature": 0.0,
            },
        },
        RESEARCH_MODULE,
        SCORES_MODULE, 
        {
            "type": "retrieve_guideline_facts",
            "settings": {
                "db_path": "pipeline/RAG_vdb/guidelines_vdb.sqlite",
                "top_k": 5,
                "min_score": 0.25,
            },
        },
        {
            "type": "rerank_evidence",
            "settings": {
                    "model_name": "BAAI/bge-reranker-v2-m3", # BAAI/bge-reranker-v2-gemma bigger but slower
                    "use_fp16": True,
                    "normalize": True,
                    "batch_size": 16,
                    "max_length": 4096,
                    "score_fields": ["abstract"],
                    "empty_relevance": 0.0,
                    "min_relevance": 0.5,
                },
            },
            {
                "type": "stance_evidence",
                "settings": {
                    "model_name": "cnut1648/biolinkbert-mednli",
                    "use_fp16": True,
                    "batch_size": 16,
                    "max_length": 512,
                    "evidence_fields": ["abstract"],

                    # optional: only compute stance on Top-M evidence per statement (by ev.relevance)
                    # "top_m_by_relevance": 5,

                    # optional: if both support/refute are weak, force "neutral"
                    "threshold_decisive": 0.3,
                }
                },
        
        
            
    ],
}


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
                "whisper_model": "turbo",
                "translate_non_english": True,
            },
        },
        {
            "type": "extraction",
            "settings": {
                "model": "gemma3:27b",
                "prompt_template": PROMPT_TMPL_S2,
                "temperature": 0.0,
            },
        },
        
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC
                }
            },
        {
                "type": "generate_query",  # Step 3
                "settings": {
                    "model": "gemma3:27b",
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC_COUNTER
                }
            },
           
       
            {
                "type": "fetch_links",  # Step 4
                "settings": {"retmax": 10}
            },
            {
                "type": "abstract_evidence",  # Step 5
                "settings": {}
            },
            {
                "type": "weight_evidence",  # Step 5.1
                "settings": {"default_weight": 0.15}
            },
            SCORES_MODULE,      
          
    ],
}
