from pipeline.test_configs.preprompts import PROMPT_TMPL_S2, PROMPT_TMPL_S7
from pipeline.test_configs.test_extraction import RESEARCH_MODULE, SCORES_MODULE, VERIFICATION_MODULE


VIDEO_PIPELINE_CONFIG = {
    "name": "Video_To_Audio_Run",
    "debug": True,
    "steps": [
        {
            "type": "video_to_audio",
            "settings": {
                "video_path": "downloads/DFfQnWiMjtl.mp4",
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
                "model": "gemma3:12b",
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
                "top_k": 3,
                "min_score": 0.25,
            },
        },
        {
            "type": "rerank_evidence",
            "settings": {},
        },
        {
            "type": "stance_evidence",
            "settings": {},
        },
        {
                "type": "truthness",
                "settings": {
                    "model": "gemma3:12b",
                    "prompt_template": PROMPT_TMPL_S7
                }
            },       
        VERIFICATION_MODULE,

    ],
}
