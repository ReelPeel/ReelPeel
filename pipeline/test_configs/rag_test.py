from .test_extraction import RESEARCH_MODULE, VERIFICATION_MODULE, PROMPT_TMPL_S7

RAG_TEST_CONFIG = {
    "name": "Guideline_RAG_Test",
    "debug": True,
    "steps": [
        {
            "type": "mock_statements",
            "settings": {
                "statements": [
                    {
                        "id": 1,
                        "text": "Healthy term infants should receive 400 IU of vitamin D daily.",
                    },
                    
                ]
            },
        },
        
         RESEARCH_MODULE,
        
        {
            "type": "retrieve_guideline_facts",
            "settings": {
                "db_path": "pipeline/RAG_vdb/guidelines_vdb.sqlite",
                "top_k": 1,
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
    ],
}
