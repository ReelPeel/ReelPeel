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
                    {
                        "id": 2,
                        "text": "Exclusive breastfeeding is recommended for around the first 6 months of life.",
                    },
                    {
                        "id": 3,
                        "text": "Early introduction of peanut-containing foods can reduce peanut allergy risk.",
                    },
                ]
            },
        },
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
            "settings": {},
        },
        {
            "type": "stance_evidence",
            "settings": {},
        },
    ],
}
