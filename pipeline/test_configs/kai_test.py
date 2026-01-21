from pipeline.test_configs.preprompts import (
    PROMPT_TMPL_S2, 
    PROMPT_TMPL_S6, 
    PROMPT_TMPL_S7,
    PROMPT_TMPL_S3_ATM_ASSISTED,
    PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER,
    PROMPT_TMPL_S3_BALANCED,
    PROMPT_TMPL_S3_BALANCED_COUNTER,
    PROMPT_TMPL_S3_SPECIFIC,
    PROMPT_TMPL_S3_SPECIFIC_COUNTER,
)

BASE_TEMPERATURE = 0.0
BASE_MODEL = "gemma3:27b"

# 1. Define the Research Module (Steps 3, 4, 5, 5.1)
RESEARCH_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! PubMed Research Engine",
        "steps": [
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
                    "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            {
                "type": "generate_query",  # Step 3 (counter-evidence)
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_BALANCED_COUNTER,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            {
                "type": "generate_query",  # Step 3 (counter-evidence)
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_SPECIFIC_COUNTER,
                    "temperature": BASE_TEMPERATURE,
                }
            },
            {
                "type": "generate_query",  # Step 3 (counter-evidence)
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S3_ATM_ASSISTED_COUNTER,
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
                "settings": {"default_weight": 0.5}
            }
        ]
    }
}


SCORES_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! Scores Engine",
        "steps": [
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
                    "min_relevance": 0.6,
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
            # {"type": "similarity_penalty", "settings": {...}},
        ]
    }
}


VERIFICATION_MODULE = {
    "type": "module",
    "settings": {
        "name": "MODULE! Verification Engine",
        "debug": True,
        "steps": [
            # Step 6: Filter Irrelevant Evidence
            {
                "type": "filter_evidence",
                "settings": {
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S6,
                    "temperature": BASE_TEMPERATURE,
                    # "max_tokens": 512 # Currently hardcoded in individual step
                }
            },
            # Step 7: Determine Truthness
            {
                "type": "truthness",
                "settings": {
# ---------------------- Instruct tuned, Medical Domain Models ----------------------
# - `hf.co/mradermacher/Llama3-OpenBioLLM-70B-i1-GGUF:Llama3-OpenBioLLM-70B.i1-IQ4_XS.gguf` (~37 GB): Llama 3 biomedical finetune; strong medical vocabulary and domain reasoning; i1 (imatrix) GGUF IQ4_XS to fit single A100 40GB without sharding.
# - `hf.co/mradermacher/Llama3-Med42-70B-i1-GGUF:Llama3-Med42-70B.i1-IQ4_XS.gguf` (~37 GB): Med42 biomedical/clinical finetune; strong evidence-style reasoning; i1 (imatrix) GGUF IQ4_XS to fit single A100 40GB without sharding.

# ------------------------------ Foundation, Medical Models ----------------------
# - `hf.co/mradermacher/Meditron3-70B-GGUF:latest` (~38 GB): Meditron3 70B medical foundation model (not instruction-tuned); good for clinical-style summaries and evidence synthesis; GGUF format.

# ------------------------------ Reasoning, Medical Models ----------------------
# - `hf.co/mradermacher/DeepSeek-R1-Distill-Qwen-32B-Medical-GGUF:Q8_0` (~34 GB): distilled (R1) Qwen-based medical model; strong analytical reasoning; higher-fidelity Q8 quantization with comfortable VRAM headroom on A100 40GB.

# ---------------------- Instruct tuned, General Domain Models ----------------------
# - `gemma3:12b` (`gemma3:12b-it-q4_K_M`) (~8.1 GB): instruction-tuned general model; fast and light, good for quick extraction/filtering.
# - `gemma3:27b` (`gemma3:27b-it-q4_K_M`) (~17 GB): instruction-tuned mid-size model; stronger reasoning than 12b with moderate VRAM cost.
                    "model": BASE_MODEL,
                    "prompt_template": PROMPT_TMPL_S7,
                    "temperature": BASE_TEMPERATURE,
                    # "max_tokens": 512 # Currently hardcoded in individual step
                }
            },
            # Step 8: Final Score
            {
                "type": "scoring",
                "settings": {
                    "threshold": 0.15
                }
            }
        ]
    }
}

# 2. Define the Full Pipeline Config
FULL_PIPELINE_CONFIG = {
    "name": "MODULE! Full_End_to_End_Run",
    "debug": True,
    "steps": [
        # STEP 1: Mock Input (Simulating Whisper)
        {
            "type": "mock_transcript",
            "settings": {
                "transcript_text": (
                    # "Fasting for 72 hours triggers autophagy and renews the immune system. "
                    # "Also, drinking celery juice every morning cures all inflammation."
                    
                    # Transkript von: C0hXZ3bNAbH
                    """Was gibst du deinem Baby da? Aber Babys dürfen doch im ersten Lebensjahr keine Eier und keine Kuhmilchprodukte konsumieren. Diese Empfehlung hält sich
                    hartnäckig, ist aber mittlerweile veraltet. Die aktuelle Empfehlung der Leitlinie zur Allergieprävention lautet, nach neuesten wissenschaftlichen
                    Erkenntnissen gibt es in Bezug auf Allergieprävention keine Einschränkung in der Lebensmittelauswahl mehr. Ein verzögertes Einführen von potent
                    allergenen Lebensmitteln kann nicht mehr empfohlen werden. Und eine Restriktion hat auch keine positiven Auswirkungen hinsichtlich der
                    Allergieprävention. Einfach gesagt bedeutet dies, biete Allergene früh und häufig an. Aber was sind denn überhaupt die häufigsten Allergene? Dies sind
                    Milchprodukte, Ei, Erdnüsse, Baumnüsse, Soja, Weizen, Fisch und Meeresfrüchte. Diese sollten spätestens ab dem siebten Lebensmonat eingeführt werden.
                    Jedes Allergen sollte einzeln und in kleinen Mengen eingeführt werden, um potenzielle Reaktionen genau beobachten zu können. Im ersten Lebensjahr
                    dürfen Babys bis zu 200 ml verarbeitete Milchprodukte am Tag konsumieren. Diese dürfen jedoch nicht als Ersatz für Muttermilch oder Formular gegeben
                    werden. Im ersten Lebensjahr dürfen Babys nur ein bis zwei Eier die Woche konsumieren. Dies liegt am hohen Proteingehalt, der die unreifen Nieren
                    belastet und den Wasserhaushalt durcheinander bringt. Die Wahl des Zeitpunktes ist auch entscheidend. Die Einführung neuer Lebensmittel sollte
                    vorzugsweise am Morgen erfolgen, um mögliche Reaktionen während der nächsten Stunden genau überwachen zu können. Insgesamt ist es aber so, dass
                    Lebensmittelallergien nicht zwangsläufig bei der ersten Gabe auftreten, sondern sich eher über Zeit bilden können. Falls ihr selbst oder
                    Geschwisterkinder von Allergien betroffen sind, sprecht dies unbedingt nochmal genauer bei eurem Kinderarzt an."""

                )
            }
        },

        # STEP 2: Extraction
        {
            "type": "extraction",
            "settings": {
                "model": BASE_MODEL,
                "prompt_template": PROMPT_TMPL_S2,
                "temperature": BASE_TEMPERATURE,
                # "max_tokens": 512 # Currently hardcoded in individual step
            }
        },

        # STEP 3-5.1: The Research Module
        RESEARCH_MODULE,
        
        {
            "type": "retrieve_guideline_facts",
            "settings": {
                "db_path": "pipeline/RAG_vdb/guidelines_vdb.sqlite",
                "top_k": 5,
                "min_score": 0.25,
            },
        },
        
        # Step 5.99: Scores Module
        SCORES_MODULE,
        
        # STEP 6-8: The Verification Module
        VERIFICATION_MODULE
    ]
}
