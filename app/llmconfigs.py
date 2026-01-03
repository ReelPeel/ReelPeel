import openai

OPENAI_BASE_URL   = "http://localhost:11434/v1"
BASE_MODEL        = "gemma3:12b"     
BASE_API_KEY    = "ollama" 
BASE_TEMP        = 0.7
# ────────────────────────────────────────────────────────────────────
# Configuration STEP2
# ────────────────────────────────────────────────────────────────────
CLIENT_2 = openai.OpenAI(
    base_url=OPENAI_BASE_URL,  
    api_key=BASE_API_KEY
)

MODEL_2 = BASE_MODEL  
TEMP_2 = BASE_TEMP
MAX_TOKENS_2 = 128

# ────────────────────────────────────────────────────────────────────
# Configuration STEP3
# ────────────────────────────────────────────────────────────────────
CLIENT_3 = openai.OpenAI(
    base_url=OPENAI_BASE_URL,  
    api_key=BASE_API_KEY
)

MODEL_3 = BASE_MODEL  
TEMP_3 = BASE_TEMP
MAX_TOKENS_3 = 1024



# ────────────────────────────────────────────────────────────────────
# Configuration STEP5
# ────────────────────────────────────────────────────────────────────

CLIENT_5 = openai.OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=BASE_API_KEY
)

MODEL_5 = "gemma3:12b"
TEMP_5 = BASE_TEMP
MAX_TOKENS_5 = 512


# ────────────────────────────────────────────────────────────────────
# Configuration STEP6
# ────────────────────────────────────────────────────────────────────

CLIENT_6 = openai.OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=BASE_API_KEY
)

MODEL_6 = BASE_MODEL
TEMP_6 = BASE_TEMP
MAX_TOKENS_6 = 512

# ────────────────────────────────────────────────────────────────────
# Configuration STEP7
# ────────────────────────────────────────────────────────────────────

CLIENT_7 = openai.OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=BASE_API_KEY
)

MODEL_7 = BASE_MODEL
TEMP_7 = BASE_TEMP
MAX_TOKENS_7 = 512