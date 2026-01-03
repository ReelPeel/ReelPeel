from app.preprompts import PROMPT_TMPL_S2

# This text simulates what Whisper would output
MOCK_TRANSCRIPT_TEXT = (
    "Did you know that intermittent fasting can regenerate your immune system? "
    "Studies show that fasting for 72 hours triggers autophagy which cleans out old cells. "
    "Also, drinking lemon water every morning cures depression instantly."
)

TEST_CONFIG_V1 = {
    "name": "Extraction_Test_Suite_01",
    "steps": [
        # STEP 1: The Mock Loader (Swapped out the real Whisper step)
        {
            "type": "mock_transcript",
            "settings": {
                "transcript_text": MOCK_TRANSCRIPT_TEXT
            }
        },

        # STEP 2: The Real Extraction Step (Running exactly as in production)
        {
            "type": "extraction",
            "settings": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "gemma3:27b",
                "temperature": 0.1,  # Low temp for factual extraction
                "max_tokens": 512,
                "prompt_template": PROMPT_TMPL_S2
            }
        }
    ]
}