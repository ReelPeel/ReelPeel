from pipeline.test_configs.preprompts import PROMPT_TMPL_S2
from pipeline.test_configs.test_extraction import RESEARCH_MODULE, VERIFICATION_MODULE


AUDIO_PIPELINE_CONFIG = {
    "name": "Audio_To_Transcript_Run",
    "debug": True,
    "steps": [
        {
            "type": "audio_to_transcript",
            "settings": {
                "audio_path": "audios/audio.wav",
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
        VERIFICATION_MODULE,
    ],
}
