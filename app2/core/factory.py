from typing import Dict, Any

# Import your steps
from ..steps.extraction import TranscriptToStatementStep
from ..steps.mocks import MockTranscriptLoader


class StepFactory:
    _registry = {
        "mock_transcript": MockTranscriptLoader,
        "extraction": TranscriptToStatementStep
    }

    @classmethod
    def register(cls, name: str, step_class):
        cls._registry[name] = step_class

    @classmethod
    def create(cls, step_def: Dict[str, Any]):
        step_type = step_def["type"]
        step_config = step_def.get("settings", {})

        step_class = cls._registry.get(step_type)
        if not step_class:
            raise ValueError(f"Step type '{step_type}' not registered.")

        return step_class(step_config)