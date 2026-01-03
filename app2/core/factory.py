from typing import Dict, Any

# Import your concrete steps here (BUT NOT PipelineModule)
from ..steps.mocks import MockTranscriptLoader
from ..steps.extraction import TranscriptToStatementStep
from ..steps.research import (
    StatementToQueryStep,
    QueryToLinkStep,
    LinkToSummaryStep,
    PubTypeWeightStep
)


class StepFactory:
    # Do NOT put "module" in here to avoid circular imports
    _registry = {
        "mock_transcript": MockTranscriptLoader,
        "extraction": TranscriptToStatementStep,
        "generate_query": StatementToQueryStep,
        "fetch_links": QueryToLinkStep,
        "summarize_evidence": LinkToSummaryStep,
        "weight_evidence": PubTypeWeightStep,
    }

    @classmethod
    def register(cls, name: str, step_class):
        cls._registry[name] = step_class

    @classmethod
    def create(cls, step_def: Dict[str, Any]):
        step_type = step_def["type"]
        step_config = step_def.get("settings", {})

        if step_type == "module":
            # 2. Import locally to prevent circular dependency
            from .base import PipelineModule
            return PipelineModule(step_config)

        step_class = cls._registry.get(step_type)
        if not step_class:
            raise ValueError(f"Step type '{step_type}' not registered.")

        return step_class(step_config)