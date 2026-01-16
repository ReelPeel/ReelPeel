from typing import Dict, Any

from ..steps.audio_to_transcript import AudioToTranscriptStep
from ..steps.reel_utils import DownloadReelStep
from ..steps.video_to_audio import VideoToAudioStep
from ..steps.extraction import TranscriptToStatementStep
# Import your concrete steps here (BUT NOT PipelineModule)
from ..steps.mocks import MockTranscriptLoader, MockStatementLoader
from ..steps.rerank import RerankEvidenceStep
from ..steps.research import (
    StatementToQueryStep,
    QueryToLinkStep,
    LinkToAbstractStep,
    PubTypeWeightStep
)
from ..steps.retrieve_guideline_facts_RAG import RetrieveGuidelineFactsStep
from ..steps.stance import StanceEvidenceStep
from ..steps.verification import (
    FilterEvidenceStep,
    TruthnessStep,
    ScoringStep
)


class StepFactory:
    # Do NOT put "module" in here to avoid circular imports
    _registry = {
        "download_reel": DownloadReelStep,
        "audio_to_transcript": AudioToTranscriptStep,
        "video_to_audio": VideoToAudioStep,
        "mock_transcript": MockTranscriptLoader,
        "mock_statements": MockStatementLoader,
        "extraction": TranscriptToStatementStep,
        "generate_query": StatementToQueryStep,
        "fetch_links": QueryToLinkStep,
        "summarize_evidence": LinkToAbstractStep,
        "weight_evidence": PubTypeWeightStep,
        "retrieve_guideline_facts": RetrieveGuidelineFactsStep,
        "rerank_evidence": RerankEvidenceStep,
        "stance_evidence": StanceEvidenceStep,
        "filter_evidence": FilterEvidenceStep,
        "truthness": TruthnessStep,
        "scoring": ScoringStep
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
