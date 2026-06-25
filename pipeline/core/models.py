from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated


class StanceLabel(str, Enum):
    """Represents how a piece of evidence relates to a given statement.

    Values:
        SUPPORTS: The evidence is consistent with the statement and tends to
            strengthen or support it.
        REFUTES: The evidence contradicts the statement or tends to weaken
            or undermine it.
        NEUTRAL: The evidence is relevant to the topic but does not clearly
            support or refute the statement (e.g., mixed, inconclusive, or
            purely background/contextual information).
    """

    SUPPORTS = "Supports"
    REFUTES = "Refutes"
    NEUTRAL = "Neutral"


class Stance(BaseModel):
    """Represents the stance of a piece of evidence towards a statement."""

    abstract_label: Optional[StanceLabel] = None
    abstract_p_supports: Optional[float] = None
    abstract_p_refutes: Optional[float] = None
    abstract_p_neutral: Optional[float] = None


class SourceType(str, Enum):
    PUBMED = "PubMed"
    RAG = "RAG"
    EPISTEMONIKOS = "Epistemonikos"


class EvidenceBase(BaseModel):
    source_type: SourceType
    weight: float = 0.15
    relevance: Optional[float] = None
    relevance_abstract: Optional[float] = None
    stance: Optional[Stance] = None


class PubMedEvidence(EvidenceBase):
    source_type: Literal[SourceType.PUBMED] = SourceType.PUBMED
    pubmed_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    pub_type: Optional[Union[str, List[str]]] = None


class EpistemonikosEvidence(EvidenceBase):
    source_type: Literal[SourceType.EPISTEMONIKOS] = SourceType.EPISTEMONIKOS
    epistemonikos_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    pub_type: Optional[Union[str, List[str]]] = None


class RAGEvidence(EvidenceBase):
    source_type: Literal[SourceType.RAG] = SourceType.RAG
    chunk_id: str
    score: float
    source_path: str
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    pages: List[int] = Field(default_factory=list)
    abstract: str
    weight: float = 1.0


Evidence = Annotated[
    Union[PubMedEvidence, RAGEvidence, EpistemonikosEvidence],
    Field(discriminator="source_type"),
]


class GuidelineDocumentResult(BaseModel):
    document_id: str
    source_path: str
    title: Optional[str] = None
    raw_retrieved_chunk_count: int = 0
    retrieved_chunk_count: int = 0
    evidence: List[RAGEvidence] = Field(default_factory=list)
    label: Optional[str] = None
    cited_chunk_ids: List[str] = Field(default_factory=list)
    classification_status: Optional[str] = None
    fallback_label_used: bool = False


class Statement(BaseModel):
    id: int
    text: str
    translated_text_de: Optional[str] = None
    translated_text_en: Optional[str] = None
    normalized_text: Optional[str] = None
    canonical_claim_de: Optional[str] = None
    canonical_claim_en: Optional[str] = None
    claim_type: Optional[str] = None
    routing_reason: Optional[str] = None
    retrieval_status: Optional[str] = None
    classification_status: Optional[str] = None
    failure_stage: Optional[str] = None
    fallback_label_used: Optional[bool] = None
    raw_retrieved_chunk_count: Optional[int] = None
    usable_retrieved_chunk_count: Optional[int] = None
    verdict: Optional[str] = None
    rationale: Optional[str] = None
    guideline_label: Optional[str] = None
    guideline_documents: List[GuidelineDocumentResult] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    queries: List[str] = Field(default_factory=list)
    queries_fetched: List[str] = Field(default_factory=list)
    retrieval_queries: List[str] = Field(default_factory=list)
    topic_flags: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)


class PipelineState(BaseModel):
    """The 'Source of Truth' passing between steps and modules."""

    transcript: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    statements: List[Statement] = Field(default_factory=list)
    overall_truthiness: Optional[float] = None
    generated_at: Optional[str] = None

    execution_log: List[Dict[str, Any]] = Field(default_factory=list)

    depth: int = 0

    def to_json(self):
        return self.model_dump()
