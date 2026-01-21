from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


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
    pages: List[int] = Field(default_factory=list)
    abstract: str
    weight: float = 1.0


Evidence = Annotated[
    Union[PubMedEvidence, RAGEvidence, EpistemonikosEvidence],
    Field(discriminator="source_type"),
]



class Statement(BaseModel):
    id: int
    text: str
    verdict: Optional[str] = None
    rationale: Optional[str] = None
    score: Optional[float] = None
    queries: List[str] = Field(default_factory=list)
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

    # ---  Track recursion depth for pretty printing ---
    depth: int = 0

    def to_json(self):
        return self.model_dump()
