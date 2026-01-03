from typing import List, Optional, Union, Dict, Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    pubmed_id: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    pub_type: Optional[Union[str, List[str]]] = None
    weight: float = 0.5
    relevance: Optional[str] = None

    abstract: Optional[str] = None


class Statement(BaseModel):
    id: int
    text: str
    verdict: Optional[str] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    query: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)


class PipelineState(BaseModel):
    """The 'Source of Truth' passing between steps and modules."""
    transcript: Optional[str] = None
    statements: List[Statement] = Field(default_factory=list)
    overall_truthiness: Optional[float] = None
    generated_at: Optional[str] = None

    execution_log: List[Dict[str, Any]] = Field(default_factory=list)

    def to_json(self):
        return self.model_dump()