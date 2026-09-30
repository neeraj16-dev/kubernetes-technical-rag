from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Technical Kubernetes query", example="How does Kubernetes handle container crashes?")
    top_n: int = Field(default=5, ge=1, le=10, description="Number of reranked chunks to feed the generator")
    verify_grounding: bool = Field(default=True, description="Run online grounding verification guardrail")


class CitationModel(BaseModel):
    chunk_id: str
    source: str
    page: int


class VerificationModel(BaseModel):
    is_grounded: bool
    unsupported_claims: List[str]
    hallucination_score: float
    explanation: str
    phantom_citations: List[str]


class ChunkMetadata(BaseModel):
    chunk_id: Optional[str]
    source: Optional[str]
    page: Optional[int]
    section: Optional[str]
    content_preview: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    verification: Optional[VerificationModel] = None
    retrieved_chunks: List[ChunkMetadata]
    metrics: Dict[str, float]