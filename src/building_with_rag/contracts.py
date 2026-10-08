"""Shared API contracts. Later stories extend additively; never rename or add provider variants."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from building_with_rag.ingestion.mongodb_schema import (
    validate_access_level,
    validate_act,
    validate_status,
)
from building_with_rag.registry import Pattern


class SemanticFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)

    @field_validator("act")
    @classmethod
    def _check_act(cls, values: list[str]) -> list[str]:
        return [validate_act(v) for v in values]

    @field_validator("status")
    @classmethod
    def _check_status(cls, values: list[str]) -> list[str]:
        return [validate_status(v) for v in values]

    @field_validator("access_level")
    @classmethod
    def _check_access_level(cls, values: list[str]) -> list[str]:
        return [validate_access_level(v) for v in values]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Pattern
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] | None = None
    chapter: str | None = None

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty or whitespace-only")
        return value


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str
    score: float
    # Optional source details; missing values stay None, never guessed.
    chunk_index: int | None = None
    act_label: str | None = None
    status: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    section_number: int | str | None = None
    source_pdf: str | None = None
    source_sha256: str | None = None
    needs_review: bool | None = None
    # Story 4.1 hybrid-only route evidence; None when the route did not return the chunk.
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    # Story 4.2 re-ranking evidence; omitted_reason is set only on omitted candidates.
    rerank_score: float | None = None
    rerank_rank: int | None = None
    omitted_reason: str | None = None


class GenerationClaim(BaseModel):
    text: str
    evidence_labels: list[str] = Field(default_factory=list)


class GenerationCitation(BaseModel):
    label: str
    chunk_id: str
    section_id: str
    act: str
    heading: str = ""
    chapter: str | None = None
    section_number: int | str | None = None
    source_pdf: str | None = None


class GenerationIssue(BaseModel):
    attempt: int
    check: str
    detail: str


class GenerationAttempt(BaseModel):
    attempt: int
    status: Literal["passed", "failed", "unjudged"]
    chars: int = 0
    latency_ms: int = 0


class GenerationResult(BaseModel):
    text: str = ""
    model: str | None = None
    # Story 3.1 additive fields.
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"] | None = None
    claims: list[GenerationClaim] = Field(default_factory=list)
    citations: list[GenerationCitation] = Field(default_factory=list)
    supporting_passages: list[RetrievedChunk] = Field(default_factory=list)
    provider: str | None = None
    trace: dict = Field(default_factory=dict)
    context_outcome: Literal["assembled", "empty"] | None = None
    # Story 3.2 additive fields.
    confidence: Literal["high", "low"] | None = None
    issues: list[GenerationIssue] = Field(default_factory=list)
    attempts: list[GenerationAttempt] = Field(default_factory=list)
    draft_answer: str = ""
    low_confidence_reason: str = ""


class QueryResult(BaseModel):
    pattern: str
    status: str
    message: str
    trace: dict
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    # Additive, empty-by-default fields later modes use:
    omitted_candidates: list[RetrievedChunk] = Field(default_factory=list)
    subquestions: list[str] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


class ChatRagFilters(BaseModel):
    """Nested filters as sent by the Open WebUI Pipe."""

    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)


class ChatRagOptions(BaseModel):
    pattern: Pattern = Pattern.SEMANTIC
    act: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    access_level: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None
    filters: ChatRagFilters | None = None

    @model_validator(mode="after")
    def _merge_nested_filters(self) -> "ChatRagOptions":
        if self.filters:
            self.act = self.act or self.filters.act
            self.status = self.status or self.filters.status
        return self


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|developer|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    n: int = 1
    rag_options: ChatRagOptions | None = None
