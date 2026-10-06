"""Typed contracts (models only, no behavior). Later stories extend additively."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Pattern = Literal["semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde"]

# Fixed embedding choices (reused by every later story).
EMBEDDING_PROVIDER = "voyage"
EMBEDDING_MODEL = "voyage-3.5"
EMBEDDING_VERSION = "voyage-3.5"
EMBEDDING_DIMENSIONS = 1024


class SemanticFilters(BaseModel):
    act: list[str] | None = None
    status: list[str] | None = None
    access_level: list[str] | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Pattern = "semantic"
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] | None = None
    chapter: str | None = None


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    score: float | None = None
    source: str | None = None
    status: str | None = None
    access_level: str | None = None
    chapter: str | None = None
    section_number: str | None = None
    page: int | None = None


class OmittedCandidate(BaseModel):
    chunk_id: str
    omitted_reason: str


class SubquestionEvidence(BaseModel):
    subquestion: str
    status: Literal["evidenced", "no_evidence"]
    results: list[RetrievedChunk] = Field(default_factory=list)
    reason: str | None = None


class StructuredSignals(BaseModel):
    intent: Literal["exact_lookup", "filter", "aggregation"]
    act: str | None = None
    section_number: str | None = None
    chapter: str | None = None


class GenerationResult(BaseModel):
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"]
    answer: str | None = None
    claims: list[Any] = Field(default_factory=list)
    citations: list[Any] = Field(default_factory=list)
    supporting_passages: list[Any] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    trace: dict[str, Any] = Field(default_factory=dict)
    context_outcome: str | None = None
    confidence: float | str | None = None
    draft_answer: str | None = None
    issues: list[Any] = Field(default_factory=list)
    attempts: int | None = None
    low_confidence_reason: str | None = None


class QueryResult(BaseModel):
    pattern: str
    status: str
    message: str
    trace: dict[str, Any] = Field(default_factory=dict)
    results: list[RetrievedChunk] = Field(default_factory=list)
    generation: GenerationResult | None = None
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)
    subquestions: list[SubquestionEvidence] = Field(default_factory=list)
    hyde_direct_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_query_candidates: list[RetrievedChunk] = Field(default_factory=list)
    hyde_hypothetical_text_debug: str | None = None


# --- OpenAI-compatible shapes (text only) ---


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class RagOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Pattern | None = None
    filters: SemanticFilters | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    required_acts: list[str] | None = None
    chapter: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    rag_options: RagOptions | None = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ChatCompletionDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class OpenAIErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorEnvelope(BaseModel):
    error: OpenAIErrorBody


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "building-with-rag"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


# --- MongoDB schema contract models (no connection in the seed) ---


class SectionDocument(BaseModel):
    section_id: str  # act-qualified, e.g. BNS-103
    act: str
    section_number: str
    heading: str | None = None
    text: str
    chapter: str | None = None
    status: str | None = None
    access_level: str | None = None
    source: str | None = None


class ChunkDocument(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    status: str | None = None
    access_level: str | None = None
    source: str | None = None
    embedding: list[float] | None = None
    embedding_provider: str = EMBEDDING_PROVIDER
    embedding_model: str = EMBEDDING_MODEL
    embedding_version: str = EMBEDDING_VERSION
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
