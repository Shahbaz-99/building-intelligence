"""Shared retrieve -> answer path used by /v1/query and /v1/chat/completions."""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.rerank import run_hybrid_reranked
from building_with_rag.retrieval.semantic import run_semantic
from building_with_rag.retrieval.structured import run_structured

REAL_PATTERNS = frozenset(
    {
        Pattern.SEMANTIC.value,
        Pattern.HYBRID.value,
        Pattern.HYBRID_RERANKED.value,
        Pattern.STRUCTURED.value,
    }
)
UNAVAILABLE_AFTER_TEXT = "\n\nAnswer generation unavailable — the text above is an unchecked draft."
UNAVAILABLE_NO_TEXT = "Answer generation unavailable."


def retrieve(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        return QueryResult(**run_semantic(request))
    if request.pattern is Pattern.HYBRID:
        return run_hybrid(request)
    if request.pattern is Pattern.HYBRID_RERANKED:
        return run_hybrid_reranked(request)
    if request.pattern is Pattern.STRUCTURED:
        return run_structured(request)
    return QueryResult(**run_pattern(request.pattern, request.question, request.caller_id))


def generation_applies(retrieval: QueryResult) -> bool:
    if retrieval.pattern == Pattern.STRUCTURED.value:
        return retrieval.status == "ok"
    return retrieval.pattern in REAL_PATTERNS


def answer_events(
    question: str, retrieval: QueryResult
) -> Iterator[tuple[str, str | GenerationResult | None]]:
    """Yield ("text"|"notice", str) pieces, then ("final", GenerationResult | None)."""
    if not generation_applies(retrieval):
        yield "text", retrieval.message
        yield "final", None
        return
    yield from stream_answer(question, retrieval.results)


def run_generation(question: str, retrieval: QueryResult) -> GenerationResult | None:
    final = None
    for kind, payload in answer_events(question, retrieval):
        if kind == "final":
            final = payload
    return final


def closing_text(generation: GenerationResult | None, streamed_text: bool) -> str:
    """Text appended after the stream: footer, low-confidence block, or failure line."""
    if generation is None:
        return ""
    if generation.outcome == "answered":
        lines = ["\n\n---\nEvidence check passed — confidence: high", "Sources:"]
        by_label = {c.label: c for c in generation.citations}
        for label, c in by_label.items():
            lines.append(f"{label} · {c.act} §{c.section_number} · {c.heading} · {c.section_id}")
        return "\n".join(lines)
    if generation.outcome == "insufficient_evidence":
        reason = str(generation.trace.get("reason") or "").strip()[:200]
        sentence = "The retrieved passages do not support an answer"
        return ("\n\n" if streamed_text else "") + (
            f"{sentence}: {reason}" if reason else f"{sentence}."
        )
    if generation.confidence == "low":
        details = "\n".join(f"- {i.detail}" for i in generation.issues)
        return (
            "\n\nDRAFT — low confidence, not the final answer.\n"
            f"{generation.low_confidence_reason}\n{details}"
        )
    if generation.outcome == "unavailable":
        return UNAVAILABLE_AFTER_TEXT if streamed_text else UNAVAILABLE_NO_TEXT
    # malformed without confidence: nothing judged
    if streamed_text:
        return "\n\nDRAFT — the answer could not be checked and is not final."
    return "The answer could not be produced."
