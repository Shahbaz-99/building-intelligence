"""Query endpoint: semantic is real; other modes use the shared run_pattern placeholder."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.answer import generate_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import run_semantic

router = APIRouter()

_OUTCOME_MESSAGES = {
    "answered": "Answer generated from the cited passages.",
    "insufficient_evidence": "Retrieved passages do not support an answer.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "Generation output was invalid and discarded; retrieved passages are returned.",
}


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        result = QueryResult(**run_semantic(request))
        if request.generate_answer:
            result.generation = generate_answer(request.question, result.results)
            result.message = f"{result.message} {_OUTCOME_MESSAGES[result.generation.outcome]}"
        return result
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
