"""Query endpoint: shared retrieve + answer path; other modes stay placeholders."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.pipeline import generation_applies, retrieve, run_generation

router = APIRouter()

_OUTCOME_MESSAGES = {
    "answered": "Answer generated from the cited passages.",
    "insufficient_evidence": "Retrieved passages do not support an answer.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "Generation output was invalid and discarded; retrieved passages are returned.",
}
_LOW_CONFIDENCE_MESSAGE = (
    "Answer failed the evidence check and is returned only as a low-confidence draft; "
    "retrieved passages are returned."
)


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    result = retrieve(request)
    if request.generate_answer and generation_applies(result):
        result.generation = run_generation(request.question, result)
        note = (
            _LOW_CONFIDENCE_MESSAGE
            if result.generation.confidence == "low"
            else _OUTCOME_MESSAGES[result.generation.outcome]
        )
        result.message = f"{result.message} {note}"
    return result
