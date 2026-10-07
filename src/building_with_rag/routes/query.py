"""Query endpoint: semantic is real; other modes use the shared run_pattern placeholder."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import run_semantic

router = APIRouter()


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        return QueryResult(**run_semantic(request))
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
