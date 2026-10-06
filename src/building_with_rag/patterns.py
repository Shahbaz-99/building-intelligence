from .contracts import QueryRequest, QueryResult
from .registry import MODES


def run_pattern(request: QueryRequest) -> QueryResult:
    """Single path for /v1/query and /v1/chat/completions. Placeholder for every mode."""
    return QueryResult(
        pattern=request.pattern,
        status="not_implemented",
        message=f"Mode '{request.pattern}' is not implemented yet.",
        trace={"model_id": MODES[request.pattern]},
        results=[],
    )
