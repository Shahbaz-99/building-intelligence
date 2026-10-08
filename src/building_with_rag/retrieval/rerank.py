"""Hybrid re-ranking: hybrid candidates -> one provider rerank call -> final evidence."""

import logging
import math
import time

import httpx

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.registry import Pattern
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.semantic import _not_ready, _upstream, check_scope
from building_with_rag.settings import Settings, get_settings

log = logging.getLogger(__name__)

MAX_CANDIDATES = 20
_FAILED = "Re-ranking request failed; no re-ranked result returned."


def validate_settings(s: Settings) -> None:
    """Raise 503 retrieval_not_ready naming the setting (never its value)."""
    if not s.rerank_api_key:
        raise _not_ready("RERANK_API_KEY is not set; hybrid-reranked requires it.")
    if not s.rerank_api_base_url:
        raise _not_ready("RERANK_API_BASE_URL is not set; hybrid-reranked requires it.")
    if not s.rerank_model_name:
        raise _not_ready("RERANK_MODEL_NAME is not set; hybrid-reranked requires it.")
    if s.rerank_request_timeout_seconds < 1:
        raise _not_ready("RERANK_REQUEST_TIMEOUT_SECONDS must be at least 1.")
    if not 1 <= s.rerank_return_limit <= s.rerank_send_limit <= s.rerank_candidate_limit <= MAX_CANDIDATES:
        raise _not_ready(
            "RERANK_RETURN_LIMIT, RERANK_SEND_LIMIT and RERANK_CANDIDATE_LIMIT must satisfy "
            f"1 <= RETURN <= SEND <= CANDIDATE <= {MAX_CANDIDATES}."
        )


def call_reranker(s: Settings, query: str, documents: list[str]) -> tuple[list[dict], int | None]:
    """One provider call; returns validated [{index, relevance_score}] and usage tokens."""
    try:
        response = httpx.post(
            f"{s.rerank_api_base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {s.rerank_api_key}"},
            json={"model": s.rerank_model_name, "query": query, "documents": documents},
            timeout=s.rerank_request_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("rerank call failed: %s", type(exc).__name__)
        raise _upstream(_FAILED) from None
    return _validate_reply(body, len(documents))


def _validate_reply(body, sent: int) -> tuple[list[dict], int | None]:
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        log.warning("rerank reply invalid: data")
        raise _upstream(_FAILED)
    seen: set[int] = set()
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        score = item.get("relevance_score") if isinstance(item, dict) else None
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < sent
            or index in seen
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            log.warning("rerank reply invalid: item")
            raise _upstream(_FAILED)
        seen.add(index)
    usage = body.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return data, tokens if isinstance(tokens, int) else None


def select(
    candidates: list[RetrievedChunk],
    scores: dict[int, float],
    send_limit: int,
    final_limit: int,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Pure selection. `scores` maps position in the sent list (fused order) -> relevance."""
    sent = candidates[:send_limit]
    omitted = [
        c.model_copy(update={"omitted_reason": "not_sent_to_reranker"})
        for c in candidates[send_limit:]
    ]
    order = sorted(range(len(sent)), key=lambda i: (-scores[i], sent[i].fused_rank or 0))
    scored = [
        sent[i].model_copy(update={"score": scores[i], "rerank_score": scores[i], "rerank_rank": r})
        for r, i in enumerate(order, start=1)
    ]
    final = scored[:final_limit]
    omitted += [c.model_copy(update={"omitted_reason": "below_return_limit"}) for c in scored[final_limit:]]
    omitted.sort(key=lambda c: c.fused_rank or 0)
    return final, omitted


def run_hybrid_reranked(request: QueryRequest) -> QueryResult:
    check_scope(request)
    s = get_settings()
    validate_settings(s)
    hybrid = run_hybrid(
        request.model_copy(update={"pattern": Pattern.HYBRID, "limit": s.rerank_candidate_limit})
    )
    htrace = hybrid.trace
    candidates = hybrid.results
    rerank = {
        "model": s.rerank_model_name,
        "candidate_limit": s.rerank_candidate_limit,
        "send_limit": s.rerank_send_limit,
        "return_limit": s.rerank_return_limit,
        "candidates": len(candidates),
    }
    trace = {
        "mode": "hybrid-reranked",
        "query": request.question,
        "filters": htrace.get("filters"),
        "caller_id": htrace.get("caller_id"),
        "hybrid": {
            k: htrace.get(k)
            for k in ("embedding", "semantic", "keyword", "fusion", "contribution", "unresolved_hits")
        },
        "rerank": rerank,
    }
    if not candidates:
        rerank.update(sent=0, returned=0, omitted_before=0, omitted_after=0)
        trace["result_count"] = 0
        return QueryResult(
            pattern="hybrid-reranked",
            status="no_results",
            message=(
                "No passages matched the filters in either route, so nothing was re-ranked. "
                "Re-ranking scores rank only; they do not prove correctness."
            ),
            trace=trace,
        )

    sent = candidates[: s.rerank_send_limit]
    documents = [f"{c.heading}\n{c.text}" for c in sent]
    started = time.monotonic()
    data, tokens = call_reranker(s, request.question, documents)
    latency_ms = int((time.monotonic() - started) * 1000)
    if len(data) != len(sent):
        log.warning("rerank reply invalid: count")
        raise _upstream(_FAILED)
    scores = {item["index"]: float(item["relevance_score"]) for item in data}
    final, omitted = select(
        candidates, scores, s.rerank_send_limit, min(request.limit, s.rerank_return_limit)
    )
    rerank.update(
        sent=len(sent),
        returned=len(final),
        omitted_before=len(candidates) - len(sent),
        omitted_after=len(sent) - len(final),
        latency_ms=latency_ms,
    )
    if tokens is not None:
        rerank["usage_tokens"] = tokens
    trace["result_count"] = len(final)
    return QueryResult(
        pattern="hybrid-reranked",
        status="ok",
        message=(
            f"Returned {len(final)} passage(s) re-ranked from {len(candidates)} hybrid candidates. "
            "Re-ranking scores rank only; they do not prove a passage is correct "
            "or answers the question."
        ),
        trace=trace,
        results=final,
        omitted_candidates=omitted,
    )
