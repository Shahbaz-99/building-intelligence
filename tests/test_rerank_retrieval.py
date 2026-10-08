"""Offline Story 4.2 tests: selection, config validation, provider failures, routing."""

import httpx
import pytest
from fastapi import HTTPException

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation.context import build_context
from building_with_rag.settings import Settings
from building_with_rag import pipeline
from building_with_rag.retrieval import rerank


def _chunk(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{n}", section_id=f"s{n}", act="A", text=f"t{n}", heading="h",
        score=1.0 / n, fused_score=1.0 / n, fused_rank=n,
    )


def _settings(**kw) -> Settings:
    base = dict(
        rerank_api_key="k", rerank_candidate_limit=4, rerank_send_limit=3, rerank_return_limit=2
    )
    return Settings(_env_file=None, **(base | kw))


def _patch(monkeypatch, s, n=4, reply=None):
    calls = {"hybrid": 0, "provider": 0}

    def fake_hybrid(req):
        calls["hybrid"] += 1
        return QueryResult(
            pattern="hybrid", status="ok", message="", trace={"filters": {}, "caller_id": "d"},
            results=[_chunk(i) for i in range(1, n + 1)],
        )

    def fake_call(s_, q, docs):
        calls["provider"] += 1
        if isinstance(reply, Exception):
            raise reply
        return reply or ([{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9},
                          {"index": 2, "relevance_score": 0.5}], 7)

    monkeypatch.setattr(rerank, "get_settings", lambda: s)
    monkeypatch.setattr(rerank, "run_hybrid", fake_hybrid)
    monkeypatch.setattr(rerank, "call_reranker", fake_call)
    monkeypatch.setattr(rerank, "check_scope", lambda r: None)
    return calls


def _req(**kw):
    return QueryRequest(question="q", pattern="hybrid-reranked", limit=5, **kw)


def test_reorders_and_omits(monkeypatch):
    _patch(monkeypatch, _settings())
    r = rerank.run_hybrid_reranked(_req())
    assert [c.chunk_id for c in r.results] == ["c2", "c3"]
    assert [c.rerank_rank for c in r.results] == [1, 2]
    assert all(c.score == c.rerank_score and c.fused_rank for c in r.results)
    by = {c.chunk_id: c for c in r.omitted_candidates}
    assert by["c4"].omitted_reason == "not_sent_to_reranker" and by["c4"].rerank_score is None
    assert by["c1"].omitted_reason == "below_return_limit" and by["c1"].rerank_rank == 3
    assert [c.chunk_id for c in r.omitted_candidates] == ["c1", "c4"]
    t = r.trace["rerank"]
    assert (t["candidates"], t["sent"], t["returned"], t["omitted_before"], t["omitted_after"]) == (4, 3, 2, 1, 1)
    assert t["usage_tokens"] == 7
    entries, by_label = build_context(r.results)
    assert {c.chunk_id for c in by_label.values()} == {"c2", "c3"}


def test_empty_key_503_no_calls(monkeypatch):
    calls = _patch(monkeypatch, _settings(rerank_api_key=""))
    with pytest.raises(HTTPException) as e:
        rerank.run_hybrid_reranked(_req())
    assert e.value.status_code == 503 and e.value.detail["code"] == "retrieval_not_ready"
    assert "RERANK_API_KEY" in e.value.detail["message"]
    assert calls == {"hybrid": 0, "provider": 0}


def test_invalid_limits_503(monkeypatch):
    calls = _patch(monkeypatch, _settings(rerank_send_limit=5))
    with pytest.raises(HTTPException) as e:
        rerank.run_hybrid_reranked(_req())
    assert e.value.status_code == 503 and calls["hybrid"] == 0


@pytest.mark.parametrize(
    "status", ["timeout", "http", "dup", "badindex", "noscore", "empty"]
)
def test_provider_failures_502(monkeypatch, status):
    s = _settings()
    bodies = {
        "dup": {"data": [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 1}]},
        "badindex": {"data": [{"index": 9, "relevance_score": 1}]},
        "noscore": {"data": [{"index": 0}]},
        "empty": {"data": []},
    }

    def post(url, **kw):
        if status == "timeout":
            raise httpx.TimeoutException("x")
        if status == "http":
            return httpx.Response(500, request=httpx.Request("POST", url))
        return httpx.Response(200, json=bodies[status], request=httpx.Request("POST", url))

    monkeypatch.setattr(rerank.httpx, "post", post)
    with pytest.raises(HTTPException) as e:
        rerank.call_reranker(s, "q", ["a", "b", "c"])
    assert e.value.status_code == 502 and e.value.detail["code"] == "retrieval_upstream_error"


def test_provider_failure_returns_no_results(monkeypatch):
    _patch(monkeypatch, _settings(), reply=HTTPException(status_code=502, detail={"code": "x"}))
    with pytest.raises(HTTPException) as e:
        rerank.run_hybrid_reranked(_req())
    assert e.value.status_code == 502


def test_pipeline_routes(monkeypatch):
    sentinel = QueryResult(pattern="hybrid-reranked", status="ok", message="", trace={})
    monkeypatch.setattr(pipeline, "run_hybrid_reranked", lambda r: sentinel)
    assert pipeline.retrieve(_req()) is sentinel
    assert pipeline.generation_applies(sentinel)
