"""Offline Story 4.1 tests: RRF fusion and run_hybrid with faked searches."""

import pytest
from fastapi import HTTPException

from building_with_rag.contracts import QueryRequest
from building_with_rag.retrieval import hybrid


def _hits(*ids):
    return [{"chunk_id": i, "score": 1.0 / n} for n, i in enumerate(ids, start=1)]


def test_both_routes_rank_first_with_rrf_score():
    out = hybrid.fuse(_hits("a", "b"), _hits("c", "b"), 5)
    top = out[0]
    assert top["chunk_id"] == "b"
    assert (top["semantic_rank"], top["keyword_rank"]) == (2, 2)
    assert top["fused_score"] == pytest.approx(2 / (hybrid.RRF_K + 2))
    assert [r["fused_rank"] for r in out] == [1, 2, 3]


def test_single_route_leaves_other_none():
    out = {r["chunk_id"]: r for r in hybrid.fuse(_hits("a"), _hits("c"), 5)}
    assert out["a"]["keyword_score"] is None and out["a"]["keyword_rank"] is None
    assert out["c"]["semantic_score"] is None and out["c"]["semantic_rank"] is None


def test_tie_order_semantic_rank_then_chunk_id():
    out = hybrid.fuse(_hits("z"), _hits("a"), 5)  # equal fused scores
    assert [r["chunk_id"] for r in out] == ["z", "a"]  # semantic first
    out = hybrid.fuse([], _hits("b") + _hits("a")[:0], 5)
    assert [r["chunk_id"] for r in out] == ["b"]


class _Settings:
    mongodb_db_name = "db"
    webui_demo_caller_id = "demo"


def _patch(monkeypatch, sem, kw):
    monkeypatch.setattr(hybrid, "get_settings", lambda: _Settings())
    monkeypatch.setattr(hybrid, "_clients", lambda: ({"db": {}}, object()))
    monkeypatch.setattr(hybrid, "_check_ready", lambda db: None)
    monkeypatch.setattr(hybrid, "embed_query", lambda v, q: [0.0])
    monkeypatch.setattr(hybrid, "vector_search", lambda *a, **k: sem)
    monkeypatch.setattr(hybrid, "keyword_search", lambda *a, **k: kw)
    chunks = {i: {"chunk_id": i, "section_id": "s" + i, "text": "t"} for i in "abc"}
    sections = {"s" + i: {"section_id": "s" + i, "act": "A", "heading": "h"} for i in "abc"}
    monkeypatch.setattr(hybrid, "resolve_sections", lambda db, hits: (chunks, sections))


def test_run_hybrid_fields(monkeypatch):
    _patch(monkeypatch, _hits("a", "b"), _hits("b", "c"))
    result = hybrid.run_hybrid(QueryRequest(question="q", pattern="hybrid", limit=3))
    assert result.status == "ok" and result.pattern == "hybrid"
    for pos, r in enumerate(result.results, start=1):
        assert r.score == r.fused_score and r.fused_rank == pos
    assert result.results[0].chunk_id == "b"
    assert result.trace["contribution"] == {"both": 1, "semantic_only": 1, "keyword_only": 1}


def test_run_hybrid_no_results(monkeypatch):
    _patch(monkeypatch, [], [])
    assert hybrid.run_hybrid(QueryRequest(question="q", pattern="hybrid")).status == "no_results"


def test_missing_keyword_index_not_ready(monkeypatch):
    class Coll:
        def __init__(self, idx):
            self.idx = idx

        def list_search_indexes(self, name=None):
            return self.idx

        def find_one(self, *a, **k):
            return {"_id": 1}

    db = {
        "embeddings": Coll([{"name": "vector_index", "queryable": True}]),
        "chunks": Coll([]),
    }
    monkeypatch.setattr(hybrid, "_ready", False)
    with pytest.raises(HTTPException) as exc:
        hybrid._check_ready(db)
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "retrieval_not_ready"
    assert "keyword_index" in exc.value.detail["message"]
