"""Offline Story 5.2 tests: validation, union, statuses, hypothetical-text boundary, routing."""

import json

import pytest
from fastapi import HTTPException

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, RetrievedChunk
from building_with_rag.generation import complete
from building_with_rag.generation.context import build_context
from building_with_rag.retrieval import decomposition, hyde, semantic
from building_with_rag.routes import chat as chat_route


def chunk(cid, act="BNS_2023"):
    return RetrievedChunk(
        chunk_id=cid, section_id=cid, act=act, text=f"text {cid}", heading="h", score=0.5
    )


def subs_json(*questions, in_scope=True):
    return json.dumps(
        {"in_scope": in_scope, "subquestions": [{"question": q, "act": None} for q in questions]}
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        subs_json(),
        subs_json("a", "A "),
        subs_json("a", "b", "c", "d"),
        subs_json("a", in_scope=False),
        subs_json(""),
        json.dumps({"in_scope": True, "subquestions": [{"question": "a", "act": "XYZ"}]}),
    ],
)
def test_validate_subquestions_rejects(raw):
    items, reason = decomposition.validate_subquestions(raw)
    assert items is None and reason


def test_validate_subquestions_accepts():
    items, reason = decomposition.validate_subquestions(subs_json("a?", "b?", "c?"))
    assert reason is None and [i["question"] for i in items] == ["a?", "b?", "c?"]


def test_union_round_robin_keeps_best_first():
    a, b, c = [chunk("a1"), chunk("a2")], [chunk("b1"), chunk("a1")], [chunk("c1"), chunk("c2")]
    ids = [x.chunk_id for x in decomposition.union_round_robin([a, b, c])]
    assert ids == ["a1", "b1", "c1", "a2", "c2"]


@pytest.fixture()
def wired(monkeypatch):
    calls = {"model": 0, "search": [], "embed": []}
    monkeypatch.setattr(decomposition, "require_settings", lambda: None)
    monkeypatch.setattr(hyde, "require_settings", lambda: None)
    monkeypatch.setattr(semantic, "open_db", lambda: (object(), object()))
    monkeypatch.setattr(
        semantic, "embed_query", lambda v, text: calls["embed"].append(text) or [0.0]
    )
    return calls


def _run_decomp(monkeypatch, wired, reply, by_search, **kw):
    def fake_model(system, user):
        wired["model"] += 1
        return reply

    monkeypatch.setattr(decomposition, "complete_text", fake_model)
    state = {"i": 0}

    def fake_search(db, vector, filters, limit):
        wired["search"].append((filters, limit))
        out = by_search[state["i"]]
        state["i"] += 1
        return out

    monkeypatch.setattr(semantic, "search_chunks", fake_search)
    return decomposition.run_decomposition(
        QueryRequest(question="BNS vs IPC murder?", pattern="decomposition", **kw)
    )


def test_decomposition_ok(monkeypatch, wired):
    r = _run_decomp(monkeypatch, wired, subs_json("a?", "b?"), [[chunk("a1")], [chunk("b1")]])
    assert r.status == "ok" and [c.chunk_id for c in r.results] == ["a1", "b1"]
    assert [s.status for s in r.subquestions] == ["supported", "supported"]
    assert pipeline.generation_applies(r)


def test_decomposition_partial_and_clarify_no_generation(monkeypatch, wired):
    r = _run_decomp(monkeypatch, wired, subs_json("a?", "b?"), [[chunk("a1")], []])
    assert r.status == "partial_answer" and "b?" in r.message
    assert not pipeline.generation_applies(r)
    assert list(pipeline.answer_events("q", r))[0] == ("text", r.message)
    r = _run_decomp(monkeypatch, wired, subs_json("a?"), [[]])
    assert r.status == "clarify" and not pipeline.generation_applies(r)


def test_decomposition_bad_output_no_retrieval(monkeypatch, wired):
    r = _run_decomp(monkeypatch, wired, "nope", [])
    assert r.status == "clarify" and wired["search"] == [] and wired["embed"] == []


def test_act_hint_only_narrows_filter(monkeypatch, wired):
    reply = json.dumps(
        {"in_scope": True, "subquestions": [{"question": '{"$ne": null}', "act": "IPC_1860"}]}
    )
    r = _run_decomp(monkeypatch, wired, reply, [[chunk("i1", "IPC_1860")]])
    filters, _ = wired["search"][0]
    assert filters == {"access_level": ["public"], "act": ["IPC_1860"]}
    assert wired["embed"] == ['{"$ne": null}'] and r.status == "ok"


def test_act_hint_excluded_by_caller_filter(monkeypatch, wired):
    reply = json.dumps({"in_scope": True, "subquestions": [{"question": "a?", "act": "IPC_1860"}]})
    r = _run_decomp(monkeypatch, wired, reply, [], filters={"act": ["BNS_2023"]})
    assert r.status == "clarify" and wired["search"] == []


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "  ",
        "text",
        json.dumps({"x": 1}),
        json.dumps({"hypothetical_passage": " "}),
        json.dumps({"hypothetical_passage": "x" * 1501}),
    ],
)
def test_validate_hypothesis_rejects(raw):
    assert hyde.validate_hypothesis(raw) is None


def test_hyde_unavailable_makes_no_search(monkeypatch, wired):
    monkeypatch.setattr(hyde, "complete_text", lambda s, u: "garbage")
    monkeypatch.setattr(semantic, "search_chunks", lambda *a: pytest.fail("no search"))
    r = hyde.run_hyde(QueryRequest(question="phone stolen?", pattern="hyde"))
    assert r.status == "hyde_unavailable" and r.results == []
    assert r.hyde_direct_candidates == [] and wired["embed"] == []


def test_hyde_hypothetical_text_never_evidence(monkeypatch, wired):
    secret = "Whoever dishonestly takes a movable thing HYPOTHETICALMARKER"
    monkeypatch.setattr(
        hyde, "complete_text", lambda s, u: json.dumps({"hypothetical_passage": secret})
    )
    lists = iter([[chunk("d1"), chunk("x")], [chunk("h1"), chunk("x")]])
    monkeypatch.setattr(semantic, "search_chunks", lambda *a: next(lists))
    r = hyde.run_hyde(QueryRequest(question="phone stolen?", pattern="hyde", limit=5))
    assert r.status == "ok" and [c.chunk_id for c in r.results] == ["d1", "h1", "x"]
    assert r.trace["contribution"] == {"d1": "direct_only", "h1": "hyde_only", "x": "both"}
    assert r.hyde_hypothetical_text_debug == secret and wired["embed"][1] == secret
    assert "HYPOTHETICALMARKER" not in r.model_dump_json(exclude={"hyde_hypothetical_text_debug"})
    entries, _ = build_context(r.results)
    assert "HYPOTHETICALMARKER" not in json.dumps(entries)


def test_missing_setting_503_before_any_call(monkeypatch):
    class S:
        generation_api_base_url = "http://x"
        generation_api_key = ""
        generation_model_name = "m"
        mongodb_uri = "u"
        voyage_api_key = "v"

    monkeypatch.setattr(complete, "get_settings", lambda: S())
    monkeypatch.setattr(decomposition, "complete_text", lambda *a: pytest.fail("no call"))
    with pytest.raises(HTTPException) as exc:
        decomposition.run_decomposition(QueryRequest(question="q", pattern="decomposition"))
    assert exc.value.status_code == 503 and "GENERATION_API_KEY" in exc.value.detail["message"]


def test_pipeline_routes(monkeypatch):
    monkeypatch.setattr(pipeline, "run_decomposition", lambda r: "D")
    monkeypatch.setattr(pipeline, "run_hyde", lambda r: "H")
    assert pipeline.retrieve(QueryRequest(question="q", pattern="decomposition")) == "D"
    assert pipeline.retrieve(QueryRequest(question="q", pattern="hyde")) == "H"
    assert chat_route.retrieve is pipeline.retrieve
