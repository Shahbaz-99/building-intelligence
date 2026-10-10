"""Offline Story 5.1 tests: classifier, predicate, outcomes, routing."""

import pytest

from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest
from building_with_rag.generation.context import build_context
from building_with_rag.retrieval import structured
from building_with_rag.retrieval.structured import classify


@pytest.mark.parametrize(
    "q,act,num",
    [
        ("What does BNS section 103 say?", "BNS_2023", 103),
        ("IPC sec. 302", "IPC_1860", 302),
        ("Indian Penal Code section 420", "IPC_1860", 420),
    ],
)
def test_classify_ok(q, act, num):
    s = classify(q)
    assert (s.status, s.act, s.section_number, s.intent) == ("ok", act, num, "exact_lookup")


@pytest.mark.parametrize(
    "q",
    ["What does section 103 say?", "BNS and IPC section 103", "BNS section 103 and section 104",
     "BNS section 103A", "BNS section 0", "BNS section 1000"],
)
def test_classify_clarification(q):
    s = classify(q)
    assert s.status == "clarification_needed" and s.act is None


def test_classify_other_intents():
    assert classify("How many sections are in BNS?").intent == "aggregation"
    assert classify("List all sections in chapter V").intent == "filter"
    s = classify("What is the punishment for theft?")
    assert (s.status, s.intent) == ("recommendation", None)


class FakeColl:
    def __init__(self, doc=None):
        self.doc, self.calls = doc, []

    def find_one(self, predicate, projection=None):
        self.calls.append(predicate)
        return self.doc


SECTION = {
    "section_id": "bns:103", "act": "BNS_2023", "section_number": 103, "status": "in_force",
    "heading": "Murder", "text": "Whoever commits murder...", "source_status_version": "v1",
}


def _run(monkeypatch, coll, question, **kw):
    monkeypatch.setattr(structured, "_sections", lambda: coll)
    return structured.run_structured(QueryRequest(question=question, pattern="structured", **kw))


def test_ok_and_predicate_only_validated(monkeypatch):
    coll = FakeColl(dict(SECTION))
    r = _run(monkeypatch, coll, 'BNS section 103 {"$ne": null}')
    assert r.status == "ok" and r.results[0].chunk_id == "bns:103" and r.results[0].score == 1.0
    assert r.trace["mongodb_called"] and r.trace["record"]["source_status_version"] == "v1"
    assert coll.calls == [
        {"act": "BNS_2023", "section_number": 103, "access_level": {"$in": ["public"]}}
    ]


def test_optional_predicate_fields(monkeypatch):
    coll = FakeColl(dict(SECTION))
    _run(monkeypatch, coll, "BNS section 103", chapter="VI",
         filters={"status": ["in_force"]})
    assert set(coll.calls[0]) == {"act", "section_number", "access_level", "status", "chapter"}


def test_non_ok_makes_no_call(monkeypatch):
    coll = FakeColl(dict(SECTION))
    r = _run(monkeypatch, coll, "What does section 103 say?")
    assert r.status == "clarification_needed" and r.results == [] and not coll.calls
    assert r.trace["mongodb_called"] is False
    assert not pipeline.generation_applies(r)
    assert list(pipeline.answer_events("q", r))[0] == ("text", r.message)


def test_not_found(monkeypatch):
    r = _run(monkeypatch, FakeColl(None), "BNS section 999")
    assert r.status == "not_found" and r.results == [] and r.trace["mongodb_called"]


def test_routing_and_context(monkeypatch):
    monkeypatch.setattr(structured, "_sections", lambda: FakeColl(dict(SECTION)))
    r = pipeline.retrieve(QueryRequest(question="BNS section 103", pattern="structured"))
    assert r.pattern == "structured" and pipeline.generation_applies(r)
    _, by_label = build_context(r.results)
    assert [c.chunk_id for c in by_label.values()] == ["bns:103"]
