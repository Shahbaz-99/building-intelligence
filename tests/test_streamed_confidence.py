"""Story 3.2 offline tests: fake provider, no network."""

import json

import pytest
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer
from building_with_rag.routes import chat

CHUNK = RetrievedChunk(
    chunk_id="c1",
    section_id="bns:303",
    act="BNS_2023",
    text="Theft text.",
    heading="Theft",
    score=0.9,
    section_number=303,
    status="in_force",
    needs_review=False,
)
GOOD = "BNS punishes theft [E1]."
BAD = "BNS punishes theft [E9]."


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setattr(
        answer,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "generation_model_name": "m",
                "generation_api_base_url": "http://x",
                "generation_api_key": "k",
            },
        )(),
    )
    monkeypatch.setattr(
        chat,
        "retrieve",
        lambda req: QueryResult(
            pattern="semantic", status="ok", message="ok", trace={}, results=[CHUNK]
        ),
    )
    state = {"scripts": [], "calls": 0, "unsupported": []}

    def deltas(base, key, body):
        script = state["scripts"][state["calls"]]
        state["calls"] += 1
        for piece in script:
            if isinstance(piece, Exception):
                raise piece
            yield piece, "m"

    monkeypatch.setattr(answer, "_provider_deltas", deltas)
    monkeypatch.setattr(answer, "_check_support", lambda *a: state["unsupported"])
    return state


def _stream(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "rag-semantic",
            "stream": True,
            "messages": [{"role": "user", "content": "theft?"}],
        },
    ) as r:
        raw = b"".join(r.iter_bytes()).decode()
    frames = [f[6:] for f in raw.split("\n\n") if f.startswith("data: ")]
    text = "".join(
        json.loads(f)["choices"][0]["delta"].get("content", "") for f in frames if f != "[DONE]"
    )
    return text, frames


def test_pass_single_call_high(env):
    env["scripts"] = [["BNS punishes ", "theft [E1]."]]
    text, frames = _stream(TestClient(create_app()))
    assert env["calls"] == 1
    assert frames[-1] == "[DONE]" and '"finish_reason": "stop"' in frames[-2]
    assert text.startswith("DRAFT — checking evidence\n\n" + GOOD)
    assert "confidence: high" in text and "E1 · BNS_2023 §303 · Theft · bns:303" in text
    env["calls"] = 0
    events = list(answer.stream_answer("q", [CHUNK]))
    assert "".join(p for k, p in events if k == "text") == events[-1][1].text == GOOD
    assert events[-1][1].confidence == "high"


def test_retry_then_pass(env):
    env["scripts"] = [[BAD], [GOOD]]
    text, _ = _stream(TestClient(create_app()))
    assert text.count("DRAFT — checking evidence") == 2
    assert "Retrying (attempt 2 of 2)" in text and "confidence: high" in text
    env["calls"] = 0
    final = list(answer.stream_answer("q", [CHUNK]))[-1][1]
    assert [a.status for a in final.attempts] == ["failed", "passed"]
    assert [i.check for i in final.issues] == ["citation_labels"] and final.confidence == "high"


def test_both_fail_low_confidence(env):
    env["scripts"] = [[BAD], [BAD]]
    text, _ = _stream(TestClient(create_app()))
    assert text.rstrip().count("DRAFT — low confidence, not the final answer.") == 1
    env["calls"] = 0
    final = list(answer.stream_answer("q", [CHUNK]))[-1][1]
    assert final.text == "" and final.draft_answer == BAD and final.confidence == "low"
    assert len(final.issues) == 2 and final.low_confidence_reason
    assert final.claims[0].evidence_labels == ["E9"]


def test_provider_failure_mid_stream(env):
    env["scripts"] = [["BNS punishes", answer.ProviderError("timeout")]]
    text, frames = _stream(TestClient(create_app()))
    assert text.endswith("Answer generation unavailable — the text above is an unchecked draft.")
    assert frames[-1] == "[DONE]" and '"finish_reason": "stop"' in frames[-2]
    env["calls"] = 0
    final = list(answer.stream_answer("q", [CHUNK]))[-1][1]
    assert final.outcome == "unavailable" and final.confidence is None
