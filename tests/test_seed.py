import json

from fastapi.testclient import TestClient

from building_with_rag.main import app
from building_with_rag.registry import MODES

client = TestClient(app)
MSG = [{"role": "user", "content": "hi"}]


def test_healthz_safe():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "app_env"}


def test_query_placeholder():
    r = client.post("/v1/query", json={"question": "What is murder?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_implemented"
    assert body["results"] == []


def test_models():
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == list(MODES.values()) and len(ids) == 6


def test_chat_json():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "rag-semantic",
            "stream": False,
            "messages": MSG,
            "rag_options": {"pattern": "semantic", "filters": {"act": ["BNS"]}, "limit": 5},
        },
    )
    assert r.status_code == 200
    assert "not implemented" in r.json()["choices"][0]["message"]["content"]


def test_chat_sse():
    r = client.post(
        "/v1/chat/completions", json={"model": "rag-hyde", "stream": True, "messages": MSG}
    )
    frames = [x for x in r.text.split("\n\n") if x]
    assert frames[-1] == "data: [DONE]"
    assert json.loads(frames[0][6:])["choices"][0]["delta"]["role"] == "assistant"
    assert json.loads(frames[-2][6:])["choices"][0]["finish_reason"] == "stop"


def test_chat_unknown_model_envelope():
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": MSG})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
