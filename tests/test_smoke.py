"""Minimal smoke tests: health, models, chat framing (no mode is a placeholder)."""

import pytest
from fastapi.testclient import TestClient

from building_with_rag.app import create_app
from building_with_rag.contracts import QueryResult
from building_with_rag.registry import PATTERN_MODEL_IDS
from building_with_rag.routes import chat


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def non_ok_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        chat,
        "retrieve",
        lambda r: QueryResult(
            pattern=r.pattern.value, status="clarify", message="Need more.", trace={}
        ),
    )


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_lists_six(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert ids == list(PATTERN_MODEL_IDS.values())


@pytest.mark.parametrize("model", ["rag-decomposition", "rag-hyde"])
def test_chat_json_non_ok_message(client: TestClient, non_ok_retrieval, model: str) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Need more."


def test_chat_stream_non_ok_message(client: TestClient, non_ok_retrieval) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "rag-decomposition",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    assert "chat.completion.chunk" in raw
    assert '"finish_reason": "stop"' in raw
    done = "[" + "DONE" + "]"
    assert raw.strip().endswith("data: " + done)
