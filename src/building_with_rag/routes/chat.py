"""OpenAI-compatible chat adapter.

Maps the selected rag-<pattern> model to the same QueryRequest and shared
retrieve + answer_events path as /v1/query, with server-side demo caller_id
and generate_answer. Answer text streams; footer renders from the final result.
Supports normal JSON responses and role/content/stop SSE frames; no custom
SSE events. The OpenAI-style error envelope applies before streaming begins.
"""

import hmac
import json
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from building_with_rag.contracts import (
    ChatCompletionRequest,
    QueryRequest,
    SemanticFilters,
)
from building_with_rag.pipeline import answer_events, closing_text, retrieve
from building_with_rag.registry import MODEL_ID_TO_PATTERN
from building_with_rag.settings import get_settings

router = APIRouter()

_FINISH_STOP = "stop"
_DONE = "[" + "DONE" + "]"


def _envelope(status: int, message: str, code: str, kind: str = "invalid_request_error"):
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": kind, "code": code}},
    )


def _require_key(authorization: str | None = Header(default=None)) -> None:
    expected = get_settings().capstone_api_key
    if not expected:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise _envelope(401, "Invalid API key.", "invalid_api_key", "authentication_error")


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _build_request(request: ChatCompletionRequest) -> QueryRequest:
    """Resolve model to pattern and build the QueryRequest; raise OpenAI-style errors."""
    pattern = MODEL_ID_TO_PATTERN.get(request.model)
    if pattern is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Model '{request.model}' not found.",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )
    latest_user = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if latest_user is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "At least one user message is required.",
                    "type": "invalid_request_error",
                    "code": "missing_user_message",
                }
            },
        )
    options = request.rag_options
    query_request = QueryRequest(
        question=latest_user,
        pattern=pattern,
        caller_id=get_settings().webui_demo_caller_id,  # server-side demo caller
        filters=(
            SemanticFilters(
                act=options.act, status=options.status, access_level=options.access_level
            )
            if options
            else None
        ),
        limit=options.limit if options else 5,
        generate_answer=True,  # server-side decision; adapter never trusts client
        required_acts=options.required_acts if options else None,
        chapter=options.chapter if options else None,
    )
    return query_request


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _retrieve_for_chat(query_request: QueryRequest):
    try:
        return retrieve(query_request)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        raise _envelope(
            exc.status_code,
            str(detail.get("message") or "Retrieval failed."),
            str(detail.get("code") or "retrieval_error"),
            "server_error",
        ) from None


def _pieces(question: str, retrieval):
    """Yield text pieces for the stream; the closing piece comes from the final result."""
    streamed = False
    generation = None
    for kind, payload in answer_events(question, retrieval):
        if kind == "final":
            generation = payload
        elif payload:
            streamed = streamed or kind == "text"
            yield payload
    closing = closing_text(generation, streamed)
    if closing:
        yield closing


@router.post("/v1/chat/completions", dependencies=[Depends(_require_key)])
def chat_completions(request: ChatCompletionRequest):
    query_request = _build_request(request)
    retrieval = _retrieve_for_chat(query_request)
    question = query_request.question
    completion_id = _completion_id()
    created = int(time.time())

    if not request.stream:
        text = "".join(_pieces(question, retrieval))
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": index,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _FINISH_STOP,
                }
                for index in range(request.n)
            ],
        }

    def chunk(delta: dict, finish: str | None = None) -> str:
        return _sse_frame(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"index": i, "delta": delta, "finish_reason": finish} for i in range(request.n)
                ],
            }
        )

    def generate():
        yield chunk({"role": "assistant"})
        for piece in _pieces(question, retrieval):
            yield chunk({"content": piece})
        yield chunk({}, _FINISH_STOP)
        yield "data: " + _DONE + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
