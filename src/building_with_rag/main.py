import json
import time
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .contracts import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
    OpenAIErrorBody,
    OpenAIErrorEnvelope,
    QueryRequest,
    QueryResult,
)
from .patterns import run_pattern
from .registry import MODEL_TO_MODE, MODES
from .settings import get_settings

app = FastAPI(title="Building with RAG")


class ChatError(Exception):
    def __init__(self, status: int, message: str, type_: str, code: str | None = None):
        self.status, self.message, self.type_, self.code = status, message, type_, code


def _error(status: int, message: str, type_: str, code: str | None = None) -> JSONResponse:
    body = OpenAIErrorEnvelope(error=OpenAIErrorBody(message=message, type=type_, code=code))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(ChatError)
async def _chat_error(_: Request, exc: ChatError) -> JSONResponse:
    return _error(exc.status, exc.message, exc.type_, exc.code)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path.startswith("/v1/chat/"):
        fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) for e in exc.errors())
        return _error(400, f"Invalid request: {fields}", "invalid_request_error")
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "app_env": get_settings().app_env}


@app.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest) -> QueryResult:
    return run_pattern(request)


@app.get("/v1/models", response_model=ModelList)
def models() -> ModelList:
    return ModelList(data=[ModelCard(id=m) for m in MODES.values()])


def _to_query_request(req: ChatCompletionRequest) -> QueryRequest:
    mode = MODEL_TO_MODE.get(req.model)
    if mode is None:
        raise ChatError(
            404, f"Model '{req.model}' not found.", "invalid_request_error", "model_not_found"
        )
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs or not user_msgs[-1].strip():
        raise ChatError(400, "A non-empty user message is required.", "invalid_request_error")
    opts = req.rag_options
    data: dict = {
        "question": user_msgs[-1],
        "pattern": opts.pattern if opts and opts.pattern else mode,
        # Server-set; never taken from the client.
        "caller_id": get_settings().webui_demo_caller_id,
        "generate_answer": False,
    }
    if opts:
        if opts.filters is not None:
            data["filters"] = opts.filters
        if opts.limit is not None:
            data["limit"] = opts.limit
        data["required_acts"] = opts.required_acts
        data["chapter"] = opts.chapter
    try:
        return QueryRequest(**data)
    except ValidationError as exc:
        raise ChatError(400, "Invalid question.", "invalid_request_error") from exc


def _stream(cid: str, created: int, model: str, text: str) -> Iterator[str]:
    def frame(delta: ChatCompletionDelta, finish: str | None = None) -> str:
        choice = ChatCompletionChunkChoice(delta=delta, finish_reason=finish)
        chunk = ChatCompletionChunk(id=cid, created=created, model=model, choices=[choice])
        return f"data: {json.dumps(chunk.model_dump())}\n\n"

    yield frame(ChatCompletionDelta(role="assistant"))
    yield frame(ChatCompletionDelta(content=text))
    yield frame(ChatCompletionDelta(), "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    result = run_pattern(_to_query_request(req))
    text = result.message
    cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    if req.stream:
        return StreamingResponse(
            _stream(cid, created, req.model, text), media_type="text/event-stream"
        )
    return ChatCompletionResponse(
        id=cid,
        created=created,
        model=req.model,
        choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=text))],
    )
