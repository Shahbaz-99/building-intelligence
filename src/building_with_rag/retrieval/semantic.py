"""Semantic retrieval: embed the question, run MongoDB $vectorSearch, resolve passages."""

import threading

from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from voyageai import Client as VoyageClient
from voyageai.error import VoyageError

from building_with_rag.contracts import QueryRequest, RetrievedChunk
from building_with_rag.ingestion.mongodb_schema import (
    CHUNKS_COLLECTION,
    DEFAULT_ACCESS_LEVEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_VERSION,
    EMBEDDINGS_COLLECTION,
    SECTIONS_COLLECTION,
    VECTOR_INDEX_NAME,
)
from building_with_rag.settings import get_settings

MIN_CANDIDATES = 50
MAX_CANDIDATES = 200
MONGO_TIMEOUT_MS = 5000
VOYAGE_TIMEOUT_S = 15

_lock = threading.Lock()
_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


def num_candidates(limit: int) -> int:
    """numCandidates = max(limit, 50), capped at 200."""
    return min(MAX_CANDIDATES, max(limit, MIN_CANDIDATES))


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _not_ready(message: str) -> HTTPException:
    return _error(503, "retrieval_not_ready", message)


def _upstream(message: str) -> HTTPException:
    return _error(502, "retrieval_upstream_error", message)


def _clients() -> tuple[MongoClient, VoyageClient]:
    global _mongo, _voyage
    settings = get_settings()
    if not settings.mongodb_uri or not settings.voyage_api_key:
        raise _not_ready("MONGODB_URI and VOYAGE_API_KEY must be set.")
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=MONGO_TIMEOUT_MS * 3,
            )
        if _voyage is None:
            _voyage = VoyageClient(
                api_key=settings.voyage_api_key, max_retries=1, timeout=VOYAGE_TIMEOUT_S
            )
    return _mongo, _voyage


def _check_ready(db) -> None:
    global _ready
    if _ready:
        return
    try:
        index = next(iter(db[EMBEDDINGS_COLLECTION].list_search_indexes(VECTOR_INDEX_NAME)), None)
        sample = db[EMBEDDINGS_COLLECTION].find_one(
            {}, {"model": 1, "model_version": 1, "dimensions": 1}
        )
    except PyMongoError:
        raise _upstream("MongoDB readiness check failed.") from None
    if index is None or not index.get("queryable"):
        raise _not_ready(f"Index '{VECTOR_INDEX_NAME}' missing or not queryable; run Story 2.2 ingest.")
    if sample is None:
        raise _not_ready("Collection 'embeddings' is empty; run Story 2.2 ingest.")
    if (
        sample.get("model") != EMBEDDING_MODEL
        or sample.get("model_version") != EMBEDDING_MODEL_VERSION
        or sample.get("dimensions") != EMBEDDING_DIMENSIONS
    ):
        raise _not_ready(f"Stored embeddings do not match {EMBEDDING_MODEL}/{EMBEDDING_DIMENSIONS}.")
    _ready = True


def _reset_ready() -> None:
    global _ready
    _ready = False


def effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    """Server scope is public only; caller lists narrow, never widen."""
    out: dict[str, list[str]] = {"access_level": [DEFAULT_ACCESS_LEVEL]}
    caller = request.filters
    if caller:
        if caller.act:
            out["act"] = list(caller.act)
        if caller.status:
            out["status"] = list(caller.status)
    return out


def check_scope(request: QueryRequest) -> None:
    """Reject scope semantic/hybrid mode cannot honour (HTTP 422)."""
    problems = []
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        problems.append("caller_id does not match the effective caller")
    if request.required_acts is not None:
        problems.append("required_acts is not supported by this mode")
    if request.chapter is not None:
        problems.append("chapter is not supported by this mode")
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems) + ".")


def embed_query(voyage: VoyageClient, question: str) -> list[float]:
    try:
        vector = voyage.embed([question], model=EMBEDDING_MODEL, input_type="query").embeddings[0]
    except VoyageError:
        raise _upstream("Voyage embedding request failed.") from None
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise _not_ready(
            f"Query embedding has {len(vector)} dimensions, expected {EMBEDDING_DIMENSIONS}."
        )
    return vector


def vector_search(db, vector: list[float], filters: dict, limit: int, n_cand: int) -> list[dict]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "vector",
                "queryVector": vector,
                "numCandidates": n_cand,
                "limit": limit,
                "filter": {k: {"$in": v} for k, v in filters.items()},
            }
        },
        {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(db[EMBEDDINGS_COLLECTION].aggregate(pipeline))


def resolve_sections(db, hits: list[dict]) -> tuple[dict, dict]:
    chunks = {
        c["chunk_id"]: c
        for c in db[CHUNKS_COLLECTION].find({"chunk_id": {"$in": [h["chunk_id"] for h in hits]}})
    }
    section_ids = list({c["section_id"] for c in chunks.values()})
    sections = {
        s["section_id"]: s
        for s in db[SECTIONS_COLLECTION].find({"section_id": {"$in": section_ids}})
    }
    return chunks, sections


def chunk_result(chunk: dict, section: dict, score: float, **extra) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"],
        section_id=section["section_id"],
        act=section["act"],
        text=chunk["text"],
        heading=section.get("heading") or "",
        score=score,
        chunk_index=chunk.get("chunk_index"),
        act_label=section.get("act_label"),
        status=section.get("status"),
        chapter=section.get("chapter"),
        chapter_title=section.get("chapter_title"),
        section_number=section.get("section_number"),
        source_pdf=section.get("source_pdf"),
        source_sha256=section.get("source_sha256"),
        needs_review=section.get("needs_review"),
        **extra,
    )


def run_semantic(request: QueryRequest) -> dict:
    check_scope(request)
    settings = get_settings()
    filters = effective_filters(request)
    mongo, voyage = _clients()
    db = mongo[settings.mongodb_db_name]
    _check_ready(db)

    n_cand = num_candidates(request.limit)
    vector = embed_query(voyage, request.question)
    try:
        hits = vector_search(db, vector, filters, request.limit, n_cand)
        chunks, sections = resolve_sections(db, hits)
    except PyMongoError:
        _reset_ready()
        raise _upstream("MongoDB query failed.") from None

    results: list[RetrievedChunk] = []
    unresolved = 0
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            unresolved += 1
            continue
        results.append(chunk_result(chunk, section, hit["score"]))

    trace = {
        "mode": "semantic",
        "query": request.question,
        "embedding": {
            "model": EMBEDDING_MODEL,
            "input_type": "query",
            "dimensions": EMBEDDING_DIMENSIONS,
        },
        "index": VECTOR_INDEX_NAME,
        "limit": request.limit,
        "num_candidates": n_cand,
        "filters": filters,
        "caller_id": settings.webui_demo_caller_id,
        "result_count": len(results),
        "ignored": [],
        "unresolved_hits": unresolved,
    }
    if results:
        status = "ok"
        message = (
            f"Returned {len(results)} passage(s). Scores rank similarity only; "
            "they do not prove a passage is correct or answers the question."
        )
    else:
        status = "no_results"
        message = (
            "No passages matched the filters. Scores rank similarity only; "
            "they do not prove correctness."
        )
    return {
        "pattern": "semantic",
        "status": status,
        "message": message,
        "trace": trace,
        "results": results,
    }
