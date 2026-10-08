"""Hybrid retrieval: Atlas Search keyword route + vector route, fused by RRF."""

from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion.mongodb_schema import (
    CHUNKS_COLLECTION,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDINGS_COLLECTION,
    KEYWORD_INDEX_NAME,
    VECTOR_INDEX_NAME,
)
from building_with_rag.retrieval.semantic import (
    _clients,
    _not_ready,
    _upstream,
    check_scope,
    chunk_result,
    effective_filters,
    embed_query,
    num_candidates,
    resolve_sections,
    vector_search,
)
from building_with_rag.settings import get_settings

RRF_K = 60
_KEYWORD_COMMAND = "uv run python -m building_with_rag.ingestion.keyword_index"

_ready = False


def route_depth(limit: int) -> int:
    return max(limit, min(50, max(20, 4 * limit)))


def fuse(semantic_hits: list[dict], keyword_hits: list[dict], limit: int) -> list[dict]:
    """Reciprocal Rank Fusion over ranks (hits are ordered best-first, ranked from 1)."""
    fused: dict[str, dict] = {}
    for route, hits in (("semantic", semantic_hits), ("keyword", keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            row = fused.setdefault(
                hit["chunk_id"],
                {
                    "chunk_id": hit["chunk_id"],
                    "semantic_score": None,
                    "semantic_rank": None,
                    "keyword_score": None,
                    "keyword_rank": None,
                    "fused_score": 0.0,
                },
            )
            if row[f"{route}_rank"] is not None:
                continue
            row[f"{route}_score"] = hit["score"]
            row[f"{route}_rank"] = rank
            row["fused_score"] += 1.0 / (RRF_K + rank)
    ordered = sorted(
        fused.values(),
        key=lambda r: (
            -r["fused_score"],
            r["semantic_rank"] is None,
            r["semantic_rank"] or 0,
            r["chunk_id"],
        ),
    )
    for position, row in enumerate(ordered, start=1):
        row["fused_rank"] = position
    return ordered[:limit]


def _check_ready(db) -> None:
    global _ready
    if _ready:
        return
    try:
        coll = db[EMBEDDINGS_COLLECTION]
        vector = next(iter(coll.list_search_indexes(VECTOR_INDEX_NAME)), None)
        keyword = next(iter(db[CHUNKS_COLLECTION].list_search_indexes(KEYWORD_INDEX_NAME)), None)
        sample = coll.find_one({}, {"_id": 1})
    except PyMongoError:
        raise _upstream("MongoDB readiness check failed.") from None
    if vector is None or not vector.get("queryable"):
        raise _not_ready(f"Index '{VECTOR_INDEX_NAME}' missing or not queryable; run Story 2.2 ingest.")
    if keyword is None or not keyword.get("queryable"):
        raise _not_ready(
            f"Index '{KEYWORD_INDEX_NAME}' missing or not queryable; run: {_KEYWORD_COMMAND}"
        )
    if sample is None:
        raise _not_ready("Collection 'embeddings' is empty; run Story 2.2 ingest.")
    _ready = True


def _reset_ready() -> None:
    global _ready
    _ready = False


def keyword_search(db, question: str, filters: dict, depth: int) -> list[dict]:
    pipeline = [
        {
            "$search": {
                "index": KEYWORD_INDEX_NAME,
                "compound": {
                    "must": [{"text": {"query": question, "path": "text"}}],
                    "filter": [{"in": {"path": k, "value": v}} for k, v in filters.items()],
                },
            }
        },
        {"$limit": depth},
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "section_id": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]
    return list(db[CHUNKS_COLLECTION].aggregate(pipeline))


def run_hybrid(request: QueryRequest) -> QueryResult:
    check_scope(request)
    settings = get_settings()
    filters = effective_filters(request)
    mongo, voyage = _clients()
    db = mongo[settings.mongodb_db_name]
    _check_ready(db)

    depth = route_depth(request.limit)
    n_cand = num_candidates(depth)
    vector = embed_query(voyage, request.question)
    try:
        sem_hits = vector_search(db, vector, filters, depth, n_cand)
        kw_hits = keyword_search(db, request.question, filters, depth)
        chunks, sections = resolve_sections(db, sem_hits + kw_hits)
    except PyMongoError:
        _reset_ready()
        raise _upstream("MongoDB query failed.") from None

    def resolvable(hit: dict) -> bool:
        chunk = chunks.get(hit["chunk_id"])
        return chunk is not None and chunk["section_id"] in sections

    sem_ok = [h for h in sem_hits if resolvable(h)]
    kw_ok = [h for h in kw_hits if resolvable(h)]
    unresolved = len(sem_hits) + len(kw_hits) - len(sem_ok) - len(kw_ok)

    results: list[RetrievedChunk] = []
    contribution = {"both": 0, "semantic_only": 0, "keyword_only": 0}
    for row in fuse(sem_ok, kw_ok, request.limit):
        chunk = chunks[row["chunk_id"]]
        results.append(
            chunk_result(
                chunk,
                sections[chunk["section_id"]],
                row["fused_score"],
                semantic_score=row["semantic_score"],
                semantic_rank=row["semantic_rank"],
                keyword_score=row["keyword_score"],
                keyword_rank=row["keyword_rank"],
                fused_score=row["fused_score"],
                fused_rank=row["fused_rank"],
            )
        )
        if row["semantic_rank"] and row["keyword_rank"]:
            contribution["both"] += 1
        elif row["semantic_rank"]:
            contribution["semantic_only"] += 1
        else:
            contribution["keyword_only"] += 1

    trace = {
        "mode": "hybrid",
        "query": request.question,
        "embedding": {
            "model": EMBEDDING_MODEL,
            "input_type": "query",
            "dimensions": EMBEDDING_DIMENSIONS,
        },
        "filters": filters,
        "caller_id": settings.webui_demo_caller_id,
        "result_count": len(results),
        "unresolved_hits": unresolved,
        "semantic": {
            "index": VECTOR_INDEX_NAME,
            "limit": depth,
            "num_candidates": n_cand,
            "hit_count": len(sem_hits),
        },
        "keyword": {
            "index": KEYWORD_INDEX_NAME,
            "path": "text",
            "operator": "text",
            "limit": depth,
            "hit_count": len(kw_hits),
        },
        "fusion": {
            "method": "rrf",
            "k": RRF_K,
            "weights": {"semantic": 1.0, "keyword": 1.0},
            "route_depth": depth,
        },
        "contribution": contribution,
    }
    if results:
        status = "ok"
        message = (
            f"Returned {len(results)} passage(s) fused from keyword and semantic search. "
            "The fused score ranks only; it does not prove a passage is correct "
            "or answers the question."
        )
    else:
        status = "no_results"
        message = (
            "No passages matched the filters in either route. The fused score ranks only; "
            "it does not prove correctness."
        )
    return QueryResult(pattern="hybrid", status=status, message=message, trace=trace, results=results)
