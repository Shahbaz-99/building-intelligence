"""Structured exact retrieval: rule-based classifier + one read-only sections lookup."""

import logging
import re
import threading

from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult, StructuredSignals
from building_with_rag.ingestion.mongodb_schema import SECTIONS_COLLECTION
from building_with_rag.retrieval.semantic import (
    MONGO_TIMEOUT_MS,
    _not_ready,
    _upstream,
    chunk_result,
    effective_filters,
)
from building_with_rag.settings import get_settings

log = logging.getLogger(__name__)

_AGGREGATION = re.compile(r"\bhow many\b|\bcount\b|\btotal number\b", re.IGNORECASE)
_FILTER = re.compile(r"\blist\b|\ball sections\b|\bwhich sections\b|\bsections\s+(?:in|under)\b", re.IGNORECASE)
_SECTION_REF = re.compile(r"(?:\bsection|\bsec\b\.?|(?<![A-Za-z])s\.|§)\s*(\d+(?:\.\d+)?)([A-Za-z]?)\b", re.IGNORECASE)
_ACTS = (
    ("BNS_2023", re.compile(r"\bBNS\b|\bBharatiya\s+Nyaya\s+Sanhita\b", re.IGNORECASE)),
    ("IPC_1860", re.compile(r"\bIPC\b|\bIndian\s+Penal\s+Code\b", re.IGNORECASE)),
)
_CHAPTER = re.compile(r"[A-Za-z0-9 .\-]{1,40}")
_RECOMMEND_OPEN = (
    "Structured mode only looks up a named section (e.g. 'BNS section 103'). "
    "For open questions use the 'semantic' or 'hybrid' mode."
)

_lock = threading.Lock()
_mongo: MongoClient | None = None


def classify(question: str, chapter: str | None = None) -> StructuredSignals:
    """Pure rule-based classification; never guesses an act or section."""

    def sig(status: str, reason: str, **kw) -> StructuredSignals:
        return StructuredSignals(status=status, reason=reason, chapter=chapter, **kw)

    if _AGGREGATION.search(question):
        return sig(
            "recommendation",
            "Aggregation questions are not executed in structured mode; use 'hybrid' or 'semantic'.",
            intent="aggregation",
        )
    if _FILTER.search(question):
        return sig(
            "recommendation",
            "Listing/filter questions are not executed in structured mode; use 'hybrid' or 'semantic'.",
            intent="filter",
        )
    refs = _SECTION_REF.findall(question)
    if not refs:
        return sig("recommendation", _RECOMMEND_OPEN)
    acts = [code for code, rx in _ACTS if rx.search(question)]
    numbers = {(num, suffix) for num, suffix in refs}
    problems = []
    if len(numbers) > 1:
        problems.append("several section numbers named; ask for one section")
    (raw, suffix) = next(iter(sorted(numbers)))
    number = int(raw) if raw.isdigit() and not suffix else None
    if len(numbers) == 1 and (number is None or not 1 <= number <= 999):
        problems.append("section number must be an integer from 1 to 999")
    if not acts:
        problems.append("act missing; say BNS or IPC")
    elif len(acts) > 1:
        problems.append("both BNS and IPC named; ask for one act")
    if problems:
        return sig("clarification_needed", "; ".join(problems).capitalize() + ".", intent="exact_lookup")
    return sig(
        "ok",
        "Exact section reference.",
        intent="exact_lookup",
        act=acts[0],
        section_number=number,
    )


def _sections():
    global _mongo
    settings = get_settings()
    if not settings.mongodb_uri:
        raise _not_ready("MONGODB_URI must be set for structured lookups.")
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=MONGO_TIMEOUT_MS * 3,
            )
    return _mongo[settings.mongodb_db_name][SECTIONS_COLLECTION]


def check_scope(request: QueryRequest) -> None:
    problems = []
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        problems.append("caller_id does not match the effective caller")
    if request.required_acts is not None:
        problems.append("required_acts is not supported by structured mode")
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems) + ".")


def _result(status: str, message: str, trace: dict, results=None) -> QueryResult:
    return QueryResult(
        pattern="structured", status=status, message=message, trace=trace, results=results or []
    )


def run_structured(request: QueryRequest) -> QueryResult:
    check_scope(request)
    if request.chapter is not None and not _CHAPTER.fullmatch(request.chapter):
        raise HTTPException(
            status_code=422,
            detail={"code": "unsupported_option", "message": "chapter must be 1-40 letters, digits, spaces, '.' or '-'."},
        )
    settings = get_settings()
    filters = effective_filters(request)
    signals = classify(request.question, request.chapter)
    if (
        signals.status == "ok"
        and "act" in filters
        and signals.act not in filters["act"]
    ):
        signals = signals.model_copy(
            update={
                "status": "clarification_needed",
                "reason": f"Requested {signals.act} is excluded by the act filter.",
            }
        )
    trace = {
        "mode": "structured",
        "signals": signals.model_dump(),
        "mongodb_called": False,
        "collection": SECTIONS_COLLECTION,
        "filters": filters,
        "caller_id": settings.webui_demo_caller_id,
        "result_count": 0,
    }
    if signals.status != "ok":
        return _result(signals.status, signals.reason, trace)

    predicate: dict = {
        "act": signals.act,
        "section_number": signals.section_number,
        "access_level": {"$in": filters["access_level"]},
    }
    if "status" in filters:
        predicate["status"] = {"$in": filters["status"]}
    if signals.chapter is not None:
        predicate["chapter"] = signals.chapter
    coll = _sections()
    trace["mongodb_called"] = True
    try:
        section = coll.find_one(predicate, {"provenance": 0, "_id": 0})
    except PyMongoError as exc:
        log.warning("structured lookup failed: %s", type(exc).__name__)
        raise _upstream("MongoDB query failed.") from None
    if section is None:
        return _result(
            "not_found",
            "This corpus has no such record. That does not mean the law has no such section.",
            trace,
        )
    chunk = {"chunk_id": section["section_id"], "text": section["text"]}
    results = [chunk_result(chunk, section, 1.0)]
    trace["result_count"] = 1
    trace["record"] = {
        "section_id": section["section_id"],
        "status": section.get("status"),
        "source_status_version": section.get("source_status_version"),
    }
    return _result(
        "ok",
        "Exact record from the supplied corpus (score 1.0 marks an exact match, not similarity). "
        "status/source_status_version describe the source document and do not claim "
        "current legal applicability.",
        trace,
        results,
    )
