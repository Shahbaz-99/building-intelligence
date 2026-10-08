"""Create or reuse the Atlas Search keyword index on chunks.text (Story 4.1).

Run: uv run python -m building_with_rag.ingestion.keyword_index
No Voyage calls, no data writes. A differing index is reported, never replaced.
"""

import time

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings


def _find(coll):
    return next(
        (i for i in coll.list_search_indexes() if i.get("name") == schema.KEYWORD_INDEX_NAME), None
    )


def _ready(idx) -> bool:
    return bool(idx) and (bool(idx.get("queryable")) or idx.get("status") == "READY")


def _diffs(have: dict, want: dict, path: str = "") -> list[str]:
    out = []
    for key in sorted(set(have) | set(want)):
        h, w = have.get(key), want.get(key)
        if isinstance(h, dict) and isinstance(w, dict):
            out += _diffs(h, w, f"{path}{key}.")
        elif h != w:
            out.append(f"{path}{key}: have {h!r}, want {w!r}")
    return out


def main() -> None:
    settings = get_settings()
    if not settings.mongodb_uri:
        raise SystemExit("MONGODB_URI must be set.")
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10000)
    coll = client[settings.mongodb_db_name][schema.CHUNKS_COLLECTION]
    idx = _find(coll)
    if idx is not None:
        diffs = _diffs(
            idx.get("latestDefinition") or idx.get("definition") or {},
            schema.KEYWORD_INDEX_DEFINITION,
        )
        if diffs:
            raise SystemExit(
                f"Index '{schema.KEYWORD_INDEX_NAME}' differs: {'; '.join(diffs)}. "
                "Drop it manually in Atlas, then re-run."
            )
    else:
        coll.create_search_index(
            SearchIndexModel(
                name=schema.KEYWORD_INDEX_NAME,
                type="search",
                definition=schema.KEYWORD_INDEX_DEFINITION,
            )
        )
    status = None
    for _ in range(24):
        idx = _find(coll)
        status = idx.get("status") if idx else "UNKNOWN"
        if _ready(idx):
            break
        time.sleep(5)
    else:
        print(f"{schema.KEYWORD_INDEX_NAME} not ready after 120s (status: {status})")
        raise SystemExit(1)
    print(f"{schema.KEYWORD_INDEX_NAME} {status} queryable")


if __name__ == "__main__":
    main()
