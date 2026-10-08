# Capstone Architecture — Building Intelligence with RAG

Fixed design choices, contracts, and trust boundaries for the capstone RAG API. Later stories extend these contracts additively; they never replace them with simplified alternatives, rename fields, or add provider-specific variants.

## Scope

One small application: the capstone RAG API. No chat frontend (Open WebUI is a separate trainer-supplied client) and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Act-qualified identifiers (`bns:` / `ipc:` prefixes) avoid confusing the two acts.
- Supplied provenance lives at `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence, never application instructions.

## Trust boundaries

Kept light for this course:

- Validate API input.
- Preserve source origin on retrieved passages.
- Do not put secrets in code, responses, or logs.
- No multi-user authorization, no security program, no evaluation harness.

## Fixed embedding choices

- Model: `voyage-3.5`, model version `voyage-3.5`, 1,024 dimensions.
- Used for every document and query embedding.
- Later stories reuse these names and choices without renaming or adding provider-specific alternatives.

## Course modes and shared registry

One shared registry (single source of truth) holds only:

| Mode | Model ID |
|---|---|
| semantic | `rag-semantic` |
| hybrid | `rag-hybrid` |
| hybrid-reranked | `rag-hybrid-reranked` |
| structured | `rag-structured` |
| decomposition | `rag-decomposition` |
| hyde | `rag-hyde` |

`semantic` (Story 2.3) and `hybrid` (Story 4.1) are real on `POST /v1/query`; chat for `rag-semantic` and `rag-hybrid` runs the same path with streamed answers (Story 3.2); every other mode still returns an honest `not_implemented` placeholder until its own story adds behavior.

## API contracts

### `QueryRequest` (`POST /v1/query`)

- `question`: string, 1–4,000 characters, required.
- `pattern`: one of the six modes.
- `caller_id`: optional.
- `filters`: optional `SemanticFilters` — `act`, `status`, `access_level`, each a list.
- `limit`: default 5, range 1–20.
- `generate_answer`: default false.
- `required_acts`: optional.
- `chapter`: optional.

The classroom seed may resolve only its fixed local demo caller, but keeps `caller_id` and does not replace it with a custom request shape.

### `QueryResult`

- `pattern`, `status`, `message`, `trace`, `results`.
- Optional `generation`.
- Additive empty-by-default fields later modes use: `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Hybrid adds optional `semantic_score/rank`, `keyword_score/rank`, `fused_score/rank` (all `None` in semantic mode).

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`; `semantic` goes to `retrieval/semantic.py`, `hybrid` to `retrieval/hybrid.py`, other modes return the `run_pattern` placeholder.
- `GET /v1/models` — lists the six `rag-<pattern>` model IDs.
- `POST /v1/chat/completions` — OpenAI-compatible, text-only `ChatCompletionRequest`: `model`, `messages` with `system`/`developer`/`user`/`assistant` roles, `stream`, `n`, and optional strict `rag_options` (`pattern`, list filters, `limit`, `required_acts`, `chapter`).

The chat adapter maps the selected model to the same `QueryRequest` and shared `pipeline.retrieve` + `answer_events` path as `/v1/query`; it sets server-side demo `caller_id` and `generate_answer`, and accepts the Pipe's nested `rag_options.filters.{act,status}` as well as flat `act`/`status`. Supports normal OpenAI Chat Completions JSON responses and role/content/stop frames, plus the OpenAI-style error envelope before streaming begins. No duplicated implementations, no custom SSE events that Open WebUI cannot render.

## Semantic retrieval (Story 2.3)

Flow: validate -> scope filters -> embed query -> `$vectorSearch` -> resolve chunk/section -> `QueryResult`.

- Validate: `question` trimmed, non-empty; `SemanticFilters` forbids unknown fields and only accepts known `act`/`status`/`access_level` strings; `caller_id` must be omitted or equal `WEBUI_DEMO_CALLER_ID`; `required_acts` and `chapter` rejected (HTTP 422).
- Scope: server fixes `access_level=["public"]`; caller `act`/`status` lists only narrow (`$in` inside the `$vectorSearch` `filter`).
- Embed: raw question, `voyage-3.5`, `input_type="query"`, must return 1,024 dims. `numCandidates = min(200, max(limit, 50))`.
- Resolve: hit `chunk_id` -> `chunks` (text) + `sections` (heading, chapter, source). Score order kept; chunks not de-duplicated; unresolved hits omitted and counted in `trace.unresolved_hits`.
- Outcomes: `ok` (passages); `no_results` (HTTP 200, filters match nothing); 503 `retrieval_not_ready` (missing credentials, index not queryable, empty/mismatched embeddings); 502 `retrieval_upstream_error` (Voyage/MongoDB error). No score cutoff; scores rank similarity only.
- `generate_answer: true` on `/v1/query` triggers grounded generation (see Story 3.1 below).
- Added optional `RetrievedChunk` fields: `chunk_index`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `source_pdf`, `source_sha256`, `needs_review`.

Diagnostic (text truncated):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question": "What is the punishment for theft?", "pattern": "semantic", "limit": 3}'   | jq '{status, trace, results: [.results[] | {chunk_id, section_id, act, heading, score, text: .text[:80]}]}'
```

## Context and answer boundaries (Story 3.1)

Flow: semantic or hybrid result -> bounded labelled context (`generation/context.py`: max 5 passages, 12,000 chars, labels `E1..`, no mid-text cuts) -> one `POST {GENERATION_API_BASE_URL}/chat/completions` (`generation/answer.py`, httpx, 30 s, no retries) -> strict JSON parse -> citations resolved from supplied context only.

- Outcomes (`GenerationResult.outcome`): `answered` (text, claims, citations, supporting passages), `insufficient_evidence` (also when no passages; no model call), `unavailable` (missing settings, timeout, connection error, non-2xx), `malformed` (non-JSON, no `choices`, unknown label, rule violation; no repair or retry). Non-answered outcomes carry empty text, claims, citations.
- HTTP 200 for unavailable/malformed; retrieval `results` and `status` are unchanged. No URL or key in messages or trace.
- Added optional `GenerationResult` fields: `outcome`, `claims`, `citations`, `supporting_passages`, `provider`, `trace`, `context_outcome`.
- Evidence is untrusted source text, never instructions. No claims of current legal applicability beyond the supplied BNS/IPC documents.
- The single-call JSON flow above is superseded by the streamed flow in Story 3.2 below.

## Open WebUI (trainer-supplied, separate client)

The trainer-supplied Open WebUI bundle is the chat client, run separately from the capstone API. Its pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The capstone's adapter accepts that exact request and uses server-side `caller_id`/`generate_answer`.

`/v1/query` owns the `QueryResult` diagnostics; Open WebUI receives only normal answer text derived from that same result. Final confidence, sources, and low-confidence warnings render as labelled text after answer writing (Story 3.2); the full `GenerationResult` stays in `QueryResult.generation`.

## Environment

- `.env` is untracked; secrets are never committed.
- The application must start with no database or model credentials and expose a safe `GET /healthz`.
- Canonical environment values are defined in `.env.example`.

## Corpus

Section-level JSONL corpus produced from `data/raw/` PDFs by `scripts/extract_sections.py`. One JSON object per line, one record per section. No MongoDB, no embeddings, no vector indexes.

### Parser

- **Library**: `pymupdf` (fitz) 1.28.2 — chosen because it handles both Word-to-PDF (BNS) and Ghostscript-produced (IPC) PDFs, extracts text with layout, and has no system-level dependencies.
- **Extraction command**: `uv run python scripts/extract_sections.py`

### Output format

JSONL files at `data/processed/`:

| File | Records | Sections |
|---|---|---|
| `data/processed/bns_sections.jsonl` | 358 | 1–358 |
| `data/processed/ipc_sections.jsonl` | 500 | 1–511 (11 unextractable) |

Each record has 14 fields: `section_id`, `act`, `act_label`, `status`, `chapter`, `chapter_title`, `section_number`, `heading`, `text`, `source_pdf`, `source_sha256`, `parser`, `parser_version`, `source_status_version`, `needs_review`.

### Known limitations

- **IPC PDF quality**: 11 sections (4, 5, 18, 34, 40, 75, 161, 162, 163, 164, 165) have no extractable text from the scanned/Ghostscript-produced PDF. Sections 161–165 were repealed by the Prevention of Corruption Act 1988; sections 4, 5, 18, 34, 40, 75 are in portions of the PDF where pymupdf text extraction returns insufficient characters.
- **IPC section headings**: Some IPC sections (e.g. 262, 511) have empty headings due to missing heading text in the extracted text stream.
- **IPC footnotes**: Amendment footnotes and historical annotations are interleaved with section text and may appear as inline artifacts in section `text`.
- **BNS chapter markers**: Chapter boundaries are detected from `CHAPTER <roman>` lines in the body text. The BNS index (pages 2–19) provides section headings; the correspondence table (pages 20–73) is skipped.
- **Source-hash safety rule**: If a source PDF hash changes, the corpus for that act is regenerated as an atomic replacement. Records from different PDF versions are never mixed in one corpus file.

## Streamed answers and confidence (Story 3.2)

- Shared path: `pipeline.retrieve` -> `pipeline.answer_events` (`generation/answer.py: stream_answer`). `/v1/query` drains the events into `QueryResult.generation`; chat forwards `text`/`notice` events as SSE content chunks and appends a footer from the same final `GenerationResult`. One request = one generation/validation operation. Retrieval and context assembly are not streamed.
- Event flow: per attempt, `notice` `DRAFT — checking evidence`, streamed `text`, validation; failed non-final attempt -> `notice` `Check failed: ... Retrying (attempt 2 of 2)…`; last event `final`. The first characters are buffered so `INSUFFICIENT_EVIDENCE: <reason>` is never streamed as an answer.
- `MAX_ATTEMPTS = 2`. Checks per attempt, in order: `citation_labels` (all labels supplied), `claim_cited` (every sentence/bullet has a label, text non-empty), `support` (one non-streamed validator call, `temperature: 0`; skipped when the first two fail). Invalid citations are never stripped.
- New `GenerationResult` fields: `confidence` (`high` = final attempt passed; `low` = final attempt failed a check; absent = nothing judged), `issues` (`attempt`, `check`, `detail`, all attempts), `attempts` (`attempt`, `status`, `chars`, `latency_ms`), `draft_answer` (last unpassed text), `low_confidence_reason`.
- Outcome mapping: passed -> `answered` (+ `high`); failed final check -> `malformed`, empty `text`, `draft_answer`, `low`; provider failure -> `unavailable`; validator unreachable -> `unavailable` with `draft_answer`; validator invalid reply -> `malformed`; `insufficient_evidence` and empty context unchanged.
- Chat labels: `DRAFT — checking evidence`, `Evidence check passed — confidence: high` + `Sources:` lines (`E1 · BNS §303 · Theft · bns:303`), `DRAFT — low confidence, not the final answer.` + reason and failed checks. Drafts cannot be retracted once streamed.
- Failure after text began: final line `Answer generation unavailable — the text above is an unchecked draft.`, then `stop` and `[DONE]`; HTTP stays 200 once streaming starts. Errors before streaming (auth, model, retrieval) use the OpenAI-style envelope.
- `CAPSTONE_API_KEY`: when non-empty, `/v1/chat/completions` requires `Authorization: Bearer <key>` (constant-time compare, 401 `invalid_api_key`). Empty = no check. Other endpoints unchanged.

## Hybrid retrieval (Story 4.1)

- Mechanism: Atlas Search `$search` with the `text` operator (BM25, `{$meta: "searchScore"}`) on `chunks.text`; not `$text`, not `vectorSearch`.
- Index `chunk_text_index` on `chunks` (definition: `KEYWORD_INDEX_DEFINITION` in `ingestion/mongodb_schema.py`; `text` `lucene.standard`, `act`/`status`/`access_level` token). Create/reuse: `uv run python -m building_with_rag.ingestion.keyword_index`. A differing index is reported, never replaced.
- Filters: `$search.compound.filter` with `in` on the token fields; same effective filters as semantic (`access_level=["public"]` fixed; caller lists narrow only). The question is only the `text.query` value.
- Fusion: Reciprocal Rank Fusion over ranks. `ROUTE_DEPTH = max(limit, min(50, max(20, 4*limit)))` per route; `fused_score = sum 1/(RRF_K + rank)`, `RRF_K = 60`, equal weights. Order: fused desc, `semantic_rank` (missing last), `chunk_id`. Fused by `chunk_id`; `fused_rank` = position. `score` = `fused_score`.
- `RetrievedChunk` fields: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`; a route that did not return the chunk leaves its pair `None`.
- Trace: `mode`, `query`, `embedding`, `filters`, `caller_id`, `result_count`, `unresolved_hits`, `semantic`, `keyword`, `fusion`, `contribution` (`both`/`semantic_only`/`keyword_only`).
- Outcomes: `ok`, `no_results` (both routes empty); 503 `retrieval_not_ready` (keyword or vector index missing/not queryable); 502 `retrieval_upstream_error`. No silent fallback to semantic.
- Limitation: rank-only fusion ignores score magnitude; `text` matches any query term (OR), so long questions pull in common words; section numbers match only inside chunk `text`; no stemming/synonyms beyond the standard analyzer.
- Diagnostic: `curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"<q>","pattern":"hybrid","limit":5}'`; read `trace.contribution` and per-result `semantic_rank`/`keyword_rank`/`fused_rank` (truncate `text`).
- Limitations update: Atlas required; no `$text`. The Atlas Search index `chunk_text_index` on `chunks` replaces "no keyword/hybrid index fields".
