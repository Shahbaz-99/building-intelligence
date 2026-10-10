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

`semantic` (Story 2.3), `hybrid` (Story 4.1) `hybrid-reranked` (Story 4.2) `structured` (Story 5.1), `decomposition` and `hyde` (Story 5.2) are real on `POST /v1/query`; chat for their `rag-*` models runs the same path with streamed answers (Story 3.2). No mode is a placeholder now; `run_pattern` stays only as an unused fallback.

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
- `subquestions` is a list of `SubquestionEvidence` (`subquestion`, `status` `supported`/`unsupported`, `results`, optional `reason`; Story 5.2 retypes the earlier `list[str]`; empty for other modes).
- Additive empty-by-default fields later modes use: `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`.
- No parallel top-level `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` fields.

### `RetrievedChunk`

A retrieved passage is always this shape: `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, and available source fields. Hybrid adds optional `semantic_score/rank`, `keyword_score/rank`, `fused_score/rank` (all `None` in semantic mode). Re-ranking adds optional `rerank_score`, `rerank_rank`, `omitted_reason`; `QueryResult.omitted_candidates` now holds re-rank cuts.

### Endpoints

- `GET /healthz` — safe, no credentials required.
- `POST /v1/query` — accepts `QueryRequest`, returns `QueryResult`; `semantic` goes to `retrieval/semantic.py`, `hybrid` to `retrieval/hybrid.py`, `hybrid-reranked` to `retrieval/rerank.py`, `structured` to `retrieval/structured.py`, `decomposition` to `retrieval/decomposition.py`, `hyde` to `retrieval/hyde.py`.
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

Flow: result from any real mode (semantic, hybrid, hybrid-reranked, structured with status `ok`) -> bounded labelled context (`generation/context.py`: max 5 passages, 12,000 chars, labels `E1..`, no mid-text cuts) -> one `POST {GENERATION_API_BASE_URL}/chat/completions` (`generation/answer.py`, httpx, 30 s, no retries) -> strict JSON parse -> citations resolved from supplied context only.

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

## Re-ranking (Story 4.2)

- Settings (defaults): `RERANK_API_KEY` (empty; required), `RERANK_API_BASE_URL` (`https://api.voyageai.com/v1`), `RERANK_MODEL_NAME` (`rerank-2.5`), `RERANK_REQUEST_TIMEOUT_SECONDS` (30), `RERANK_CANDIDATE_LIMIT` (20), `RERANK_SEND_LIMIT` (10), `RERANK_RETURN_LIMIT` (5). Valid when `1 <= RETURN <= SEND <= CANDIDATE <= 20` and timeout >= 1; else 503 `retrieval_not_ready` naming the setting. Checked before any Voyage/MongoDB call.
- Request: one `POST {base}/rerank` (httpx, Bearer key, `{model, query, documents}`, no `top_k`, no retries). Document = `"{heading}
{chunk text}"`.
- Reply validation: `data` non-empty list; `index` int, in sent range, unique; `relevance_score` finite number; all sent candidates scored. Else 502 `retrieval_upstream_error`. No score is invented.
- Selection (`rerank.select`): candidates = top `RERANK_CANDIDATE_LIMIT` hybrid results; first `RERANK_SEND_LIMIT` sent, rest `omitted_reason: "not_sent_to_reranker"`; sent ordered by `relevance_score` desc (ties `fused_rank`), `rerank_rank` 1-based; final = first `min(limit, RERANK_RETURN_LIMIT)`, rest `"below_return_limit"` (scores kept).
- Result: `results` ordered by `rerank_rank`, `score == rerank_score`, hybrid fields kept as "before"; `omitted_candidates` in `fused_rank` order. Hybrid `no_results` -> `no_results`, no provider call.
- Trace: `mode`, `query`, `filters`, `caller_id`, `result_count`, `hybrid` (embedding, semantic, keyword, fusion, contribution, unresolved_hits), `rerank` (model, limits, counts, `latency_ms`, `usage_tokens`).
- Outcomes: `ok`, `no_results`, 503 (config), 502 (provider). No fallback to hybrid.
- Limitations: candidates beyond the top `RERANK_CANDIDATE_LIMIT` are never seen; each passage is scored independently against the question; scores are model-specific, uncalibrated, not comparable with fused scores or across questions; no score cutoff; one extra provider call per request (latency, cost, rate limits); no retry, no fallback; answer context cap (5 passages / 12,000 chars) still applies.
- Diagnostic: `curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" -d '{"question":"<q>","pattern":"hybrid-reranked","limit":5}'`; read `trace.rerank`, then `results` and `omitted_candidates` (`fused_rank`, `rerank_rank`, `omitted_reason`), truncating `text`.

## Structured exact retrieval (Story 5.1)

- `StructuredSignals` (`contracts.py`, `extra="forbid"`): `intent`, `act`, `section_number` (1-999), `chapter`, `status` (`ok`/`recommendation`/`clarification_needed`), `reason`. `retrieval/structured.py::classify` is pure and rule-based (no LLM, no I/O).
- Classified: `section N` / `sec. N` / `s. N` / `§N` (integer) plus one act token (BNS, IPC, full names) -> `ok`. Refused: aggregation ("how many", "count", "total number") and filter ("list", "all sections", "which sections", "sections in/under") -> `recommendation`, never executed; no act, both acts, several numbers, non-integer or out-of-range number -> `clarification_needed` (acts never guessed); anything else -> `recommendation` (use `semantic`/`hybrid`).
- Exact-input contract: only classifier output (validated `act`, `section_number`) plus server-fixed `access_level=["public"]`, optional `status` filter and optional validated `chapter` (1-40 of letters, digits, space, `.`, `-`; else 422 `unsupported_option`; never parsed from the question) reach MongoDB. Raw question text never does. One read-only `find_one` on `sections`, projection without `provenance`. No Voyage or index use.
- Outcomes: `ok` (one `RetrievedChunk`, `chunk_id = section_id`, `score = 1.0` = exact match, not similarity), `not_found` (this corpus has no record; not a claim about the law), `clarification_needed`, `recommendation`; 503 `retrieval_not_ready` (no `MONGODB_URI`, only when a lookup is needed); 502 `retrieval_upstream_error`. Scope: `caller_id` mismatch and `required_acts` rejected (422); `chapter` accepted.
- Answer boundary: retrieval returns the exact record; explanation only through the existing grounded-answer path when `generate_answer` is true (always in chat). Non-`ok` results never call a model; chat streams `message` as plain text. `status`/`source_status_version` are source metadata, not current legal applicability.
- Limitations: integer sections only (no `103A`); no multi-section or cross-act comparison; filter/aggregation recognised but not executed; IPC sections 4, 5, 18, 34, 40, 75, 161-165 absent from `sections`; rule-based phrasing misses; a section over the 12,000-char context cap gives `insufficient_evidence` when answered (direct inspection still returns it).

## Decomposition and HyDE (Story 5.2)

Both reuse existing settings only (`VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`/`KEY`/`MODEL_NAME`, `MONGODB_URI`); a missing one is 503 `retrieval_not_ready` naming it, before any call. Model replies come from `generation/complete.py` (one non-streamed call, `temperature: 0`, 30 s, no retries; provider failure is 502 `retrieval_upstream_error`). Statuses other than `ok` never call the answer model; chat streams `message` as plain text.

- **Decomposition**: one model call -> `validate_subquestions` (1-3 unique items, each at most 300 chars, optional `act`, `in_scope` true) -> per accepted subquestion one semantic vector search (limit 2; the `act` hint only narrows the `$in` filter) -> `QueryResult.subquestions` with per-step passages; top-level `results` = de-duplicated union ordered round-robin so each subquestion's best passage survives the 5-passage context cap. Rejected decomposition -> `clarify`, no retrieval. All subquestions supported -> `ok`; some -> `partial_answer`; none -> `clarify`. Trace: `decompose`, `subquestions`, `union_count`.
- **HyDE**: one model call -> `validate_hypothesis` (JSON `hypothetical_passage`, non-empty, at most 1,500 chars; else `hyde_unavailable`, no search, no fallback) -> embed it with `embed_query` (`voyage-3.5`) -> `hyde_query_candidates`; the raw question search -> `hyde_direct_candidates`; `results` = de-duplicated interleave cut to `limit` (`trace.contribution`: `both`/`direct_only`/`hyde_only`). Statuses `ok`, `no_results`, `hyde_unavailable`.
- **Evidence rule**: the hypothetical text is search input only. It lives in `hyde_hypothetical_text_debug`; never in `results`, context, citations, trace (only its length), or chat.
- **Limitations**: `supported` means passages were retrieved (no score cutoff), not that they are relevant; the grounded-answer validator is the relevance gate. At most 3 subquestions with 2 passages each; the context cap can drop passages. Extra model call per request, no retry. Model output may be wrong, so it is validated strictly and used only as search text. HyDE can drift toward the model's wording. No automatic routing, agent loop, conversation memory, or graph.
- Diagnostics: see `docs/manual-tests.md` (Story 5.2).
