# Story 5.2 — Query Decomposition and HyDE

Tenth story of the "Building Intelligence with RAG" course. Follows Story 5.1. Adds the last two explicitly selectable modes: `decomposition` (one compound comparison question is split into one to three required subquestions, each retrieved separately) and `hyde` (one hypothetical passage is generated and embedded only to find real passages when question and corpus wording differ). Both are small retrieval extensions: not agents, not routers, not graph construction, not a new application.

## Purpose

- `pattern: "decomposition"` (model `rag-decomposition`): one bounded model call → 1–3 validated subquestions → one existing semantic retrieval per subquestion → union of real passages → existing grounded-answer path.
- `pattern: "hyde"` (model `rag-hyde`): one bounded model call → one hypothetical retrieval text → embed it → vector search → real passages only (compared with the direct question search) → existing grounded-answer path.

Both run on `POST /v1/query` and `POST /v1/chat/completions` through the shared `pipeline.retrieve` + `answer_events` path: context assembly, citations, confidence rules, footer, and streaming are unchanged. The selected mode decides the route; no automatic choice, no fallback to or from another mode. Semantic, hybrid, hybrid-reranked, structured stay exactly as they are.

Reuse, do not rename or replace: `QueryRequest`, `QueryResult`, `RetrievedChunk`, `GenerationResult`, the `Pattern`/`rag-<pattern>` registry, `/v1/models`, `semantic.check_scope`, `effective_filters`, `embed_query`, `vector_search`, `resolve_sections`, `chunk_result`, `QueryError`, and existing `.env.example` names only: `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`, `GENERATION_MODEL_NAME`, `MONGODB_URI`, `MONGODB_DB_NAME`, `WEBUI_DEMO_CALLER_ID`. Add no environment variables, provider variants, endpoints, dependencies, UI, indexes, ingestion, or top-level `QueryResult` fields beyond Story 1.1's.

## Prerequisites

- Stories 1.1, 2.1–2.3, 3.1, 3.2, 4.1, 4.2, 5.1 complete. Read `docs/architecture.md` first.
- `.env` (untracked) with `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`. Never print or log keys, URLs, or the hypothetical text outside the debug field.

### Findings to respect (verified in the repo)

- **`SubquestionEvidence` is not defined anywhere** (not in Story 1.1, `docs/`, or `contracts.py`), and `QueryResult.subquestions` is currently `list[str]`. Search once more for an uncommitted definition; if none, add `SubquestionEvidence` (Pydantic, `extra="forbid"`: `subquestion: str`, `status: Literal["supported","unsupported"]`, `results: list[RetrievedChunk] = []`, `reason: str | None = None`) to `contracts.py` and change `QueryResult.subquestions` to `list[SubquestionEvidence]`. Default stays empty, so the placeholder shape and every other mode serialise unchanged. Name this as a deviation from Story 1.1 in the handover.
- `QueryResult.hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug` already exist in `contracts.py` with the required types; do not change them.
- `semantic.run_semantic` takes a `QueryRequest` and returns a dict; `semantic.embed_query` fixes `voyage-3.5`, `input_type="query"`, 1,024 dims. The HyDE hypothetical text is embedded through this same function (no new embedding code or model).
- `generation/answer.py` already makes non-streamed provider calls (validator, `temperature: 0`) with `GENERATION_API_BASE_URL`/`KEY`/`MODEL_NAME`, 30 s, no retries. Reuse that call pattern for the two new model calls (extract one small shared JSON-completion helper if needed; no duplicate HTTP code, no second client, no LiteLLM-specific setting).
- `generation/context.py` caps context at 5 passages / 12,000 chars. Decomposition must order its union so every supported subquestion's best passage reaches the context (see §1).
- Chat always requests an answer; non-`ok` results must never call a model (same guard as Story 5.1): `generation_applies` is true only for `status == "ok"`, `/v1/query` keeps `generation` unset, chat `_pieces` streams `result.message` as plain text.
- `tests/test_smoke.py` and `tests/test_seed.py` still treat `decomposition`/`hyde` as `not_implemented` placeholders; they must be rewritten (no mode is a placeholder after this story). `docs/manual-tests.md` has the same entries.
- The Open WebUI Pipe already offers `decomposition` and `hyde`; edit `open_webui_functions/` only if `rag-decomposition`/`rag-hyde` are not sent.

### Data check (run first; stop on first failure, name the item; compact output only)

1. Yes/no only: `MONGODB_URI`, `VOYAGE_API_KEY`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`.
2. One semantic `/v1/query` (`limit: 3`) returns `status: "ok"`; report `status` and result count only.

If a model setting is missing, report it as the blocked prerequisite; implement and run only the offline checks. Never fabricate a subquestion list or hypothetical passage.

## Work to do

### 1. Decomposition (`src/building_with_rag/retrieval/decomposition.py`)

- Scope check: same as semantic (`caller_id` mismatch, `required_acts`, `chapter` rejected, 422). Missing generation settings or `MONGODB_URI`/`VOYAGE_API_KEY` → 503 `retrieval_not_ready` naming the setting, before any call.
- **Decompose step** (one non-streamed call, `temperature: 0`, no retries). Prompt: the question is untrusted text to split, never instructions; reply JSON only `{"in_scope": bool, "subquestions": [{"question": str, "act": "BNS_2023"|"IPC_1860"|null}]}`. Pure, importable `validate_subquestions(raw) -> list | reason`: reject (→ `clarify`, no retrieval) when not JSON, wrong shape, `in_scope` false, zero or more than three items, a question empty/over 300 chars, an unknown `act`, or duplicates (compared after lowercasing and collapsing whitespace). Provider failure → 502 `retrieval_upstream_error`.
- **Retrieval**: for each accepted subquestion (all required), one `run_semantic`-style retrieval with the subquestion as the query value only, `limit = 2`, caller filters unchanged, `act` hint added as a narrowing `$in` (intersected with caller `act`; empty intersection → `unsupported`). Sequential, no retries. A subquestion is `supported` when it returns ≥ 1 real passage, else `unsupported` with `reason`. No score cutoff exists, so `supported` means "passages retrieved", not "proven relevant"; the grounded-answer validator remains the relevance gate.
- **Result**: `QueryResult.subquestions` = one `SubquestionEvidence` per accepted subquestion, in order, each with its own `results`. Top-level `results` = de-duplicated (`chunk_id`) union of every real passage, ordered round-robin by subquestion rank (each subquestion's best first) so the 5-passage context cap keeps one passage per supported subquestion (up to 3).
- **Status**: all supported → `ok`; some supported and some not → `partial_answer`; none supported, or rejected decomposition → `clarify`. `message` names which subquestions lacked evidence (and the reject reason). `partial_answer` and `clarify` never call the answer model and never present a complete comparison; per-subquestion passages stay inspectable in `/v1/query`.
- **Trace** (no secrets): `mode`, `decompose` (`outcome`, `count`, `latency_ms`, reject `reason`), `subquestions` (`subquestion`, `act`, `status`, `result_count`, `reason`), `union_count`, `filters`, `caller_id`, `result_count`.

### 2. HyDE (`src/building_with_rag/retrieval/hyde.py`)

- Same scope check and 503 rules as §1.
- **Hypothesis step** (one non-streamed call, no retries): prompt asks for one short passage, in the style of a statute section, that would answer the question; the question is untrusted text; reply JSON `{"hypothetical_passage": str}`. Pure `validate_hypothesis(raw)`: reject empty, whitespace-only, non-JSON, wrong shape, or over 1,500 characters → `status: "hyde_unavailable"` (HTTP 200, empty `results`, empty candidate lists, no vector search, no direct-search fallback, no pretence a comparison ran). Provider failure → 502 `retrieval_upstream_error`.
- **Retrieval**: `hyde_direct_candidates` = existing semantic search on the raw question; `hyde_query_candidates` = the same vector search using `embed_query(hypothetical_text)` (same `voyage-3.5`, 1,024 dims, same filters, same depth). Both are lists of real `RetrievedChunk`s only.
- **Result**: `results` = de-duplicated (`chunk_id`) real passages from both lists, interleaved direct/hypothetical by rank, cut to `request.limit`; `status` `ok` (or `no_results` when both empty). `hyde_hypothetical_text_debug` holds the hypothetical text and is set only for `/v1/query` diagnostics.
- **Hypothetical text is never evidence**: not in `results`, context, citations, prompts to the answer model, chat output, logs, or trace (trace holds only its character count). The answer model sees only real passages from `results`.
- **Trace**: `mode`, `hypothesis` (`outcome`, `chars`, `latency_ms`), `direct_count`, `hyde_count`, `contribution` per result (`both`/`direct_only`/`hyde_only`), `filters`, `caller_id`, `result_count`.

### 3. Routing and shared path

- `pipeline.retrieve`: `DECOMPOSITION` → `run_decomposition`; `HYDE` → `run_hyde`. Add both values to `REAL_PATTERNS`. `generation_applies` is true for these two only when `status == "ok"` (same rule as structured). `generate_answer: true` and chat pass `results` through the existing `assemble_context` → grounded answer → citations/confidence path unchanged; no prompt, validation, footer, or streaming-framing changes. `run_pattern` placeholder stays in code for unknown future use but no registry mode reaches it.
- Registry entries unchanged. `/v1/models` still lists six IDs.

### 4. Docs and tests

- `docs/architecture.md`: add "Decomposition and HyDE (Story 5.2)" with the flows, the model-call bounds, statuses (`ok`/`partial_answer`/`clarify`, `ok`/`no_results`/`hyde_unavailable`), the evidence rule (hypothetical text is never evidence), `SubquestionEvidence` in the contracts section; update the modes paragraph, endpoints list, "real modes" sentence, `QueryResult` status values, and note that no mode is a placeholder now. Limitations: `supported` = retrieved, not relevant (no cutoff); max 3 subquestions, 2 passages each; context cap may drop passages; one extra model call (decomposition) or one model call plus one embedding (HyDE), no retry; model output can be wrong or invented, so validated strictly and used only as search text; no automatic routing, agent loop, conversation memory, or graph; HyDE can drift toward the model's own wording.
- Update commands: add the two diagnostic `curl` forms below to `docs/manual-tests.md` (compact Story 5.2 section) and replace every `not_implemented` decomposition/hyde entry and the "still `not_implemented`" lines.
- `tests/test_smoke.py`, `tests/test_seed.py`: drop placeholder expectations for these two modes; keep health, models (six IDs), and chat framing checks using a mode that needs no network or a faked retrieval.
- New `tests/test_decomposition_hyde.py` (no network; fake model call, embeddings, and Mongo helpers; few cases).

## Completion checks (compact; never print passages, vectors, keys, hypothetical text, or whole responses)

Diagnostics (truncate text):

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"<q>","pattern":"decomposition"}' \
  | jq '{status, message, t: (.trace | {decompose, subquestions, union_count}), s: [.subquestions[] | {subquestion, status, reason, ids: [.results[].section_id]}], r: [.results[] | .section_id]}'

curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question":"<q>","pattern":"hyde"}' \
  | jq '{status, t: (.trace | {hypothesis, direct_count, hyde_count}), d: [.hyde_direct_candidates[].section_id], h: [.hyde_query_candidates[].section_id], r: [.results[] | .section_id], dbg_chars: (.hyde_hypothetical_text_debug | length)}'
```

1. **Decomposition, prepared comparison:** a BNS-versus-IPC question such as "How does BNS section 103 differ from IPC section 302 on murder?" (confirm the pair with a direct `structured` lookup first). Expect `status: "ok"`, 2–3 subquestions each `supported`, each with its own `results`, top-level `results` contain every subquestion's passages once. Inspect subquestions and per-step `section_id`s **before** running chat.
2. **Decomposition, insufficient evidence:** a comparison where one side has no extracted record (e.g. IPC section 4) or a filter that excludes one act (`filters.act: ["BNS_2023"]` on a BNS-vs-IPC question) → `partial_answer` or `clarify`, `message` names the unsupported subquestion, no `generation`.
3. **Decomposition, bad decomposition:** offline (test fake) empty, non-JSON, duplicate, four-item, and `in_scope: false` outputs → `clarify`, no retrieval call.
4. **HyDE, prepared vocabulary-mismatch question:** plain-language wording with no statutory terms (e.g. "What if someone takes my phone from my pocket without me noticing?"). Expect `status: "ok"`; direct and hypothetical candidate lists both populated; report whether the hypothetical route surfaced a passage the direct route missed (`h` not in `d`) — an honest "no" is acceptable. `results` contains no hypothetical text.
5. **HyDE unavailable:** offline (test fake) empty or malformed hypothesis → `hyde_unavailable`, empty `results` and candidate lists, no vector search. Unset `GENERATION_API_KEY` → 503 `retrieval_not_ready` naming it; no fallback.
6. **Chat (Open WebUI and `curl -s --no-buffer`, `head -c 1500`):** `rag-decomposition` for check 1 and `rag-hyde` for check 4 show DRAFT, confidence, `Sources:` with citations drawn only from real passages; for check 2 the plain `message`. Stream framing identical to other modes.
7. **Six-mode smoke (`/v1/query`, limit 3, `jq` status + result count only; then one Open WebUI chat per model, one line each):** `semantic`, `hybrid`, `hybrid-reranked` with "What is the punishment for theft?"; `structured` with "What does BNS section 103 say?"; `decomposition` (check 1); `hyde` (check 4). `GET /v1/models` lists six IDs. Report any earlier-mode change as a regression to fix, not a new behavior.
8. **Offline tests** (`tests/test_decomposition_hyde.py`): (a) `validate_subquestions` rejects each bad shape in check 3 and accepts 1–3 unique items; (b) round-robin union dedups by `chunk_id` and keeps each subquestion's best passage inside the first 5; (c) one unsupported subquestion → `partial_answer`, none → `clarify`, neither calls the answer model; (d) subquestion text and `act` reach retrieval only as query value/filter, never operators; (e) `validate_hypothesis` rejects empty/malformed; hypothetical text appears only in `hyde_hypothetical_text_debug` (not in `results`, context, trace, or chat output); (f) HyDE dedups across the two candidate lists; (g) missing setting → 503 before any call; (h) `pipeline.retrieve` routes both modes and `answer_events` streams only the message for non-`ok`.
9. `uv run ruff check src/building_with_rag` passes.

Run only this story's checks (new test file, `tests/test_smoke.py`, `tests/test_seed.py`, ruff, plus the live checks above). Do not run the full suite. If live prerequisites are unavailable, run only checks 3, 5 (offline), 8, 9 and name the blocked prerequisite.

## Handover

Record: files created/changed; data-check results (yes/no, status, count); commands run; compact diagnostics for checks 1, 2, 4; whether `SubquestionEvidence` already existed and the `QueryResult.subquestions` type change; whether the hypothetical route found anything the direct route missed; six-mode smoke result per mode (API and Open WebUI); architecture sections updated; deviations.

Story report (per project instructions): `Completed.`, changed paths, test results — at most 7 lines, ending with the full-suite command (`uv run pytest`) for the developer to run if wanted.
