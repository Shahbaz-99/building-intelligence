# Manual tests

## Story 1.1 — Architecture and Project Seed

What it adds: FastAPI project seed with health, query, model-listing, and OpenAI-compatible chat endpoints — all returning honest `not_implemented` placeholders.

Prerequisite: start the API — `uv run uvicorn building_with_rag.app:app --host 127.0.0.1 --port 8000`

### Health

```bash
curl -s http://127.0.0.1:8000/healthz
```

Expected: `{"status":"ok"}`.

### List models

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Expected: six lines: `rag-semantic`, `rag-hybrid`, `rag-hybrid-reranked`, `rag-structured`, `rag-decomposition`, `rag-hyde`.

### Query — each RAG mode (semantic)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "limit": 3}'
```

Expected (needs `.env` credentials and Story 2.2 data): `"status":"ok"`, up to 3 `results` in non-increasing `score` order, populated `trace`. Without credentials: HTTP 503 `retrieval_not_ready`.

### Query — hybrid

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid"}'
```

Expected (needs `.env` and `chunk_text_index`): `"status":"ok"`, `trace.contribution`, per-result `semantic_rank`/`keyword_rank`/`fused_rank`, `score == fused_score`. Index missing: HTTP 503 `retrieval_not_ready`.

### Query — hybrid-reranked

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hybrid-reranked"}'
```

Expected: `"status":"not_implemented"`, message references `hybrid-reranked`.

### Query — structured

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "structured"}'
```

Expected: `"status":"not_implemented"`, message references `structured`.

### Query — decomposition

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "decomposition"}'
```

Expected: `"status":"not_implemented"`, message references `decomposition`.

### Query — hyde

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "hyde"}'
```

Expected: `"status":"not_implemented"`, message references `hyde`.

### Query — empty question (failure)

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "", "pattern": "semantic"}'
```

Expected: 422 validation error (question below min_length 1).

### Chat completions — JSON (non-streaming)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet`.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-hybrid-reranked", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
```

Expected: SSE `data:` frames with `delta` role then content, ending with `data: [DONE]`.

### Chat completions — invalid model (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Expected: 400 with `"type":"invalid_request_error"`, `"code":"model_not_found"`.

### Chat completions — no user message (failure)

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "messages": [{"role": "system", "content": "You are helpful."}]}'
```

Expected: 400 with `"code":"missing_user_message"`.

## Story 2.1 — Document Ingestion

What it adds: PDF extraction script that produces inspectable JSONL section-level corpora from the BNS and IPC bare act PDFs.

Prerequisite: Story 1.1 complete, `data/raw/` PDFs and `PROVENANCE.md` present, `uv sync` done.

### Run extraction

```bash
uv run python scripts/extract_sections.py
```

Expected: prints parser name/version, record counts (~358 BNS, ~500 IPC), count of `needs_review` flags, count of empty-text records. Both `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl` exist.

### Re-run (skip)

```bash
uv run python scripts/extract_sections.py
```

Expected: prints "BNS corpus up to date — skipping" and "IPC corpus up to date — skipping". No records appended or overwritten.
## Story 3.1 — Grounded Answer Generation

What it adds: `generate_answer: true` on a semantic `/v1/query` returns a grounded `generation` (`answered`, `insufficient_evidence`, `unavailable`, or `malformed`) with resolved citations.

Prerequisite: start the API as in Story 1.1; `.env` needs `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` (never print them).

### Answerable question

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft under the BNS?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, model, provider, context_outcome, text: (.text[:300]), claims: [.claims[] | {t: .text[:80], e: .evidence_labels}], citations: [.citations[] | {label, chunk_id, section_id, act, heading}], trace}), ctx: [.results[] | {chunk_id, section_id, act, score}]}'
```

Expected: `status` `ok`, `generation.outcome` `answered`, non-empty `text`, claims with labels, `citations` whose `chunk_id` appear in `ctx`.

### Unsupported question (edge case)

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is the GST rate on restaurant services?", "pattern": "semantic", "limit": 5, "generate_answer": true}' \
  | jq '{status, g: (.generation | {outcome, text, claims, citations}), n: (.results | length)}'
```

Expected: `outcome` `insufficient_evidence`, empty `text`, no claims or citations, `n` > 0.

### Unavailable generation (failure)

Set `GENERATION_API_KEY=` (empty) in `.env`, restart the API, rerun the first command.

Expected: HTTP 200, `generation.outcome` `unavailable`, empty `text`, `results` still present.

### Without generate_answer

```bash
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What is theft?", "pattern": "semantic", "limit": 3}' | jq '{status, generation}'
```

Expected: `status` `ok`, `generation` null.

## Story 3.2 — Streamed Answers with Confidence

What it adds: `/v1/chat/completions` for `rag-semantic` streams a grounded answer (`DRAFT` line, labelled text, confidence/sources footer) from the same result as `/v1/query`.

Prerequisite: start the API as in Story 1.1; `.env` needs `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` (never print them). Add `-H "Authorization: Bearer <key>"` only if `CAPSTONE_API_KEY` is set.

```bash
# Supported question, streamed
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model":"rag-semantic","stream":true,"messages":[{"role":"user","content":"What is the punishment for theft under the BNS?"}]}'   | grep '^data: {' | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' | head -c 1500

# Unsupported question, streamed
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model":"rag-semantic","stream":true,"messages":[{"role":"user","content":"What is the GST rate on restaurant services?"}]}'   | grep '^data: {' | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' | head -c 1500

# Same question on /v1/query for comparison
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question":"What is the punishment for theft under the BNS?","pattern":"semantic","limit":5,"generate_answer":true}'   | jq '{status, g: (.generation | {outcome, confidence, attempts, issues, citations: [.citations[] | {label, section_id}]})}'
```

Expected: first command prints `DRAFT — checking evidence`, answer text with `[E1]`-style labels, then `Evidence check passed — confidence: high` and `Sources:` lines. Second prints one insufficient-evidence sentence and no confidence. Third shows `status` `ok`, `outcome` `answered`, `confidence` `high`, and the same citations.

```bash
# Stream shape: ends with [DONE]
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model":"rag-semantic","stream":true,"messages":[{"role":"user","content":"What is theft?"}]}' | tail -n 3

# Failure: wrong Bearer (only when CAPSTONE_API_KEY is set)
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -H "Authorization: Bearer wrong"   -d '{"model":"rag-semantic","messages":[{"role":"user","content":"What is theft?"}]}'
```

Expected: the stream's final line is `data: [DONE]`, preceded by a chunk with `"finish_reason": "stop"`. Wrong Bearer returns 401 with `"code":"invalid_api_key"`.

## Story 4.1 — Hybrid search

```bash
uv run python -m building_with_rag.ingestion.keyword_index   # ends: chunk_text_index READY queryable
curl -s http://127.0.0.1:8000/v1/query -H "Content-Type: application/json"   -d '{"question":"criminal breach of trust","pattern":"hybrid","limit":5}'   | jq '{status, t: (.trace | {semantic, keyword, fusion, contribution}), r: [.results[] | {section_id, score, sr: .semantic_rank, kr: .keyword_rank, fr: .fused_rank, text: .text[:60]}]}'
curl -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json"   -d '{"model":"rag-hybrid","stream":true,"messages":[{"role":"user","content":"criminal breach of trust"}]}' | head -c 1500
```

Expected: `status` `ok`, results in non-increasing `score`, each with a `semantic_rank` or `keyword_rank`; chat shows DRAFT, confidence, Sources. `hybrid-reranked`, `structured`, `decomposition`, `hyde` still `not_implemented`.
