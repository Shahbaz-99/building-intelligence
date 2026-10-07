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

Expected: `"status":"not_implemented"`, message references `hybrid`.

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
  -d '{"model": "rag-semantic", "messages": [{"role": "user", "content": "What is theft?"}]}'
```

Expected: `"object":"chat.completion"`, `"finish_reason":"stop"`, content contains `not implemented yet`.

### Chat completions — streaming

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-semantic", "messages": [{"role": "user", "content": "What is theft?"}], "stream": true}'
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
