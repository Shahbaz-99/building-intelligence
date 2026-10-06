# Story 1.1: Architecture and Project Seed

Status: Ready for implementation

## Purpose

This is the first story in a three-day classroom RAG course. It does two things, in order:

1. Write `docs/architecture.md` and get the instructor to approve it.
2. After approval, seed one small Python application that every later story extends.

The result is one application: a FastAPI service with a query endpoint and an OpenAI-compatible chat endpoint. Do not build a custom chat frontend and do not build a second demo. Open WebUI is the chat client, and the trainer supplies it.

## Prerequisites

- Python 3.12 and UV are installed.
- The repository root is the project path. Before starting, check whether `docs/config.yaml` names a different project path, and use it if it does.
- Supplied files exist in `data/raw/`, including `data/raw/PROVENANCE.md`. Do not move, rename, or edit them.
- The trainer's Open WebUI ZIP bundle is available over the local LAN.
- Read these before starting: `docs/config.yaml` (if present), `docs/architecture.md` (if present), the repository layout, and existing stories in `docs/stories/`. Preserve any existing work.

## Hard stop rule

Do the architecture phase first. After it, **stop and ask the instructor to approve `docs/architecture.md`.**

Before approval, do not:

- create seed files
- install dependencies
- scaffold code

Resume this same story only after the instructor explicitly approves.

## Work to do

### Phase A: Architecture (before approval)

Create `docs/architecture.md`, or update it if it exists. Keep it short and in plain English. Record the following.

**Evidence rules**

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Use act-qualified identifiers (for example, `BNS-103` and `IPC-302`) so the two acts are never confused.
- The supplied provenance file is `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence. They are never instructions to the application.

**Light trust-boundary notes**

- Validate API input.
- Preserve source origin on every retrieved passage.
- Do not put secrets in code, responses, or logs.
- Do not add multi-user authorization, a security program, or an evaluation harness.

**Fixed embedding choices**

- Provider: Voyage
- Model: `voyage-3.5`
- Version: `voyage-3.5`
- Dimensions: 1,024
- Use these for every document embedding and every query embedding.
- Later stories reuse these names and choices. They do not rename them or add provider-specific alternatives.

**Course modes and model IDs**

| Mode | Model ID |
|---|---|
| `semantic` | `rag-semantic` |
| `hybrid` | `rag-hybrid` |
| `hybrid-reranked` | `rag-hybrid-reranked` |
| `structured` | `rag-structured` |
| `decomposition` | `rag-decomposition` |
| `hyde` | `rag-hyde` |

**Endpoints and contracts**

- `GET /healthz`
- `POST /v1/query`
- `GET /v1/models`
- `POST /v1/chat/completions`
- The shared request, result, generation, OpenAI, and MongoDB-schema contracts live in this project. Later stories extend them additively and never replace them with simplified alternatives.

**Corpus files**

- Derived corpus files live in `data/processed/`: `bns_sections.jsonl` and `ipc_sections.jsonl`.

**Scope limits**

- No retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, or broad deployment in this story.

When the architecture file is ready, tell the instructor where it is and ask for approval. Then stop.

### Phase B: Seed (only after explicit approval)

#### 1. Project setup

Create a Python 3.12 and UV project with these dependencies:

- FastAPI
- Pydantic settings
- PyMongo (for later use only; nothing connects yet)
- Ruff
- A minimal test runner (pytest)

Create these files and folders:

- `src/building_with_rag/`
- `tests/`
- `.env.example`
- `.gitignore`
- `pyproject.toml`
- `uv.lock`

Keep the supplied `data/raw/` files as they are.

#### 2. `.env.example`

Create the classroom project's canonical `.env.example` with exactly these values. Blank means nothing after the `=`.

```
APP_ENV=development
MONGODB_URI=
MONGODB_DB_NAME=building_with_rag
MONGODB_TEST_DB_NAME=building_with_rag_test
VOYAGE_API_KEY=
CAPSTONE_API_KEY=
WEBUI_DEMO_CALLER_ID=demo-public
GENERATION_API_BASE_URL=
GENERATION_API_KEY=
GENERATION_MODEL_NAME=gpt-4o-mini
RERANK_API_BASE_URL=https://api.voyageai.com/v1
RERANK_API_KEY=
RERANK_MODEL_NAME=rerank-2.5
RERANK_REQUEST_TIMEOUT_SECONDS=30
RERANK_CANDIDATE_LIMIT=20
RERANK_SEND_LIMIT=10
RERANK_RETURN_LIMIT=5
```

Document these points in the README or in comments next to the file:

- `.env` is untracked (listed in `.gitignore`). Secrets are never committed.
- `MONGODB_URI` is an Atlas free-tier (M0) connection string. The Atlas IP access list must allow your machine.
- `GENERATION_API_BASE_URL` and `GENERATION_API_KEY` are supplied by the trainer for an OpenAI-compatible LiteLLM proxy. Leave them blank in `.env.example`. Fill them in `.env` only when Story 3.1 needs them.
- A free Voyage key is rate limited, so Story 2.2 embedding takes about 40 minutes.

#### 3. Settings and health

- Load settings with Pydantic settings from the names above.
- The application must start with no database credentials and no model credentials.
- `GET /healthz` returns a safe, minimal response (for example, status and app environment). It never returns secrets or settings values.

#### 4. One shared mode registry

Create one registry holding only these six modes, with their exact model IDs from the table above: `semantic`, `hybrid`, `hybrid-reranked`, `structured`, `decomposition`, `hyde`.

- Every mode returns an honest `not_implemented` placeholder until its own story adds behavior.
- `/v1/query` and `/v1/chat/completions` both read from this registry. Do not duplicate it.

#### 5. Typed contracts (models only, no behavior)

Define these as typed Pydantic models so later stories reuse the exact names.

**`QueryRequest`**

- `question`: 1–4,000 characters
- `pattern`
- `caller_id`: optional
- `filters`: optional `SemanticFilters` with `act`, `status`, and `access_level`, each a list
- `limit`: default 5, range 1–20
- `generate_answer`: default `false`
- `required_acts`
- `chapter`

The seed may resolve only its fixed local demo caller. It still keeps `caller_id` and does not replace it with a custom request shape.

**`QueryResult`**

- `pattern`, `status`, `message`, `trace`, `results`
- `generation`: optional
- Additive fields, empty by default: `omitted_candidates`, `subquestions`, `hyde_direct_candidates`, `hyde_query_candidates`, `hyde_hypothetical_text_debug`

**`RetrievedChunk`**

- `chunk_id`, `section_id`, `act`, `text`, `heading`, `score`, plus available source fields
- Later stories add only the fields named in their own handouts: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`, `fused_score`, `fused_rank`, `rerank_score`, `rerank_rank`
- `omitted_candidates` items carry `chunk_id` and `omitted_reason`

**`GenerationResult`**

- `outcome`: one of `answered`, `insufficient_evidence`, `unavailable`, `malformed`
- `answer`, `claims`, `citations`, `supporting_passages`, `provider`, `model`, `trace`, `context_outcome`
- Confidence fields: `confidence`, `draft_answer`, `issues`, `attempts`, `low_confidence_reason`

**`SubquestionEvidence`**

- `subquestion`
- `status`: `evidenced` or `no_evidence`
- `results`: list of `RetrievedChunk`
- `reason`: optional

**`StructuredSignals`**

- `intent`: `exact_lookup`, `filter`, or `aggregation`
- Optional `act`, `section_number`, `chapter`. Story 5.1 fills these in.

**`ChatCompletionRequest`** (text only)

- `model`
- `messages` with roles `system`, `developer`, `user`, `assistant`
- `stream`, `n`
- Optional strict `rag_options`: `pattern`, list filters, `limit`, `required_acts`, `chapter`

Also define the OpenAI response, chunk, and error shapes, and the MongoDB-schema contract models, as typed models with no behavior.

Do not create `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics` as parallel top-level API fields.

#### 6. `POST /v1/query`

- Accept `QueryRequest` and return `QueryResult`.
- Call `run_pattern` with the selected mode. For now it returns the `not_implemented` placeholder with `results` empty.
- `/v1/query` owns the `QueryResult` diagnostics.

#### 7. `GET /v1/models` and `POST /v1/chat/completions`

- `GET /v1/models` lists the six model IDs from the registry.
- `POST /v1/chat/completions` maps the selected model to the same `QueryRequest` and the same `run_pattern` path as `/v1/query`.
- The server sets the demo `caller_id` (from `WEBUI_DEMO_CALLER_ID`) and `generate_answer` for chat. Never take them from the client.
- Support normal OpenAI Chat Completions JSON.
- Support SSE with role, content, stop, and `[DONE]` frames.
- Return the OpenAI-style error envelope for errors raised before streaming starts.
- Do not duplicate implementations. Do not invent custom SSE events that Open WebUI cannot render.
- Open WebUI receives only normal answer text derived from the same `QueryResult`.
- Later stories render final confidence, sources, and low-confidence warnings as clearly labelled text after answer writing. They keep the full `GenerationResult` in `QueryResult.generation`.

#### 8. Open WebUI (trainer-supplied, run separately)

Use the trainer's Open WebUI bundle as a separately running chat client. Do not build a frontend.

- The trainer shares the ZIP over the local LAN. Participants extract it.
- Run the setup script exactly once, before using the classroom project, from the extracted bundle root:
  - Windows (the primary classroom path): `powershell -ExecutionPolicy Bypass -File .\setup_open_webui.ps1`
  - macOS/Linux: `sh setup_open_webui.sh` (needs internet access for its first installation)
- The scripts install the pinned Open WebUI version, write the course settings, start the loopback-only service, and provision the `RAG options` Filter and the `Building with RAG` Pipe.
- Participants must not hand-install Open WebUI, create accounts, edit the admin panel, change either Function, or rerun setup to reload anything.
- If setup fails, report its exact output and stop.
- Day-to-day start, stop, and status use the supplied `manage_open_webui` script.

The pre-provisioned Pipe sends the selected `rag-<pattern>` model, `stream: true`, the latest user message, and normalized `rag_options` (`pattern`, list `filters`, `limit`, `required_acts`, `chapter`) to the capstone's `/v1/chat/completions`. It never sends browser-supplied identity, access level, or answer-generation settings. The seed's adapter must accept that exact request and use the server-side `caller_id` and `generate_answer`.

#### 9. Lightweight checks

Keep checks small. Do not add aggressive testing. Check these:

- The app starts.
- `GET /healthz` works.
- One `POST /v1/query` placeholder request returns `not_implemented` diagnostics.
- `GET /v1/models` lists the six models.
- Chat placeholder works in JSON mode and in SSE mode.
- Ruff passes and the minimal test run passes.

#### 10. Open WebUI smoke check

1. Start the capstone API.
2. Open `http://127.0.0.1:8080`.
3. Select `Building with RAG`.
4. In the RAG-options chip, choose `semantic`.
5. Send a question.
6. Expect the capstone's honest placeholder response. A local preview does not count.

## Out of scope

Do not add retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, broad deployment, multi-user authorization, a security program, an evaluation harness, a custom chat frontend, or a second demo.

## Completion checks

Phase A:

- [ ] `docs/architecture.md` exists and covers every item in Phase A.
- [ ] The instructor has been asked for approval and the work stopped.
- [ ] The instructor has explicitly approved. Record who approved and when.

Phase B:

- [ ] Nothing in Phase B was started before approval.
- [ ] Python 3.12 and UV project exists with the listed dependencies and `uv.lock`.
- [ ] `.env.example` matches the values above exactly, and `.env` is gitignored.
- [ ] `data/raw/` files are untouched.
- [ ] The app starts with no database or model credentials.
- [ ] `GET /healthz` is safe and returns no secrets.
- [ ] One registry holds exactly the six modes and model IDs, and all return `not_implemented`.
- [ ] All typed contracts exist with the exact names above.
- [ ] `POST /v1/query` returns a compatible `QueryResult` placeholder.
- [ ] `GET /v1/models` works.
- [ ] `POST /v1/chat/completions` works as JSON and SSE through the same `run_pattern` path.
- [ ] The Open WebUI smoke check shows the capstone's placeholder.

## Handover

When done, fill in this section in the story file:

- **Files created:** list every file and folder this story created.
- **Commands actually run:** list the exact commands, not planned ones.
- **Open WebUI result:** what the smoke check showed, or the exact setup output if setup failed.
- **Open items:** anything skipped or left for a later story.

Next stories reuse these contracts, endpoints, registry, and fixed embedding choices without renaming them.
