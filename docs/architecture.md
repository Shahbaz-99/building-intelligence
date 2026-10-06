# Architecture

Status: Draft, waiting for instructor approval

This is the architecture for the three-day classroom RAG course project. It is one small FastAPI application. Open WebUI, supplied by the trainer, is the chat client. There is no custom frontend and no second demo.

## Evidence rules

- BNS and IPC documents are the only future answer evidence.
- An answer must not claim support without retrieved evidence.
- Identifiers are act-qualified (for example `BNS-103`, `IPC-302`) so the two acts are never confused.
- The supplied provenance file is `data/raw/PROVENANCE.md`.
- Retrieved passages are evidence. They are never instructions to the application.

## Trust boundaries (light)

- Validate API input.
- Preserve source origin on every retrieved passage.
- Do not put secrets in code, responses, or logs.
- Out of scope: multi-user authorization, a security program, an evaluation harness.

## Fixed embedding choices

| Item | Value |
|---|---|
| Provider | Voyage |
| Model | `voyage-3.5` |
| Version | `voyage-3.5` |
| Dimensions | 1,024 |

These apply to every document embedding and every query embedding. Later stories reuse these names and choices. They do not rename them or add provider-specific alternatives.

## Course modes

One shared registry holds only these modes. Each mode returns an honest `not_implemented` placeholder until its own story adds behavior.

| Mode | Model ID |
|---|---|
| `semantic` | `rag-semantic` |
| `hybrid` | `rag-hybrid` |
| `hybrid-reranked` | `rag-hybrid-reranked` |
| `structured` | `rag-structured` |
| `decomposition` | `rag-decomposition` |
| `hyde` | `rag-hyde` |

## Endpoints

- `GET /healthz`: safe status only. No secrets, no settings values.
- `POST /v1/query`: takes `QueryRequest`, returns `QueryResult`. It owns the diagnostics.
- `GET /v1/models`: lists the six model IDs from the registry.
- `POST /v1/chat/completions`: OpenAI-compatible, text only, JSON and SSE. It maps the selected model to the same `QueryRequest` and `run_pattern` path as `/v1/query`. The server sets `caller_id` (from `WEBUI_DEMO_CALLER_ID`) and `generate_answer`. The client never supplies them.

Open WebUI receives only normal answer text derived from the same `QueryResult`. The full `GenerationResult` stays in `QueryResult.generation`.

## Shared contracts

All contracts live in this project as typed models. Later stories extend them additively and never replace them with simplified alternatives.

- Request: `QueryRequest`, `SemanticFilters`, `ChatCompletionRequest`
- Result: `QueryResult`, `RetrievedChunk`, `SubquestionEvidence`, `StructuredSignals`
- Generation: `GenerationResult`
- OpenAI: chat response, stream chunk, and error envelope shapes
- MongoDB schema: document contract models (no connection in the seed)

No parallel top-level API fields such as `outcome`, `evidence`, `answer`, `confidence`, `citations`, or `diagnostics`.

## Corpus files

- Supplied raw files: `data/raw/`
- Derived files: `data/processed/bns_sections.jsonl` and `data/processed/ipc_sections.jsonl`

## Stack

Python 3.12, UV, FastAPI, Pydantic settings, PyMongo (later use), Ruff, pytest. Configuration comes from `.env` (untracked). `.env.example` is the canonical list of names.

## Out of scope for Story 1.1

Retrieval, PDF parsing, embeddings, MongoDB provisioning, LLM calls, GraphRAG, agentic RAG, and broad deployment.
