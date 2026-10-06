# Building with RAG (capstone seed)

Run: `uv sync` then `uv run uvicorn building_with_rag.main:app --port 8000`.
Checks: `uv run ruff check . && uv run pytest`.

## Environment

Copy `.env.example` to `.env`. `.env` is untracked (in `.gitignore`); never commit secrets.

- `MONGODB_URI`: Atlas free-tier (M0) connection string. The Atlas IP access list must allow your machine.
- `GENERATION_API_BASE_URL` / `GENERATION_API_KEY`: supplied by the trainer for an OpenAI-compatible LiteLLM proxy. Leave blank; fill in `.env` only when Story 3.1 needs them.
- A free Voyage key is rate limited, so Story 2.2 embedding takes about 40 minutes.
- The app starts with no database or model credentials.

## Open WebUI

Trainer-supplied, run separately (see Story 1.1). Not part of this repo.
