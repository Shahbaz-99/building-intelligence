"""The one shared mode registry. Both /v1/query and /v1/chat/completions read it."""

MODES: dict[str, str] = {
    "semantic": "rag-semantic",
    "hybrid": "rag-hybrid",
    "hybrid-reranked": "rag-hybrid-reranked",
    "structured": "rag-structured",
    "decomposition": "rag-decomposition",
    "hyde": "rag-hyde",
}

MODEL_TO_MODE: dict[str, str] = {model_id: mode for mode, model_id in MODES.items()}
