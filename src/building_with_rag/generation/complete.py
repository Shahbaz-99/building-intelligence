"""One non-streamed JSON-oriented completion for the small Story 5.2 model steps."""

import httpx
from fastapi import HTTPException

from building_with_rag.generation.answer import REQUEST_TIMEOUT_SECONDS, ProviderError, _headers
from building_with_rag.settings import get_settings


def require_settings() -> None:
    """503 naming the first missing setting; checked before any call."""
    settings = get_settings()
    for name, value in (
        ("GENERATION_API_BASE_URL", settings.generation_api_base_url),
        ("GENERATION_API_KEY", settings.generation_api_key),
        ("GENERATION_MODEL_NAME", settings.generation_model_name),
        ("MONGODB_URI", settings.mongodb_uri),
        ("VOYAGE_API_KEY", settings.voyage_api_key),
    ):
        if not value:
            raise HTTPException(
                status_code=503,
                detail={"code": "retrieval_not_ready", "message": f"{name} must be set."},
            )


def complete_text(system: str, user: str) -> str:
    """Return the reply text of one chat completion (temperature 0, no retries)."""
    settings = get_settings()
    body = {
        "model": settings.generation_model_name,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
    }
    try:
        response = httpx.post(
            f"{settings.generation_api_base_url.strip().rstrip('/')}/chat/completions",
            json=body,
            headers=_headers(settings.generation_api_key),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        raise ProviderError(f"http {exc.response.status_code}") from None
    except httpx.TimeoutException:
        raise ProviderError("timeout") from None
    except httpx.HTTPError:
        raise ProviderError("connection error") from None
    except (ValueError, KeyError, IndexError, TypeError):
        raise ProviderError("malformed reply") from None
    return content if isinstance(content, str) else ""
