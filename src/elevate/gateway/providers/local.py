"""Local model servers: Ollama and vLLM, both through their OpenAI-compatible ``/v1`` API."""

from __future__ import annotations

from elevate.gateway.providers.openai_compat import (
    DEFAULT_TIMEOUT_S,
    OpenAICompatibleProvider,
    Transport,
)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"
# Local generation on a laptop CPU can be slow; allow more time than for hosted APIs.
LOCAL_TIMEOUT_S = max(DEFAULT_TIMEOUT_S, 600.0)


def _with_v1(base_url: str) -> str:
    """Ollama's documented base URL has no ``/v1``; its OpenAI-compatible API lives there."""
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def ollama_provider(
    base_url: str | None = None,
    *,
    timeout_s: float = LOCAL_TIMEOUT_S,
    transport: Transport | None = None,
) -> OpenAICompatibleProvider:
    """Ollama adapter; ``base_url`` defaults to ``http://localhost:11434`` (``/v1`` appended)."""
    return OpenAICompatibleProvider(
        name="ollama",
        base_url=_with_v1(base_url or OLLAMA_DEFAULT_BASE_URL),
        api_key=None,
        max_tokens_field="max_tokens",
        timeout_s=timeout_s,
        transport=transport,
    )


def vllm_provider(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    timeout_s: float = LOCAL_TIMEOUT_S,
    transport: Transport | None = None,
) -> OpenAICompatibleProvider:
    """vLLM adapter; ``api_key`` is the server's ``--api-key`` if it was started with one."""
    return OpenAICompatibleProvider(
        name="vllm",
        base_url=base_url or VLLM_DEFAULT_BASE_URL,
        api_key=api_key,
        max_tokens_field="max_tokens",
        timeout_s=timeout_s,
        transport=transport,
    )


__all__ = [
    "LOCAL_TIMEOUT_S",
    "OLLAMA_DEFAULT_BASE_URL",
    "VLLM_DEFAULT_BASE_URL",
    "ollama_provider",
    "vllm_provider",
]
