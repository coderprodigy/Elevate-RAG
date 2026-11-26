"""Anthropic Messages API adapter.

Uses the optional ``anthropic`` SDK (``uv sync --extra providers``), imported on first use.
The SDK's own retries are disabled (``max_retries=0``) because the gateway retries.
Anthropic has no seed parameter and no embeddings endpoint.
"""

from __future__ import annotations

import importlib
from typing import Any

from elevate.gateway.providers.base import (
    PROVIDERS_EXTRA_HINT,
    EmbeddingRequest,
    EmbeddingResponse,
    PermanentProviderError,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    TransientProviderError,
    is_transient_status,
    split_system,
)

DEFAULT_TIMEOUT_S = 300.0
_TRANSIENT_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
    }
)


def _default_client(api_key: str, timeout_s: float) -> Any:
    try:
        module = importlib.import_module("anthropic")
    except ImportError as exc:
        raise PermanentProviderError(
            f"the 'anthropic' package is not installed; {PROVIDERS_EXTRA_HINT}"
        ) from exc
    return module.Anthropic(api_key=api_key, max_retries=0, timeout=timeout_s)


def classify_error(exc: Exception) -> ProviderError:
    """Map an SDK exception to a transient or permanent provider error."""
    status = getattr(exc, "status_code", None)
    detail = f"anthropic {type(exc).__name__}: {str(exc)[:500]}"
    if isinstance(status, int):
        detail = f"anthropic HTTP {status}: {str(exc)[:500]}"
        if is_transient_status(status):
            return TransientProviderError(detail)
        return PermanentProviderError(detail)
    if type(exc).__name__ in _TRANSIENT_ERROR_NAMES or isinstance(
        exc, TimeoutError | ConnectionError
    ):
        return TransientProviderError(detail)
    return PermanentProviderError(detail)


class AnthropicProvider:
    """Adapter for ``client.messages.create``; ``client`` may be injected for tests."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self.timeout_s = timeout_s

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise PermanentProviderError("ANTHROPIC_API_KEY is not set")
            self._client = _default_client(self._api_key, self.timeout_s)
        return self._client

    def message_kwargs(self, request: ProviderRequest) -> dict[str, Any]:
        """Keyword arguments for ``messages.create`` (exposed for tests and logging)."""
        system, messages = split_system(request.messages)
        kwargs: dict[str, Any] = {
            "model": request.model_id,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        # No seed parameter in the Messages API; the gateway records the omission.
        return kwargs

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        client = self._get_client()
        try:
            resp = client.messages.create(**self.message_kwargs(request))
        except ProviderError:
            raise
        except Exception as exc:
            raise classify_error(exc) from exc
        blocks = getattr(resp, "content", None) or []
        text = "".join(
            str(getattr(block, "text", ""))
            for block in blocks
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(resp, "usage", None)
        return ProviderResponse(
            text=text,
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            finish_reason=getattr(resp, "stop_reason", None),
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise PermanentProviderError("Anthropic has no embeddings API; use another embedder")


__all__ = ["AnthropicProvider", "classify_error"]
