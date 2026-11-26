"""OpenAI-compatible chat-completions and embeddings adapter.

Talks plain HTTP (stdlib ``urllib``) to any server implementing the OpenAI REST API: OpenAI
itself, vLLM and Ollama (``/v1``). No SDK is needed, so the adapter works without the
``providers`` extra. The HTTP call is an injectable :data:`Transport`, so tests run offline.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse

from elevate.gateway.providers.base import (
    EmbeddingRequest,
    EmbeddingResponse,
    PermanentProviderError,
    ProviderRequest,
    ProviderResponse,
    TransientProviderError,
    is_transient_status,
)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_S = 120.0

Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], tuple[int, Any]]
"""``(url, headers, json_payload, timeout_s) -> (http_status, decoded_json_or_text)``.

Raises :class:`TransientProviderError` for network-level failures."""


def urllib_transport(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> tuple[int, Any]:
    """POST ``payload`` as JSON with ``urllib``; return the status and decoded body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _decode(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TransientProviderError(f"connection to {url} failed: {reason}") from exc


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _error_detail(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and "message" in err:
            return str(err["message"])[:500]
        if isinstance(err, str):
            return err[:500]
    return str(body)[:500]


class OpenAICompatibleProvider:
    """Adapter for ``POST {base_url}/chat/completions`` and ``/embeddings``.

    Args:
        name: provider label used in errors (``openai``, ``vllm``, ``ollama``).
        base_url: API root including the version segment, e.g. ``http://host:8000/v1``.
        api_key: bearer token; ``None`` sends no ``Authorization`` header (local servers).
        max_tokens_field: ``max_completion_tokens`` (OpenAI) or ``max_tokens`` (vLLM/Ollama).
        transport: HTTP function; defaults to :func:`urllib_transport`.
    """

    def __init__(
        self,
        *,
        name: str = "openai",
        base_url: str = OPENAI_DEFAULT_BASE_URL,
        api_key: str | None = None,
        max_tokens_field: str = "max_completion_tokens",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: Transport | None = None,
    ) -> None:
        scheme = urlparse(base_url).scheme
        if scheme not in {"http", "https"}:
            raise PermanentProviderError(
                f"{name} base URL must start with http:// or https://, got {base_url!r}"
            )
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.max_tokens_field = max_tokens_field
        self.timeout_s = timeout_s
        self._transport = transport or urllib_transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        status, body = self._transport(url, self._headers(), payload, self.timeout_s)
        if status >= 400:
            detail = f"{self.name} HTTP {status}: {_error_detail(body)}"
            if is_transient_status(status):
                raise TransientProviderError(detail)
            raise PermanentProviderError(detail)
        if not isinstance(body, dict):
            raise TransientProviderError(f"{self.name} returned a non-JSON body: {str(body)[:200]}")
        return body

    def chat_payload(self, request: ProviderRequest) -> dict[str, Any]:
        """The JSON body for one chat-completions call (exposed for tests and logging)."""
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [m.to_dict() for m in request.messages],
            self.max_tokens_field: request.max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        body = self._post("/chat/completions", self.chat_payload(request))
        try:
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise TransientProviderError(
                f"{self.name} response has no choices[0].message: {str(body)[:200]}"
            ) from exc
        usage = body.get("usage") or {}
        return ProviderResponse(
            text=str(text),
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            finish_reason=choice.get("finish_reason"),
        )

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        body = self._post("/embeddings", {"model": request.model_id, "input": list(request.texts)})
        try:
            rows = sorted(body["data"], key=lambda row: int(row.get("index", 0)))
            vectors = [[float(x) for x in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise TransientProviderError(
                f"{self.name} embeddings response is malformed: {str(body)[:200]}"
            ) from exc
        if len(vectors) != len(request.texts):
            raise TransientProviderError(
                f"{self.name} returned {len(vectors)} embeddings for {len(request.texts)} texts"
            )
        usage = body.get("usage") or {}
        tokens = _int_or_none(usage.get("prompt_tokens"))
        return EmbeddingResponse(vectors=vectors, prompt_tokens=tokens)


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "OPENAI_DEFAULT_BASE_URL",
    "OpenAICompatibleProvider",
    "Transport",
    "urllib_transport",
]
