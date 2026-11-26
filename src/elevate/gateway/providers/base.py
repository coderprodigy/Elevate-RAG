"""Provider adapter interface used by the LLM_Gateway.

A provider turns one fully specified request (model ID, messages, temperature, seed, ...)
into one response. Providers do not retry, cache, charge or trace: the gateway does all of
that. Providers classify failures instead, raising :class:`TransientProviderError` for
errors worth retrying (timeouts, connection errors, HTTP 408/409/425/429/5xx) and
:class:`PermanentProviderError` for everything else (bad request, auth, missing SDK).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]

TRANSIENT_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
PROVIDERS_EXTRA_HINT = "install the provider SDKs with `uv sync --extra providers`"


class ProviderError(Exception):
    """A provider call failed. ``transient`` tells the gateway whether to retry."""

    transient: bool = False


class TransientProviderError(ProviderError):
    """Timeout, connection failure, rate limit or server error: retry with backoff."""

    transient = True


class PermanentProviderError(ProviderError):
    """Bad request, authentication failure, missing SDK or key: do not retry."""

    transient = False


def is_transient_status(status: int) -> bool:
    return status in TRANSIENT_HTTP_STATUS or 500 <= status < 600


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Prompt:
    """Messages for one LLM call plus metadata.

    ``metadata`` carries request facts that are not message text, such as ``question_id``,
    ``variant``, ``context_doc_ids``, ``prediction`` or ``answers``. The offline fake uses
    them to pick scripted responses; real providers ignore them. Metadata is part of the
    cache key, so it must be deterministic and JSON-serialisable.
    """

    messages: tuple[Message, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        user: str,
        *,
        system: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Prompt:
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", user))
        return cls(messages=tuple(messages), metadata=dict(metadata or {}))

    @property
    def text(self) -> str:
        """All message contents joined by newlines (for matching and token counting)."""
        return "\n".join(m.content for m in self.messages)

    def with_message(self, message: Message) -> Prompt:
        return Prompt(messages=(*self.messages, message), metadata=self.metadata)


@dataclass(frozen=True)
class ProviderRequest:
    """One provider call. ``temperature``/``seed`` are ``None`` when the model rejects them."""

    role: str
    model_id: str
    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float | None = 0.0
    seed: int | None = None
    json_mode: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(m.content for m in self.messages)


@dataclass(frozen=True)
class ProviderResponse:
    """Text plus provider-reported usage; ``None`` usage means the gateway counts tokens."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    role: str
    model_id: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    prompt_tokens: int | None = None


class Provider(Protocol):
    """A model provider adapter (OpenAI-compatible, Anthropic, local, fake)."""

    name: str

    def complete(self, request: ProviderRequest) -> ProviderResponse: ...

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


def split_system(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, str]]]:
    """System text (joined) and the remaining messages, for APIs with a separate system field."""
    system = [m.content for m in messages if m.role == "system"]
    rest = [m.to_dict() for m in messages if m.role != "system"]
    return ("\n\n".join(system) if system else None), rest


__all__ = [
    "PROVIDERS_EXTRA_HINT",
    "TRANSIENT_HTTP_STATUS",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "Message",
    "PermanentProviderError",
    "Prompt",
    "Provider",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "Role",
    "TransientProviderError",
    "is_transient_status",
    "split_system",
]
