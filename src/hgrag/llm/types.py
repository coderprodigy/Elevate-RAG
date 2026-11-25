"""Provider-neutral request and response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hgrag.core.hashing import stable_hash

CallStatus = Literal["ok", "refusal", "error"]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """A fully resolved request for one provider call."""

    model: str
    messages: tuple[Message, ...]
    max_tokens: int
    system: str | None = None
    temperature: float | None = None  # None = provider default (required for Claude 5.5)
    params: dict[str, Any] = field(default_factory=dict)

    def cache_key(self, provider: str) -> str:
        """Content hash identifying this exact request to this provider."""
        return stable_hash(
            {
                "provider": provider,
                "model": self.model,
                "messages": [(m.role, m.content) for m in self.messages],
                "max_tokens": self.max_tokens,
                "system": self.system,
                "temperature": self.temperature,
                "params": self.params,
            }
        )

    def prompt_chars(self) -> int:
        return len(self.system or "") + sum(len(m.content) for m in self.messages)


@dataclass(frozen=True)
class ProviderResult:
    """Raw outcome of one provider call. ``input_tokens`` excludes cache-read tokens."""

    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    stop_reason: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """What callers get back from the gateway."""

    text: str
    model_key: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float  # 0.0 when served from the response cache
    latency_ms: float
    cache_hit: bool
    status: CallStatus
    stop_reason: str
