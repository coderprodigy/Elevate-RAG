"""Deterministic offline provider for unit tests and dry runs. Costs nothing real."""

from __future__ import annotations

from hgrag.llm.types import LLMRequest, ProviderResult

_CHARS_PER_TOKEN = 4


class FakeProvider:
    """Echoes the last user message. Records requests so tests can inspect them."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> ProviderResult:
        self.requests.append(request)
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        text = f"FAKE[{request.model}]: {last_user[:80]}"
        return ProviderResult(
            text=text,
            input_tokens=max(1, request.prompt_chars() // _CHARS_PER_TOKEN),
            output_tokens=max(1, len(text) // _CHARS_PER_TOKEN),
            stop_reason="end_turn",
        )
