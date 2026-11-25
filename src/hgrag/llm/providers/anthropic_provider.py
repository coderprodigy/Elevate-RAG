"""Claude via the official Anthropic SDK."""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from hgrag.config import ProviderSpec
from hgrag.llm.providers.base import read_env
from hgrag.llm.types import LLMRequest, ProviderResult


class AnthropicProvider:
    def __init__(self, spec: ProviderSpec) -> None:
        client_kwargs: dict[str, Any] = {"max_retries": spec.max_retries, "timeout": spec.timeout_s}
        # If the key isn't in .env, the SDK falls back to its own credential lookup.
        api_key = read_env(spec.api_key_env)
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = AsyncAnthropic(**client_kwargs)

    async def complete(self, request: LLMRequest) -> ProviderResult:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            **request.params,  # e.g. thinking / output_config from configs/models.yaml
        }
        if request.system is not None:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        response = await self._client.messages.create(**kwargs)

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        # Cache writes bill near the input rate; count them as input (slightly conservative).
        cache_write = usage.cache_creation_input_tokens or 0
        return ProviderResult(
            text=text,
            input_tokens=usage.input_tokens + cache_write,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens or 0,
            stop_reason=response.stop_reason or "",
        )
