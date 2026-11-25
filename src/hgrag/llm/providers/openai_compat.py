"""OpenAI and any OpenAI-compatible server (vLLM on the Waterloo server, OpenRouter)."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from hgrag.config import ProviderSpec
from hgrag.llm.providers.base import ProviderConfigError, read_env, require_env
from hgrag.llm.types import LLMRequest, ProviderResult


class OpenAICompatProvider:
    def __init__(self, spec: ProviderSpec) -> None:
        base_url = spec.base_url or read_env(spec.base_url_env)
        if spec.base_url_env and base_url is None:
            raise ProviderConfigError(
                f"set {spec.base_url_env} in .env (is the SSH tunnel / server running?)"
            )
        self._client = AsyncOpenAI(
            api_key=require_env(spec.api_key_env, purpose="OpenAI-compatible provider"),
            base_url=base_url,
            max_retries=spec.max_retries,
            timeout=spec.timeout_s,
        )
        self._max_tokens_field = spec.max_tokens_field

    async def complete(self, request: LLMRequest) -> ProviderResult:
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            self._max_tokens_field: request.max_tokens,
            **request.params,  # e.g. extra_body={"provider": {...}} for OpenRouter routing
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        details = usage.prompt_tokens_details if usage else None
        cached = (details.cached_tokens or 0) if details else 0
        return ProviderResult(
            text=choice.message.content or "",
            input_tokens=prompt_tokens - cached,  # prompt_tokens includes cached tokens
            output_tokens=usage.completion_tokens if usage else 0,
            cached_input_tokens=cached,
            stop_reason=choice.finish_reason or "",
        )
