"""The single entry point for model calls: role → model, cache, budget, ledger.

Flow of :meth:`LLMGateway.acomplete`:
1. Resolve the model for the role (or the explicit ``model_key``) from ``configs/models.yaml``.
2. Build the request; drop ``temperature`` for models that reject it.
3. Cache hit → record a $0 ledger row and return.
4. Budget check with an upper-bound cost estimate → call provider → price → cache → ledger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from hgrag.config import AppConfig, ModelSpec, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.llm.budget import BudgetGuard
from hgrag.llm.cache import ResponseCache
from hgrag.llm.ledger import UsageLedger, UsageRecord
from hgrag.llm.pricing import cost_usd, estimate_max_cost_usd
from hgrag.llm.providers import Provider, build_provider
from hgrag.llm.types import CallStatus, LLMRequest, LLMResponse, Message, ProviderResult

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[ProviderSpec], Provider]
Prompt = str | Sequence[Message]


class LLMGateway:
    def __init__(
        self,
        config: AppConfig,
        *,
        cache: ResponseCache,
        ledger: UsageLedger,
        budget: BudgetGuard,
        context: RunContext,
        provider_factory: ProviderFactory = build_provider,
    ) -> None:
        self._config = config
        self._cache = cache
        self._ledger = ledger
        self._budget = budget
        self._context = context
        self._provider_factory = provider_factory
        self._providers: dict[str, Provider] = {}

    @property
    def context(self) -> RunContext:
        return self._context

    @property
    def budget(self) -> BudgetGuard:
        return self._budget

    def complete(self, role: str, prompt: Prompt, **kwargs: Any) -> LLMResponse:
        """Synchronous wrapper around :meth:`acomplete` (for scripts and the CLI)."""
        return asyncio.run(self.acomplete(role, prompt, **kwargs))

    async def acomplete(
        self,
        role: str,
        prompt: Prompt,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        model_key: str | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Run one call for ``role`` (e.g. "generator", "judge", "extractor")."""
        key, spec, request = self._prepare(role, prompt, system, max_tokens, model_key)
        cache_key = request.cache_key(spec.provider)

        if use_cache and (cached := self._cache.get(cache_key)) is not None:
            result = ProviderResult(**cached)
            response = self._to_response(key, spec, result, cost=0.0, latency_ms=0.0, hit=True)
            self._record(role, response)
            return response

        self._budget.check(
            estimate_max_cost_usd(spec.price, request.prompt_chars(), request.max_tokens)
        )
        provider = self._provider(spec.provider)
        started = time.perf_counter()
        try:
            result = await provider.complete(request)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000
            failed = ProviderResult(text="", input_tokens=0, output_tokens=0, stop_reason="error")
            self._record(role, self._to_response(key, spec, failed, 0.0, latency_ms, hit=False))
            raise
        latency_ms = (time.perf_counter() - started) * 1000

        cost = cost_usd(
            spec.price, result.input_tokens, result.output_tokens, result.cached_input_tokens
        )
        self._budget.add(cost)
        self._cache.put(cache_key, result.__dict__)
        response = self._to_response(key, spec, result, cost, latency_ms, hit=False)
        if response.status == "refusal":
            logger.warning(
                "model %s refused a %s request (run %s)", key, role, self._context.run_id
            )
        self._record(role, response)
        return response

    def estimate_cost(
        self,
        role: str,
        prompt: Prompt,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        model_key: str | None = None,
        use_cache: bool = True,
    ) -> float:
        """Upper-bound USD for this call without making it; 0.0 if it would be a cache hit."""
        _, spec, request = self._prepare(role, prompt, system, max_tokens, model_key)
        if use_cache and self._cache.get(request.cache_key(spec.provider)) is not None:
            return 0.0
        return estimate_max_cost_usd(spec.price, request.prompt_chars(), request.max_tokens)

    def model_key_for(self, role: str, model_key: str | None = None) -> str:
        """The model a call for ``role`` would use (recorded in run configs)."""
        return model_key or self._config.models.model_key_for_role(role)

    # -- helpers -----------------------------------------------------------------------------

    def _prepare(
        self,
        role: str,
        prompt: Prompt,
        system: str | None,
        max_tokens: int | None,
        model_key: str | None,
    ) -> tuple[str, ModelSpec, LLMRequest]:
        key = self.model_key_for(role, model_key)
        spec = self._model_spec(key)
        return key, spec, self._build_request(spec, prompt, system, max_tokens)

    def _model_spec(self, model_key: str) -> ModelSpec:
        try:
            return self._config.models.models[model_key]
        except KeyError:
            raise KeyError(f"unknown model '{model_key}' (see configs/models.yaml)") from None

    def _build_request(
        self, spec: ModelSpec, prompt: Prompt, system: str | None, max_tokens: int | None
    ) -> LLMRequest:
        messages = (Message("user", prompt),) if isinstance(prompt, str) else tuple(prompt)
        defaults = self._config.generation
        return LLMRequest(
            model=spec.model,
            messages=messages,
            max_tokens=max_tokens or defaults.max_tokens,
            system=system,
            temperature=defaults.temperature if spec.supports_temperature else None,
            params=dict(spec.params),
        )

    def _provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = self._provider_factory(self._config.models.providers[name])
        return self._providers[name]

    @staticmethod
    def _to_response(
        model_key: str,
        spec: ModelSpec,
        result: ProviderResult,
        cost: float,
        latency_ms: float,
        hit: bool,
    ) -> LLMResponse:
        status: CallStatus = (
            "error"
            if result.stop_reason == "error"
            else "refusal"
            if result.stop_reason == "refusal"
            else "ok"
        )
        return LLMResponse(
            text=result.text,
            model_key=model_key,
            provider=spec.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_input_tokens=result.cached_input_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            cache_hit=hit,
            status=status,
            stop_reason=result.stop_reason,
        )

    def _record(self, role: str, response: LLMResponse) -> None:
        ctx = self._context
        self._ledger.record(
            UsageRecord(
                run_id=ctx.run_id,
                experiment=ctx.experiment,
                phase=ctx.phase,
                user=ctx.user,
                role=role,
                provider=response.provider,
                model=response.model_key,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_input_tokens=response.cached_input_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                cache_hit=response.cache_hit,
                status=response.status,
            )
        )


def build_gateway(
    config: AppConfig,
    context: RunContext | None = None,
    *,
    provider_factory: ProviderFactory = build_provider,
) -> LLMGateway:
    """Wire up the gateway with the cache and ledger paths from ``config``."""
    context = context or RunContext()
    ledger = UsageLedger(config.paths.ledger_db)
    return LLMGateway(
        config,
        cache=ResponseCache(config.paths.cache_db),
        ledger=ledger,
        budget=BudgetGuard(config.budget, ledger, context.run_id),
        context=context,
        provider_factory=provider_factory,
    )
