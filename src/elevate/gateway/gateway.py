"""LLM_Gateway: the single path for every LLM and embedding call (Requirements 3 and 4).

For each call the gateway

1. resolves the Model_Registry entry of the requested role (Req 3.2). Each role has its own
   entry and settings, so the extraction and generator LLMs can differ in one Run (Req 3.7);
2. builds the request with temperature 0 (Req 3.3) and the Run seed when the model supports
   one (Req 3.9). Entries with ``supports_temperature: false`` (Claude Sonnet 5.5 rejects
   sampling parameters) get no temperature; the deviation is logged and recorded with the
   result and trace;
3. looks the request up in the Cache by content hash, model ID and parameters (Req 4.1);
4. on a miss, checks the budget before a paid call and calls the provider, retrying
   transient failures with exponential backoff (``backoff_base_s * 2**(attempt-1)``, by
   default 2, 4, 8, 16 s over 5 attempts) before raising :class:`ModelCallError` naming the
   model and the last failure (Req 3.5);
5. for JSON requests, parses the reply; on failure it runs one local repair pass (strip code
   fences, cut to the outermost object/array, drop trailing commas), then one re-request,
   and records the output as malformed if that still fails (Req 3.6). Malformed results
   are returned, not raised, so the caller can count them and continue;
6. charges the Budget_Tracker for uncached calls only (Req 4.3), after the result is cached,
   so a :class:`BudgetExceeded` stop keeps the completed result for resumption (Req 4.4);
7. logs model ID, call date, tokens, latency and cost (Req 3.4) and emits a Tracer record.

Sleeping, timing and the clock are injectable so tests never wait.
"""

from __future__ import annotations

import json as jsonlib
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from elevate.config.schema import ExperimentConfig, ModelRoles, RetrySettings
from elevate.core.errors import BudgetExceeded, ModelCallError
from elevate.core.tokens import TokenCounter, get_token_counter
from elevate.gateway.cache import (
    CACHE_FILENAME,
    Cache,
    CacheKey,
    CacheNamespace,
    SQLiteCache,
)
from elevate.gateway.providers.anthropic import AnthropicProvider
from elevate.gateway.providers.base import (
    EmbeddingRequest,
    Message,
    PermanentProviderError,
    Prompt,
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    TransientProviderError,
)
from elevate.gateway.providers.fake import FakeProvider
from elevate.gateway.providers.local import ollama_provider, vllm_provider
from elevate.gateway.providers.openai_compat import (
    OPENAI_DEFAULT_BASE_URL,
    OpenAICompatibleProvider,
)
from elevate.gateway.registry import ModelEntry, ModelRegistry, ModelRole
from elevate.gateway.tracer import LLMTrace, Tracer

logger = logging.getLogger(__name__)

T = TypeVar("T")
Vector = list[float]
ProviderFactory = Callable[[ModelEntry], Provider]

GENERATION_TEMPERATURE = 0.0
LLM_ROLES = frozenset({ModelRole.EXTRACTION, ModelRole.GENERATOR, ModelRole.JUDGE})
_NAMESPACES = {
    ModelRole.EXTRACTION: CacheNamespace.EXTRACTION,
    ModelRole.GENERATOR: CacheNamespace.GENERATION,
    ModelRole.JUDGE: CacheNamespace.JUDGE,
}
JSON_REREQUEST_INSTRUCTION = (
    "Your previous reply was not valid JSON. Reply again with only the valid JSON value, "
    "with no other text."
)
TEMPERATURE_DEVIATION = (
    "temperature/top_p/top_k omitted: the model rejects sampling parameters "
    "(deviation from Requirement 3.3)"
)
SEED_DEVIATION = "seed not sent: the provider has no seed parameter"


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class RoleSettings:
    """Per-role generation settings (Requirement 3.7). Temperature is always 0."""

    max_tokens: int = 1024


DEFAULT_ROLE_SETTINGS: Mapping[ModelRole, RoleSettings] = {
    ModelRole.EXTRACTION: RoleSettings(max_tokens=4096),
    ModelRole.GENERATOR: RoleSettings(max_tokens=1024),
    ModelRole.JUDGE: RoleSettings(max_tokens=512),
}


class GatewayBudget(Protocol):
    """The part of :class:`~elevate.gateway.budget.SQLiteBudgetTracker` the gateway uses."""

    def check(self, *, remaining_items: int | None = None) -> None: ...

    def charge_call(
        self,
        entry: ModelEntry,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cache_hit: bool,
        remaining_items: int | None = None,
    ) -> float: ...


# --------------------------------------------------------------------------- results
@dataclass(frozen=True)
class LLMResult:
    """Outcome of one :meth:`LLMGateway.complete` call.

    ``parsed`` holds the decoded JSON for JSON requests. ``json_status`` is ``ok``,
    ``repaired`` (local repair pass), ``rerequested`` (fixed by the re-request) or
    ``malformed``; ``None`` for text requests. ``cost_usd`` is what this call charged: 0 on
    a cache hit. ``called_at`` is when the provider was actually called (the original call
    for cache hits).
    """

    text: str
    role: str
    model_key: str
    model_id: str
    model_version: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool
    called_at: str
    attempts: int
    provider_calls: int
    parsed: Any = None
    json_status: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    deviations: tuple[str, ...] = ()

    @property
    def malformed(self) -> bool:
        return self.json_status == "malformed"

    @property
    def call_date(self) -> str:
        return self.called_at[:10]


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[Vector]
    model_key: str
    model_id: str
    model_version: str
    prompt_tokens: int
    cost_usd: float
    cached_count: int
    latency_s: float


# --------------------------------------------------------------------------- JSON helpers
class _NonFiniteJSONError(ValueError):
    pass


def _reject_constant(name: str) -> Any:
    raise _NonFiniteJSONError(f"non-finite JSON constant {name}")


def _loads(text: str) -> Any:
    return jsonlib.loads(text, parse_constant=_reject_constant)


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_MISSING = object()


def repair_json(text: str) -> Any:
    """One deterministic repair pass; returns the decoded value or ``_MISSING``."""
    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        if end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        for variant in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                return _loads(variant.strip())
            except ValueError:
                continue
    return _MISSING


def parse_json(text: str) -> tuple[Any, str | None]:
    """``(value, "ok"|"repaired")`` or ``(None, None)`` when even the repair pass fails."""
    try:
        return _loads(text.strip()), "ok"
    except ValueError:
        pass
    repaired = repair_json(text)
    if repaired is _MISSING:
        return None, None
    return repaired, "repaired"


def _const(value: T) -> Callable[[], T]:
    return lambda: value


# --------------------------------------------------------------------------- provider factory
def _env_lookup(
    config: ExperimentConfig | None, environ: Mapping[str, str]
) -> tuple[Callable[[str], str | None], Callable[[str], str | None]]:
    def setting(var: str) -> str | None:
        if config is not None:
            value = getattr(config.endpoints, var.lower(), None)
            if isinstance(value, str) and value:
                return value
        return environ.get(var) or None

    def secret(var: str) -> str | None:
        if config is not None:
            value = getattr(config.secrets, var.lower(), None)
            if value is not None:
                raw = value.get_secret_value()
                if raw:
                    return str(raw)
        return environ.get(var) or None

    return setting, secret


def default_provider_factory(
    config: ExperimentConfig | None = None, environ: Mapping[str, str] | None = None
) -> ProviderFactory:
    """Build provider adapters from registry entries, reading URLs and keys from the config's
    endpoints/secrets (filled from env or ``.env``) or the process environment."""
    setting, secret = _env_lookup(config, os.environ if environ is None else environ)

    def factory(entry: ModelEntry) -> Provider:
        provider = entry.provider
        if provider == "fake":
            return FakeProvider()
        if provider == "openai":
            key_var = entry.api_key_env or "OPENAI_API_KEY"
            api_key = secret(key_var)
            if not api_key:
                raise PermanentProviderError(
                    f"{key_var} is not set for '{entry.key}'; see docs/SETUP.md"
                )
            base = setting(entry.base_url_env or "OPENAI_BASE_URL") or OPENAI_DEFAULT_BASE_URL
            return OpenAICompatibleProvider(name="openai", base_url=base, api_key=api_key)
        if provider == "vllm":
            return vllm_provider(
                setting(entry.base_url_env or "VLLM_BASE_URL"),
                secret(entry.api_key_env or "VLLM_API_KEY"),
            )
        if provider == "ollama":
            return ollama_provider(setting(entry.base_url_env or "OLLAMA_BASE_URL"))
        if provider == "anthropic":
            return AnthropicProvider(api_key=secret(entry.api_key_env or "ANTHROPIC_API_KEY"))
        raise PermanentProviderError(
            f"no LLM_Gateway adapter for provider '{provider}' (entry '{entry.key}'); "
            f"pass one with providers={{'{provider}': ...}}"
        )

    return factory


# --------------------------------------------------------------------------- gateway
def _role_keys(roles: ModelRoles | Mapping[Any, str] | None) -> dict[ModelRole, str]:
    if roles is None:
        return {}
    if isinstance(roles, ModelRoles):
        return {role: getattr(roles, role.value).key for role in ModelRole}
    return {ModelRole(role): key for role, key in roles.items()}


class LLMGateway:
    """Routes every LLM and embedding call through the Model_Registry role entry.

    Args:
        registry: the Model_Registry.
        run_id: Run identifier recorded in traces.
        roles: role -> registry key (e.g. ``config.models``); unset roles use ``profile``.
        profile: registry profile for unset roles (default: the registry's default).
        providers: adapters by registry provider name; they take precedence over
            ``provider_factory`` (default :func:`default_provider_factory`).
        cache, budget, tracer: optional collaborators; ``None`` disables each.
        seed: Run seed passed to providers that support one (Req 3.9).
        retry: transient-error policy (Req 3.5).
        role_settings: per-role ``max_tokens`` (Req 3.7).
        sleep, clock, now: injectable time functions.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        run_id: str,
        roles: ModelRoles | Mapping[Any, str] | None = None,
        profile: str | None = None,
        providers: Mapping[str, Provider] | None = None,
        provider_factory: ProviderFactory | None = None,
        cache: Cache | None = None,
        budget: GatewayBudget | None = None,
        tracer: Tracer | None = None,
        seed: int | None = None,
        retry: RetrySettings | None = None,
        role_settings: Mapping[Any, RoleSettings] | None = None,
        embed_batch_size: int = 64,
        token_counter: TokenCounter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if embed_batch_size < 1:
            raise ValueError("embed_batch_size must be >= 1")
        self.registry = registry
        self.run_id = run_id
        self.profile = profile
        self.seed = seed
        self.retry = retry or RetrySettings()
        self.cache = cache
        self.budget = budget
        self.tracer = tracer
        self.embed_batch_size = embed_batch_size
        self._role_keys = _role_keys(roles)
        self._providers: dict[str, Provider] = dict(providers or {})
        self._factory = provider_factory or default_provider_factory()
        self._role_settings = {**DEFAULT_ROLE_SETTINGS}
        for role, settings in (role_settings or {}).items():
            self._role_settings[ModelRole(role)] = settings
        self._token_counter = token_counter
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._entries: dict[ModelRole, ModelEntry] = {}
        self._built: dict[str, Provider] = {}
        self._warned: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        run_id: str,
        registry: ModelRegistry | None = None,
        providers: Mapping[str, Provider] | None = None,
        cache: Cache | None = None,
        budget: GatewayBudget | None = None,
        tracer: Tracer | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> LLMGateway:
        """Gateway for one Run: config roles, seed and retry policy; SQLite cache and budget
        ledger under ``paths.cache_dir``; tracer from :func:`make_tracer`."""
        from elevate.gateway.budget import LEDGER_FILENAME, SQLiteBudgetTracker
        from elevate.gateway.tracer import make_tracer

        cache_dir = config.paths.cache_dir
        # ``is None`` checks: an empty SQLiteCache is falsy because it defines __len__.
        if cache is None:
            cache = SQLiteCache(cache_dir / CACHE_FILENAME)
        if budget is None:
            budget = SQLiteBudgetTracker(
                run_id, config.caps, ledger_path=cache_dir / LEDGER_FILENAME
            )
        if tracer is None:
            tracer = make_tracer(config, run_id)
        return cls(
            registry if registry is not None else ModelRegistry.load(),
            run_id=run_id,
            roles=config.models,
            providers=providers,
            provider_factory=default_provider_factory(config),
            cache=cache,
            budget=budget,
            tracer=tracer,
            seed=config.seed,
            retry=config.retry,
            sleep=sleep,
        )

    # ------------------------------------------------------------------ resolution
    def entry(self, role: ModelRole | str) -> ModelEntry:
        """Registry entry for ``role`` (explicit role key, else the profile assignment)."""
        role = ModelRole(role)
        with self._lock:
            cached = self._entries.get(role)
        if cached is not None:
            return cached
        found = self.registry.entry(role, key=self._role_keys.get(role), profile=self.profile)
        with self._lock:
            self._entries[role] = found
        return found

    def model_settings(self, roles: Sequence[ModelRole | str] | None = None) -> dict[str, Any]:
        """Effective per-role model settings for the Run_Record (Req 3.1, 3.7)."""
        out: dict[str, Any] = {}
        for role in roles or sorted(LLM_ROLES):
            role = ModelRole(role)
            entry = self.entry(role)
            settings = self._role_settings.get(role, RoleSettings())
            out[role.value] = {
                "key": entry.key,
                "provider": entry.provider,
                "model_id": entry.model_id,
                "version": entry.version,
                "temperature": GENERATION_TEMPERATURE if entry.supports_temperature else None,
                "seed": self.seed if entry.supports_seed else None,
                "max_tokens": settings.max_tokens,
            }
        return out

    def _provider(self, entry: ModelEntry) -> Provider:
        if entry.provider in self._providers:
            return self._providers[entry.provider]
        with self._lock:
            built = self._built.get(entry.key)
            if built is None:
                try:
                    built = self._factory(entry)
                except ProviderError as exc:
                    raise ModelCallError(model=entry.model_id, reason=str(exc), attempts=0) from exc
                self._built[entry.key] = built
            return built

    def _counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = get_token_counter()
        return self._token_counter

    def _deviations(self, entry: ModelEntry) -> tuple[str, ...]:
        found: list[str] = []
        if not entry.supports_temperature:
            found.append(TEMPERATURE_DEVIATION)
            with self._lock:
                first = entry.key not in self._warned
                self._warned.add(entry.key)
            if first:
                logger.warning("model '%s': %s", entry.model_id, TEMPERATURE_DEVIATION)
        if not entry.supports_seed and self.seed is not None:
            found.append(SEED_DEVIATION)
        return tuple(found)

    # ------------------------------------------------------------------ retry
    def _with_retry(self, entry: ModelEntry, call: Callable[[], T]) -> tuple[T, int]:
        """Run ``call``, retrying transient failures; return its value and the attempts used."""
        max_attempts = self.retry.max_attempts
        last: TransientProviderError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return call(), attempt
            except TransientProviderError as exc:
                last = exc
                if attempt < max_attempts:
                    delay = self.retry.backoff_base_s * 2 ** (attempt - 1)
                    logger.warning(
                        "model '%s' attempt %d/%d failed (%s); retrying in %.1fs",
                        entry.model_id,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    self._sleep(delay)
            except PermanentProviderError as exc:
                raise ModelCallError(
                    model=entry.model_id, reason=str(exc), attempts=attempt
                ) from exc
        assert last is not None
        raise ModelCallError(
            model=entry.model_id, reason=str(last) or type(last).__name__, attempts=max_attempts
        ) from last

    # ------------------------------------------------------------------ completion
    def _request(
        self,
        role: ModelRole,
        entry: ModelEntry,
        prompt: Prompt,
        json_mode: bool,
        max_tokens: int | None,
    ) -> ProviderRequest:
        settings = self._role_settings.get(role, RoleSettings())
        return ProviderRequest(
            role=role.value,
            model_id=entry.model_id,
            messages=tuple(prompt.messages),
            max_tokens=max_tokens or settings.max_tokens,
            temperature=GENERATION_TEMPERATURE if entry.supports_temperature else None,
            seed=self.seed if entry.supports_seed else None,
            json_mode=json_mode,
            metadata=dict(prompt.metadata),
        )

    def _tokens(self, request: ProviderRequest, response: ProviderResponse) -> tuple[int, int]:
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.completion_tokens
        if prompt_tokens is None:
            prompt_tokens = self._counter().count(request.text)
        if completion_tokens is None:
            completion_tokens = self._counter().count(response.text)
        return int(prompt_tokens), int(completion_tokens)

    def _call_llm(self, entry: ModelEntry, request: ProviderRequest) -> dict[str, Any]:
        """Uncached call: budget check, provider call(s) with retry, JSON handling."""
        if self.budget is not None and (entry.price_in_per_mtok or entry.price_out_per_mtok):
            self.budget.check()
        provider = self._provider(entry)
        called_at = self._now().isoformat()
        start = self._clock()
        response, attempts = self._with_retry(entry, lambda: provider.complete(request))
        prompt_tokens, completion_tokens = self._tokens(request, response)
        provider_calls = 1
        parsed: Any = None
        status: str | None = None
        if request.json_mode:
            parsed, status = parse_json(response.text)
            if status is None:
                retry_request = replace(
                    request,
                    messages=(
                        *request.messages,
                        Message("assistant", response.text),
                        Message("user", JSON_REREQUEST_INSTRUCTION),
                    ),
                )
                response, more = self._with_retry(entry, lambda: provider.complete(retry_request))
                attempts += more
                provider_calls += 1
                p_tok, c_tok = self._tokens(retry_request, response)
                prompt_tokens += p_tok
                completion_tokens += c_tok
                parsed, status = parse_json(response.text)
                status = "rerequested" if status is not None else "malformed"
                if status == "malformed":
                    logger.warning(
                        "model '%s' returned malformed JSON after repair and re-request",
                        entry.model_id,
                    )
        return {
            "text": response.text,
            "parsed": parsed,
            "json_status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "attempts": attempts,
            "provider_calls": provider_calls,
            "called_at": called_at,
            "provider_latency_s": self._clock() - start,
            "finish_reason": response.finish_reason,
        }

    def complete(
        self,
        role: ModelRole | str,
        prompt: Prompt | str,
        *,
        json: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """One LLM call for ``role`` (extraction, generator or judge).

        Raises :class:`ModelCallError` after retries, :class:`BudgetExceeded` at the cap
        (the result is cached first). Malformed JSON is returned with ``json_status``
        ``malformed``.
        """
        role = ModelRole(role)
        if role not in LLM_ROLES:
            raise ValueError(f"role '{role.value}' is not an LLM role; use embed() for embedders")
        if isinstance(prompt, str):
            prompt = Prompt.from_text(prompt)
        entry = self.entry(role)
        request = self._request(role, entry, prompt, json, max_tokens)
        deviations = self._deviations(entry)
        params = {
            "provider": entry.provider,
            "version": entry.version,
            "temperature": request.temperature,
            "seed": request.seed,
            "max_tokens": request.max_tokens,
            "json": json,
        }
        key = CacheKey.for_request(
            _NAMESPACES[role],
            {"messages": [m.to_dict() for m in prompt.messages], "metadata": dict(prompt.metadata)},
            entry.model_id,
            params,
        )
        start = self._clock()
        try:
            if self.cache is None:
                value, hit = self._call_llm(entry, request), False
            else:
                res = self.cache.get_or_call_result(key, lambda: self._call_llm(entry, request))
                value, hit = res.value, res.hit
        except ModelCallError as exc:
            self._emit(
                LLMTrace(
                    run_id=self.run_id,
                    role=role.value,
                    model_key=entry.key,
                    model_id=entry.model_id,
                    input=[m.to_dict() for m in prompt.messages],
                    output=None,
                    latency_s=self._clock() - start,
                    error=str(exc),
                    metadata={
                        "params": params,
                        "deviations": list(deviations),
                        "prompt_metadata": dict(prompt.metadata),
                    },
                )
            )
            raise
        latency = self._clock() - start
        prompt_tokens = int(value["prompt_tokens"])
        completion_tokens = int(value["completion_tokens"])
        cost, budget_error = self._charge(entry, prompt_tokens, completion_tokens, hit)
        result = LLMResult(
            text=str(value["text"]),
            role=role.value,
            model_key=entry.key,
            model_id=entry.model_id,
            model_version=entry.version,
            provider=entry.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency if hit else float(value["provider_latency_s"]),
            cost_usd=cost,
            cached=hit,
            called_at=str(value["called_at"]),
            attempts=int(value["attempts"]),
            provider_calls=0 if hit else int(value["provider_calls"]),
            parsed=value["parsed"],
            json_status=value["json_status"],
            params=params,
            deviations=deviations,
        )
        logger.info(
            "llm call role=%s model=%s date=%s prompt_tokens=%d completion_tokens=%d "
            "latency=%.3fs cost=$%.6f cached=%s",
            role.value,
            entry.model_id,
            result.call_date,
            prompt_tokens,
            completion_tokens,
            result.latency_s,
            cost,
            hit,
        )
        self._emit(
            LLMTrace(
                run_id=self.run_id,
                role=role.value,
                model_key=entry.key,
                model_id=entry.model_id,
                input=[m.to_dict() for m in prompt.messages],
                output=result.text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=result.latency_s,
                cost_usd=cost,
                cached=hit,
                metadata={
                    "call_date": result.call_date,
                    "called_at": result.called_at,
                    "model_version": entry.version,
                    "provider": entry.provider,
                    "params": params,
                    "json_status": result.json_status,
                    "attempts": result.attempts,
                    "provider_calls": result.provider_calls,
                    "deviations": list(deviations),
                    "prompt_metadata": dict(prompt.metadata),
                },
            )
        )
        if budget_error is not None:
            raise budget_error
        return result

    def _charge(
        self, entry: ModelEntry, prompt_tokens: int, completion_tokens: int, hit: bool
    ) -> tuple[float, BudgetExceeded | None]:
        if hit:
            return 0.0, None
        if self.budget is None:
            return entry.cost_usd(prompt_tokens, completion_tokens), None
        try:
            cost = self.budget.charge_call(entry, prompt_tokens, completion_tokens, cache_hit=False)
        except BudgetExceeded as exc:
            return entry.cost_usd(prompt_tokens, completion_tokens), exc
        return cost, None

    def _emit(self, trace: LLMTrace) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.trace(trace)
        except Exception as exc:  # tracing must never stop a Run
            logger.warning("trace failed: %s", type(exc).__name__)

    # ------------------------------------------------------------------ embeddings
    def embed(self, role: ModelRole | str, texts: Sequence[str]) -> list[Vector]:
        """Embedding vectors for ``texts`` from the ``role`` entry (normally ``embedder``)."""
        return self.embed_result(role, texts).vectors

    def embed_result(self, role: ModelRole | str, texts: Sequence[str]) -> EmbeddingResult:
        """Like :meth:`embed`, plus usage. Each text is cached separately; misses are sent
        to the provider in batches of ``embed_batch_size``."""
        role = ModelRole(role)
        entry = self.entry(role)
        params = {"provider": entry.provider, "version": entry.version}
        keys = [
            CacheKey.for_request(CacheNamespace.EMBEDDING, text, entry.model_id, params)
            for text in texts
        ]
        start = self._clock()
        found: dict[str, Vector] = {}
        if self.cache is not None:
            for key in keys:
                hit = self.cache.get(key)
                if hit is not None:
                    found[key.digest] = hit.value
        cached_count = sum(1 for key in keys if key.digest in found)
        missing: dict[str, tuple[str, CacheKey]] = {}
        for text, key in zip(texts, keys, strict=True):
            if key.digest not in found:
                missing.setdefault(key.digest, (text, key))
        pending = list(missing.values())
        total_tokens = 0
        total_cost = 0.0
        for i in range(0, len(pending), self.embed_batch_size):
            batch = pending[i : i + self.embed_batch_size]
            vectors, tokens, cost, budget_error = self._embed_batch(role, entry, batch)
            for (_text, key), vector in zip(batch, vectors, strict=True):
                stored = (
                    vector
                    if self.cache is None
                    else self.cache.get_or_call_result(key, _const(vector)).value
                )
                found[key.digest] = stored
            total_tokens += tokens
            total_cost += cost
            if budget_error is not None:
                raise budget_error
        if self.cache is not None:
            for key in keys:  # record hit statistics for cached texts
                if key.digest not in missing:
                    self.cache.get_or_call_result(key, _const(found[key.digest]))
        return EmbeddingResult(
            vectors=[found[key.digest] for key in keys],
            model_key=entry.key,
            model_id=entry.model_id,
            model_version=entry.version,
            prompt_tokens=total_tokens,
            cost_usd=total_cost,
            cached_count=cached_count,
            latency_s=self._clock() - start,
        )

    def _embed_batch(
        self, role: ModelRole, entry: ModelEntry, batch: Sequence[tuple[str, CacheKey]]
    ) -> tuple[list[Vector], int, float, BudgetExceeded | None]:
        if self.budget is not None and entry.price_in_per_mtok:
            self.budget.check()
        provider = self._provider(entry)
        request = EmbeddingRequest(
            role=role.value, model_id=entry.model_id, texts=tuple(t for t, _ in batch)
        )
        start = self._clock()
        try:
            response, attempts = self._with_retry(entry, lambda: provider.embed(request))
            if len(response.vectors) != len(batch):
                raise ModelCallError(
                    model=entry.model_id,
                    reason=f"returned {len(response.vectors)} vectors for {len(batch)} texts",
                    attempts=attempts,
                )
        except ModelCallError as exc:
            self._emit(
                LLMTrace(
                    run_id=self.run_id,
                    role=role.value,
                    model_key=entry.key,
                    model_id=entry.model_id,
                    input={"n_texts": len(batch)},
                    output=None,
                    latency_s=self._clock() - start,
                    error=str(exc),
                )
            )
            raise
        latency = self._clock() - start
        tokens = response.prompt_tokens
        if tokens is None:
            tokens = sum(self._counter().count(t) for t in request.texts)
        cost, budget_error = self._charge(entry, tokens, 0, False)
        dim = len(response.vectors[0]) if response.vectors else 0
        logger.info(
            "embedding call model=%s texts=%d tokens=%d latency=%.3fs cost=$%.6f",
            entry.model_id,
            len(batch),
            tokens,
            latency,
            cost,
        )
        self._emit(
            LLMTrace(
                run_id=self.run_id,
                role=role.value,
                model_key=entry.key,
                model_id=entry.model_id,
                input={"n_texts": len(batch)},
                output={"n_vectors": len(response.vectors), "dim": dim},
                prompt_tokens=tokens,
                latency_s=latency,
                cost_usd=cost,
                metadata={
                    "call_date": self._now().date().isoformat(),
                    "model_version": entry.version,
                    "provider": entry.provider,
                    "attempts": attempts,
                },
            )
        )
        return [list(v) for v in response.vectors], tokens, cost, budget_error


__all__ = [
    "DEFAULT_ROLE_SETTINGS",
    "GENERATION_TEMPERATURE",
    "JSON_REREQUEST_INSTRUCTION",
    "LLM_ROLES",
    "SEED_DEVIATION",
    "TEMPERATURE_DEVIATION",
    "EmbeddingResult",
    "GatewayBudget",
    "LLMGateway",
    "LLMResult",
    "ProviderFactory",
    "RoleSettings",
    "Vector",
    "default_provider_factory",
    "parse_json",
    "repair_json",
]
