"""Shared provider plumbing: usage ledger, spending cap, retries, pricing."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import random
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, TypeVar

from elevate_rag.config import ProviderConfig
from elevate_rag.types import CallRecord
from elevate_rag.utils import BudgetExceededError, ProviderError

log = logging.getLogger(__name__)
T = TypeVar("T")


class RetryableError(ProviderError):
    """Transient failure (rate limit, 5xx, timeout, connection reset)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


_CALL_COLLECTOR: contextvars.ContextVar[list[CallRecord] | None] = contextvars.ContextVar(
    "elevate_call_collector", default=None
)


@contextmanager
def collect_calls() -> Iterator[list[CallRecord]]:
    """Collect every CallRecord made in this context (per-question cost accounting).
    asyncio tasks inherit the context at creation, so concurrent questions do not mix."""
    records: list[CallRecord] = []
    token = _CALL_COLLECTOR.set(records)
    try:
        yield records
    finally:
        _CALL_COLLECTOR.reset(token)


class UsageLedger:
    """Thread-safe record of every model call, with an optional hard USD cap (FR-X5, NFR-3).

    The cap is checked *before* each paid call using a pessimistic estimate, so a run cannot
    overshoot by more than one call. Cached calls cost nothing and are flagged.
    """

    def __init__(self, max_usd: float | None = None) -> None:
        self.max_usd = max_usd
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()
        self._spent = 0.0

    def check_budget(self, estimated_usd: float | None, provider: ProviderConfig) -> None:
        if self.max_usd is None or provider.provider in FREE_PROVIDERS:
            return
        if estimated_usd is None:
            if self.max_usd == 0:
                raise BudgetExceededError(
                    f"{provider.provider}:{provider.model} is a paid/unknown-price provider but "
                    "the spending cap is 0. Set budget.max_usd (after approval) to allow paid "
                    "calls."
                )
            return
        with self._lock:
            if self._spent + estimated_usd > self.max_usd:
                raise BudgetExceededError(
                    f"spending cap ${self.max_usd:.2f} would be exceeded "
                    f"(spent ${self._spent:.4f}, next call ≈ ${estimated_usd:.4f})"
                )

    def add(self, record: CallRecord) -> None:
        collector = _CALL_COLLECTOR.get()
        if collector is not None:
            collector.append(record)
        with self._lock:
            self._records.append(record)
            if record.cost_usd and not record.cached:
                self._spent += record.cost_usd

    @property
    def spent_usd(self) -> float:
        return self._spent

    def records(self) -> list[CallRecord]:
        with self._lock:
            return list(self._records)

    def drain(self) -> list[CallRecord]:
        with self._lock:
            out, self._records = self._records, []
            return out

    def summary(self) -> dict[str, Any]:
        recs = self.records()
        by_role: dict[str, dict[str, float]] = {}
        for r in recs:
            agg = by_role.setdefault(
                r.role,
                {
                    "calls": 0,
                    "cached": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_unknown_calls": 0,
                    "errors": 0,
                },
            )
            agg["calls"] += 1
            agg["cached"] += int(r.cached)
            agg["prompt_tokens"] += r.prompt_tokens
            agg["completion_tokens"] += r.completion_tokens
            agg["errors"] += int(r.error is not None)
            if r.cost_usd is None:
                agg["cost_unknown_calls"] += int(not r.cached)
            elif not r.cached:
                agg["cost_usd"] += r.cost_usd
        return {"spent_usd": round(self._spent, 6), "by_role": by_role}

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records()]


# Providers that never cost money (local compute or deterministic offline algorithms).
FREE_PROVIDERS = frozenset(
    {"offline", "hash", "lexical", "none", "sentence-transformers", "cross-encoder"}
)


def price(cfg: ProviderConfig, prompt_tokens: int, completion_tokens: int) -> float | None:
    if cfg.provider in FREE_PROVIDERS:
        return 0.0
    if cfg.price_input_per_mtok is None:
        return None
    out_price = cfg.price_output_per_mtok or 0.0
    return (prompt_tokens * cfg.price_input_per_mtok + completion_tokens * out_price) / 1e6


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    what: str,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> T:
    """Bounded exponential backoff with full jitter; only `RetryableError` is retried."""
    attempt = 0
    while True:
        try:
            return await fn()
        except RetryableError as exc:
            attempt += 1
            if attempt > max_retries:
                raise ProviderError(
                    f"{what}: giving up after {max_retries} retries: {exc}"
                ) from exc
            delay = exc.retry_after or min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = delay * (0.5 + random.random() / 2)
            log.warning(
                "%s: transient error (%s); retry %d/%d in %.1fs",
                what,
                exc,
                attempt,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)


def http_error_to_exception(status: int, body: str, retry_after: str | None) -> ProviderError:
    if status in (408, 409, 425, 429) or status >= 500:
        ra = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else None
        return RetryableError(f"HTTP {status}: {body[:300]}", retry_after=ra)
    if status in (401, 403):
        return ProviderError(f"HTTP {status}: authentication failed — check the API key env var")
    return ProviderError(f"HTTP {status}: {body[:500]}")


class LoopLocalSemaphore:
    """An asyncio.Semaphore per event loop (providers outlive individual `asyncio.run` calls)."""

    def __init__(self, value: int) -> None:
        self._value = value
        self._sems: dict[int, asyncio.Semaphore] = {}

    def get(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._sems.get(id(loop))
        if sem is None:
            self._sems = {id(loop): (sem := asyncio.Semaphore(self._value))}
        return sem


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from synchronous code (CLI, sync providers)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    raise RuntimeError("run_sync called inside a running event loop; await instead")
