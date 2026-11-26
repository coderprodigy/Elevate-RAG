"""Budget_Tracker: per-Run and per-project spend with a hard Spending_Cap (Req 4.3-4.5).

Every uncached model call is charged once (callers skip charges when ``CacheResult.hit`` is
true). Charges are appended to a SQLite ledger, by default ``$ELEVATE_CACHE_DIR/budget.sqlite3``,
so project spend persists across Runs and processes. Run spend is the ledger sum for the
Run ID, so several workers of one Run share a cap.

Cap semantics: a charge that brings the Run or project total to its cap or above is
recorded (the money is already spent) and then raises :class:`BudgetExceeded` naming the
spent amount, the cap and the remaining work. The caller stops the Run and keeps completed
results so a rerun resumes from cache. A cap of 0 disables paid calls: call
:meth:`SQLiteBudgetTracker.check` before any call with a non-zero price. Free calls (cost 0,
e.g. the offline fakes and local models) never raise.

Cost estimation (Req 4.5): index a sample of the corpus (5% by default, see
:func:`sample_size`) with the tracker, then :meth:`SQLiteBudgetTracker.estimate` scales the
sample's tokens and cost to the full corpus.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from elevate.config.schema import CapSettings
from elevate.core.errors import BudgetExceeded
from elevate.gateway.cache import CACHE_DIR_ENV, DEFAULT_CACHE_DIR
from elevate.gateway.registry import ModelEntry

LEDGER_FILENAME = "budget.sqlite3"
DEFAULT_SAMPLE_FRACTION = 0.05
_BUSY_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class CostEstimate:
    """Full-corpus tokens and cost extrapolated from an indexed sample (Req 4.5)."""

    sample_fraction: float
    sample_tokens: int
    sample_cost_usd: float
    estimated_tokens: int
    estimated_cost_usd: float
    run_cap_usd: float
    project_remaining_usd: float

    @property
    def within_caps(self) -> bool:
        """True when the estimate fits under both the Run cap and the project remainder."""
        return self.estimated_cost_usd < min(self.run_cap_usd, self.project_remaining_usd)

    def summary(self) -> str:
        return (
            f"Estimated full run from {self.sample_fraction:.1%} sample: "
            f"{self.estimated_tokens} tokens, ${self.estimated_cost_usd:.4f} "
            f"(sample: {self.sample_tokens} tokens, ${self.sample_cost_usd:.4f}; "
            f"run cap ${self.run_cap_usd:.2f}, project remaining "
            f"${self.project_remaining_usd:.2f})"
        )


class BudgetTracker(Protocol):
    """Budget interface used by the LLM_Gateway (design: Gateway section)."""

    def charge(
        self, cost_usd: float, *, tokens: int = 0, remaining_items: int | None = None
    ) -> None: ...

    def estimate(self, sample_fraction: float = DEFAULT_SAMPLE_FRACTION) -> CostEstimate: ...


def default_ledger_path() -> Path:
    """Ledger file under ``$ELEVATE_CACHE_DIR``, or under ``.cache/`` when unset."""
    configured = os.environ.get(CACHE_DIR_ENV)
    directory = Path(configured) if configured else DEFAULT_CACHE_DIR
    return directory / LEDGER_FILENAME


def sample_size(total_items: int, fraction: float = DEFAULT_SAMPLE_FRACTION) -> int:
    """Number of items in the cost-estimation sample: ``ceil(total * fraction)``, at least 1."""
    _check_fraction(fraction)
    if total_items < 0:
        raise ValueError(f"total_items must be >= 0, got {total_items}")
    if total_items == 0:
        return 0
    return min(total_items, max(1, math.ceil(total_items * fraction)))


def _check_fraction(fraction: float) -> None:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {fraction}")


class SQLiteBudgetTracker:
    """:class:`BudgetTracker` backed by a SQLite ledger shared by all Runs of a project."""

    def __init__(
        self,
        run_id: str,
        caps: CapSettings | None = None,
        *,
        ledger_path: str | Path | None = None,
        remaining_items: int = 0,
    ) -> None:
        self.run_id = run_id
        self.caps = caps or CapSettings()
        self.remaining_items = remaining_items
        self.path = Path(ledger_path) if ledger_path is not None else default_ledger_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS charges ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " run_id TEXT NOT NULL,"
                " cost_usd REAL NOT NULL,"
                " tokens INTEGER NOT NULL,"
                " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS charges_run ON charges (run_id)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_S)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ totals
    def _totals(self, conn: sqlite3.Connection) -> tuple[float, int, float]:
        run_cost, run_tokens = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0), COALESCE(SUM(tokens), 0)"
            " FROM charges WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        (project_cost,) = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM charges").fetchone()
        return float(run_cost), int(run_tokens), float(project_cost)

    @property
    def run_spent_usd(self) -> float:
        with self._connect() as conn:
            return self._totals(conn)[0]

    @property
    def run_tokens(self) -> int:
        with self._connect() as conn:
            return self._totals(conn)[1]

    @property
    def project_spent_usd(self) -> float:
        with self._connect() as conn:
            return self._totals(conn)[2]

    @property
    def project_remaining_usd(self) -> float:
        return max(0.0, self.caps.project_spending_cap_usd - self.project_spent_usd)

    # ------------------------------------------------------------------ charging
    def _exceeded(self, run_cost: float, project_cost: float, remaining: int) -> None:
        if run_cost >= self.caps.run_spending_cap_usd:
            raise BudgetExceeded(
                spent_usd=run_cost,
                cap_usd=self.caps.run_spending_cap_usd,
                remaining_items=remaining,
            )
        if project_cost >= self.caps.project_spending_cap_usd:
            raise BudgetExceeded(
                spent_usd=project_cost,
                cap_usd=self.caps.project_spending_cap_usd,
                remaining_items=remaining,
            )

    def check(self, *, remaining_items: int | None = None) -> None:
        """Raise :class:`BudgetExceeded` if either cap is already reached.

        Call before a paid call; with a cap of 0 this blocks every paid call.
        """
        remaining = self.remaining_items if remaining_items is None else remaining_items
        with self._connect() as conn:
            run_cost, _, project_cost = self._totals(conn)
        self._exceeded(run_cost, project_cost, remaining)

    def charge(
        self, cost_usd: float, *, tokens: int = 0, remaining_items: int | None = None
    ) -> None:
        """Record one uncached call; raise :class:`BudgetExceeded` once a cap is reached.

        The charge is persisted before raising. Zero-cost calls record their tokens (for
        :meth:`estimate`) and never raise.
        """
        if cost_usd < 0 or not math.isfinite(cost_usd):
            raise ValueError(f"cost_usd must be a finite non-negative number, got {cost_usd}")
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        remaining = self.remaining_items if remaining_items is None else remaining_items
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO charges (run_id, cost_usd, tokens) VALUES (?, ?, ?)",
                (self.run_id, float(cost_usd), int(tokens)),
            )
            run_cost, _, project_cost = self._totals(conn)
        if cost_usd > 0:
            self._exceeded(run_cost, project_cost, remaining)

    def charge_call(
        self,
        entry: ModelEntry,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cache_hit: bool,
        remaining_items: int | None = None,
    ) -> float:
        """Charge an uncached call at the entry's frozen prices; return the cost charged.

        Cache hits cost nothing and are not recorded (Req 4.3).
        """
        if cache_hit:
            return 0.0
        cost = entry.cost_usd(prompt_tokens, completion_tokens)
        self.charge(cost, tokens=prompt_tokens + completion_tokens, remaining_items=remaining_items)
        return cost

    # ------------------------------------------------------------------ estimation
    def estimate(self, sample_fraction: float = DEFAULT_SAMPLE_FRACTION) -> CostEstimate:
        """Extrapolate this Run's charges so far, taken as a sample, to the full corpus."""
        _check_fraction(sample_fraction)
        with self._connect() as conn:
            run_cost, run_tokens, project_cost = self._totals(conn)
        return CostEstimate(
            sample_fraction=sample_fraction,
            sample_tokens=run_tokens,
            sample_cost_usd=run_cost,
            estimated_tokens=round(run_tokens / sample_fraction),
            estimated_cost_usd=run_cost / sample_fraction,
            run_cap_usd=self.caps.run_spending_cap_usd,
            project_remaining_usd=max(0.0, self.caps.project_spending_cap_usd - project_cost),
        )


__all__ = [
    "DEFAULT_SAMPLE_FRACTION",
    "LEDGER_FILENAME",
    "BudgetTracker",
    "CostEstimate",
    "SQLiteBudgetTracker",
    "default_ledger_path",
    "sample_size",
]
