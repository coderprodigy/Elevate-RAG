"""Budget guard: refuses a call that could push spend over the global or per-run cap."""

from __future__ import annotations

from hgrag.config import BudgetConfig
from hgrag.llm.ledger import UsageLedger


class BudgetExceededError(RuntimeError):
    """Raised before a call that could exceed a cap. Runs catch it, checkpoint and stop."""


class BudgetGuard:
    """Tracks spend in memory (seeded from the ledger), so checks don't hit the database per call.

    Spend by other processes running at the same time is not seen until the next start.
    """

    def __init__(self, config: BudgetConfig, ledger: UsageLedger, run_id: str) -> None:
        self._config = config
        self._global_spent = ledger.total_cost()
        self._run_spent = ledger.total_cost(run_id)

    @property
    def global_spent_usd(self) -> float:
        return self._global_spent

    @property
    def run_spent_usd(self) -> float:
        return self._run_spent

    def check(self, estimated_cost_usd: float) -> None:
        """Raise :class:`BudgetExceededError` if ``estimated_cost_usd`` would break a cap."""
        if self._global_spent + estimated_cost_usd > self._config.global_cap_usd:
            raise BudgetExceededError(
                f"global cap ${self._config.global_cap_usd:.2f} would be exceeded "
                f"(spent ${self._global_spent:.4f}, next call up to ${estimated_cost_usd:.4f})"
            )
        if self._run_spent + estimated_cost_usd > self._config.per_run_cap_usd:
            raise BudgetExceededError(
                f"per-run cap ${self._config.per_run_cap_usd:.2f} would be exceeded "
                f"(run spent ${self._run_spent:.4f}, next call up to ${estimated_cost_usd:.4f})"
            )

    def add(self, cost_usd: float) -> None:
        self._global_spent += cost_usd
        self._run_spent += cost_usd
