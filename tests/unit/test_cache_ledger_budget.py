from datetime import date
from pathlib import Path

import pytest

from hgrag.config import BudgetConfig
from hgrag.llm.budget import BudgetExceededError, BudgetGuard
from hgrag.llm.cache import ResponseCache
from hgrag.llm.ledger import GroupBy, UsageLedger, UsageRecord


def _record(
    run_id: str, role: str, cost: float, hit: bool = False, ts: str | None = None
) -> UsageRecord:
    extra = {"ts": ts} if ts else {}
    return UsageRecord(
        run_id=run_id,
        experiment="e0",
        phase="p0",
        user="tester",
        role=role,
        provider="fake",
        model="fake-small",
        input_tokens=100,
        output_tokens=10,
        cached_input_tokens=0,
        cost_usd=cost,
        latency_ms=5.0,
        cache_hit=hit,
        status="ok",
        **extra,
    )


def test_cache_roundtrip_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    cache = ResponseCache(path)
    assert cache.get("k") is None
    cache.put("k", {"text": "hi", "n": 1})
    cache.close()
    assert ResponseCache(path).get("k") == {"text": "hi", "n": 1}


def test_ledger_totals_and_grouping(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "u.sqlite")
    ledger.record(_record("r1", "generator", 0.50))
    ledger.record(_record("r1", "judge", 0.25))
    ledger.record(_record("r2", "generator", 1.00, hit=True))

    assert ledger.total_cost() == pytest.approx(1.75)
    assert ledger.total_cost("r1") == pytest.approx(0.75)

    by_role = {row.group: row for row in ledger.summarize(GroupBy.ROLE)}
    assert by_role["generator"].calls == 2
    assert by_role["generator"].cache_hits == 1
    assert by_role["generator"].cost_usd == pytest.approx(1.50)
    assert [row.group for row in ledger.summarize(GroupBy.ROLE, run_id="r2")] == ["generator"]


def test_ledger_since_filter(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "u.sqlite")
    ledger.record(_record("old", "generator", 1.0, ts="2026-01-01T00:00:00+00:00"))
    ledger.record(_record("new", "generator", 2.0, ts="2026-12-01T00:00:00+00:00"))
    rows = ledger.summarize(GroupBy.RUN, since=date(2026, 6, 1))
    assert [row.group for row in rows] == ["new"]


def test_budget_guard_enforces_both_caps(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "u.sqlite")
    ledger.record(_record("earlier-run", "generator", 9.0))
    guard = BudgetGuard(
        BudgetConfig(global_cap_usd=10.0, per_run_cap_usd=0.5, confirm_above_usd=1.0),
        ledger,
        run_id="this-run",
    )
    guard.check(0.4)  # within both caps
    with pytest.raises(BudgetExceededError, match="per-run"):
        guard.check(0.6)
    guard.add(0.4)
    with pytest.raises(BudgetExceededError, match="global"):
        guard.check(0.7)  # 9.4 + 0.7 > 10


def test_cache_is_safe_under_concurrent_threads(tmp_path: Path) -> None:
    """Regression: retrieval runs in worker threads that share one cache connection."""
    from concurrent.futures import ThreadPoolExecutor

    cache = ResponseCache(tmp_path / "c.sqlite")

    def work(i: int) -> bool:
        cache.put(f"k{i}", {"i": i})
        return cache.get(f"k{i}") == {"i": i}

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert all(pool.map(work, range(400)))
    assert len(cache) == 400
