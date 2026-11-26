"""Unit tests for the Budget_Tracker (Req 4.3, 4.4, 4.5)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from elevate.config.schema import CapSettings
from elevate.core.errors import BudgetExceeded
from elevate.gateway.budget import (
    LEDGER_FILENAME,
    SQLiteBudgetTracker,
    default_ledger_path,
    sample_size,
)
from elevate.gateway.cache import CACHE_DIR_ENV
from elevate.gateway.registry import ModelRegistry

REGISTRY_FILE = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


def caps(run: float = 5.0, project: float = 200.0) -> CapSettings:
    return CapSettings(run_spending_cap_usd=run, project_spending_cap_usd=project)


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "budget.sqlite3"


def test_accumulates_per_run_and_per_project(ledger: Path) -> None:
    run_a = SQLiteBudgetTracker("run-a", caps(), ledger_path=ledger)
    run_b = SQLiteBudgetTracker("run-b", caps(), ledger_path=ledger)
    run_a.charge(1.25, tokens=100)
    run_a.charge(0.75, tokens=50)
    run_b.charge(0.5, tokens=10)
    assert run_a.run_spent_usd == pytest.approx(2.0)
    assert run_a.run_tokens == 150
    assert run_b.run_spent_usd == pytest.approx(0.5)
    assert run_a.project_spent_usd == pytest.approx(2.5)
    assert run_b.project_remaining_usd == pytest.approx(197.5)


def test_project_spend_persists_across_tracker_instances(ledger: Path) -> None:
    SQLiteBudgetTracker("run-1", caps(), ledger_path=ledger).charge(3.0)
    later = SQLiteBudgetTracker("run-2", caps(), ledger_path=ledger)
    assert later.project_spent_usd == pytest.approx(3.0)
    assert later.run_spent_usd == 0.0


def test_run_cap_raises_with_spent_cap_and_remaining(ledger: Path) -> None:
    tracker = SQLiteBudgetTracker("r", caps(run=1.0), ledger_path=ledger, remaining_items=7)
    tracker.charge(0.6)
    with pytest.raises(BudgetExceeded) as info:
        tracker.charge(0.5)
    err = info.value
    assert err.spent_usd == pytest.approx(1.1)
    assert err.cap_usd == 1.0
    assert err.remaining_items == 7
    assert "7 item(s) remaining" in str(err)
    # The over-cap charge was still recorded: the money is spent.
    assert tracker.run_spent_usd == pytest.approx(1.1)
    # Further paid calls are blocked by the pre-call check.
    with pytest.raises(BudgetExceeded):
        tracker.check(remaining_items=3)
    assert pickle.loads(pickle.dumps(err)).spent_usd == err.spent_usd


def test_reaching_cap_exactly_stops(ledger: Path) -> None:
    tracker = SQLiteBudgetTracker("r", caps(run=1.0), ledger_path=ledger)
    with pytest.raises(BudgetExceeded):
        tracker.charge(1.0, remaining_items=0)


def test_project_cap_spans_runs(ledger: Path) -> None:
    SQLiteBudgetTracker("old", caps(project=2.0), ledger_path=ledger).charge(1.5)
    tracker = SQLiteBudgetTracker("new", caps(project=2.0), ledger_path=ledger)
    with pytest.raises(BudgetExceeded) as info:
        tracker.charge(0.5, remaining_items=4)
    assert info.value.cap_usd == 2.0
    assert info.value.spent_usd == pytest.approx(2.0)


def test_zero_cap_blocks_paid_calls_but_not_free_ones(ledger: Path) -> None:
    tracker = SQLiteBudgetTracker("r", caps(run=0.0, project=0.0), ledger_path=ledger)
    tracker.charge(0.0, tokens=500)  # offline fakes and local models cost nothing
    assert tracker.run_tokens == 500
    with pytest.raises(BudgetExceeded):
        tracker.check()
    with pytest.raises(BudgetExceeded):
        tracker.charge(0.01)


def test_charge_call_skips_cache_hits_and_prices_misses(ledger: Path) -> None:
    entry = ModelRegistry.load(REGISTRY_FILE).get("claude-sonnet-5.5")
    tracker = SQLiteBudgetTracker("r", caps(), ledger_path=ledger)
    hit_cost = tracker.charge_call(entry, 1_000_000, 0, cache_hit=True)
    assert hit_cost == 0.0
    assert tracker.run_spent_usd == 0.0
    assert tracker.run_tokens == 0
    miss_cost = tracker.charge_call(entry, 500_000, 0, cache_hit=False)
    assert miss_cost == pytest.approx(entry.cost_usd(500_000, 0))
    assert tracker.run_spent_usd == pytest.approx(miss_cost)
    assert tracker.run_tokens == 500_000


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_rejects_invalid_cost(ledger: Path, bad: float) -> None:
    tracker = SQLiteBudgetTracker("r", caps(), ledger_path=ledger)
    with pytest.raises(ValueError):
        tracker.charge(bad)


def test_estimate_extrapolates_five_percent_sample(ledger: Path) -> None:
    tracker = SQLiteBudgetTracker("sample", caps(run=5.0, project=10.0), ledger_path=ledger)
    tracker.charge(0.10, tokens=2_000)
    tracker.charge(0.05, tokens=1_000)
    est = tracker.estimate()
    assert est.sample_fraction == 0.05
    assert est.sample_tokens == 3_000
    assert est.sample_cost_usd == pytest.approx(0.15)
    assert est.estimated_tokens == 60_000
    assert est.estimated_cost_usd == pytest.approx(3.0)
    assert est.project_remaining_usd == pytest.approx(9.85)
    assert est.within_caps
    assert "60000 tokens" in est.summary()
    assert not tracker.estimate(0.01).within_caps  # $15 > $5 run cap


@pytest.mark.parametrize("fraction", [0.0, -0.05, 1.5])
def test_estimate_rejects_bad_fraction(ledger: Path, fraction: float) -> None:
    with pytest.raises(ValueError):
        SQLiteBudgetTracker("r", caps(), ledger_path=ledger).estimate(fraction)


def test_sample_size() -> None:
    assert sample_size(0) == 0
    assert sample_size(1) == 1
    assert sample_size(19) == 1
    assert sample_size(100) == 5
    assert sample_size(101) == 6
    assert sample_size(10, 1.0) == 10


def test_default_ledger_path_uses_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    assert default_ledger_path() == tmp_path / LEDGER_FILENAME
