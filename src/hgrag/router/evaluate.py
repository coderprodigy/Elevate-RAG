"""Offline router evaluation (RQ3): accuracy, cost, regret vs the oracle, tokens saved vs mix.

* Oracle: per question, the cheapest path that answered correctly (``correct == 1``); if none
  did, the cheapest path. Not deployable; it is the upper bound.
* Outcome labels (label scheme 2): the oracle's choice on the *training* split.
* R4 fallback: answer with the cheap path; escalate to the expensive one when the top retrieval
  score is below a threshold or the cheap answer is a refusal. The cost is both paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hgrag.eval.metrics.router import macro_f1, regret, routing_accuracy, tokens_saved
from hgrag.router.outcomes import OutcomeTable


@dataclass(frozen=True)
class RouterResult:
    router: str
    n: int
    accuracy: float
    tokens_per_query: float
    usd_per_1k: float
    regret: float  # oracle accuracy - router accuracy
    tokens_saved_vs_baseline: float
    routing_accuracy: float | None  # vs oracle choices
    routing_macro_f1: float | None
    path_share: dict[str, float]


def oracle_choices(table: OutcomeTable, cost_order: Sequence[str] | None = None) -> dict[str, str]:
    order = list(cost_order or table.cost_order())
    choices = {}
    for qid, paths in table.outcomes.items():
        correct = [p for p in order if paths[p].correct >= 1.0]
        choices[qid] = correct[0] if correct else order[0]
    return choices


def evaluate_choices(
    name: str,
    table: OutcomeTable,
    choices: dict[str, str],
    *,
    baseline_path: str,
    extra_tokens: dict[str, int] | None = None,
    extra_cost: dict[str, float] | None = None,
) -> RouterResult:
    """Score a routing decision per question. ``extra_*`` adds e.g. R3's classification call."""
    qids = table.qids
    n = max(len(qids), 1)
    extra_tokens = extra_tokens or {}
    extra_cost = extra_cost or {}
    chosen = [table.outcomes[q][choices[q]] for q in qids]
    accuracy = sum(o.correct for o in chosen) / n
    tokens = sum(o.tokens + extra_tokens.get(q, 0) for q, o in zip(qids, chosen, strict=True)) / n
    cost = sum(o.cost_usd + extra_cost.get(q, 0.0) for q, o in zip(qids, chosen, strict=True)) / n
    oracle = oracle_choices(table)
    oracle_accuracy = sum(table.outcomes[q][oracle[q]].correct for q in qids) / n
    baseline_tokens = sum(table.outcomes[q][baseline_path].tokens for q in qids) / n
    picked = [choices[q] for q in qids]
    gold = [oracle[q] for q in qids]
    return RouterResult(
        router=name,
        n=len(qids),
        accuracy=round(accuracy, 4),
        tokens_per_query=round(tokens, 1),
        usd_per_1k=round(1000 * cost, 4),
        regret=round(regret(oracle_accuracy, accuracy), 4),
        tokens_saved_vs_baseline=round(tokens_saved(tokens, baseline_tokens), 4),
        routing_accuracy=round(routing_accuracy(picked, gold), 4) if qids else None,
        routing_macro_f1=round(macro_f1(picked, gold), 4) if qids else None,
        path_share={p: round(picked.count(p) / n, 3) for p in sorted(set(picked))},
    )


def fallback_choices(
    table: OutcomeTable, cheap: str, escalate: str, threshold: float
) -> tuple[dict[str, str], dict[str, int], dict[str, float]]:
    """R4 decisions plus the cheap path's tokens/cost already spent on escalated questions."""
    choices: dict[str, str] = {}
    extra_tokens: dict[str, int] = {}
    extra_cost: dict[str, float] = {}
    for qid, paths in table.outcomes.items():
        first = paths[cheap]
        unsure = first.refused or first.top_score is None or first.top_score < threshold
        choices[qid] = escalate if unsure else cheap
        if unsure:
            extra_tokens[qid] = first.tokens
            extra_cost[qid] = first.cost_usd
    return choices, extra_tokens, extra_cost


def tune_fallback_threshold(
    table: OutcomeTable, cheap: str, escalate: str, thresholds: Sequence[float], baseline_path: str
) -> float:
    """Dev-set tuning: best accuracy, ties broken by fewer tokens."""

    def key(threshold: float) -> tuple[float, float]:
        choices, tok, cost = fallback_choices(table, cheap, escalate, threshold)
        result = evaluate_choices(
            "tune", table, choices, baseline_path=baseline_path, extra_tokens=tok, extra_cost=cost
        )
        return (result.accuracy, -result.tokens_per_query)

    return max(thresholds, key=key)
