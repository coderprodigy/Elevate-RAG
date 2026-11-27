"""Oracle router and outcome labels (FR-R5, FR-R2 outcome scheme, FR-R6 metrics).

Strictly offline: it needs reference-based correctness of *every* mode, so it is computed from
finished per-mode result files and never imported by online code paths.

"Cheapest" = lowest measured mean (prompt + completion) tokens per mode on the same questions;
the static order `MODE_COST_ORDER` only breaks ties (D-08).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from elevate_rag.pipeline import MODE_COST_ORDER

# per_mode[mode][question_id] = {"correct": bool, "tokens": int, "cost_usd": float | None}
PerMode = Mapping[str, Mapping[str, Mapping[str, Any]]]


def mode_cost_ranking(per_mode: PerMode) -> list[str]:
    def mean_tokens(m: str) -> float:
        rows = list(per_mode[m].values())
        return sum(r.get("tokens", 0) for r in rows) / len(rows) if rows else 0.0

    return sorted(per_mode, key=lambda m: (mean_tokens(m), MODE_COST_ORDER.get(m, 99), m))


def oracle_labels(per_mode: PerMode) -> dict[str, str | None]:
    """question_id → cheapest correct mode, or None when no mode answered correctly."""
    ranking = mode_cost_ranking(per_mode)
    qids = sorted(set.intersection(*(set(v) for v in per_mode.values()))) if per_mode else []
    return {q: next((m for m in ranking if per_mode[m][q].get("correct")), None) for q in qids}


def outcome_training_labels(per_mode: PerMode) -> dict[str, str]:
    """Outcome-label scheme for R2: unanswerable-by-any questions get the cheapest mode (spending
    more cannot help them) — documented choice."""
    ranking = mode_cost_ranking(per_mode)
    return {q: (m or ranking[0]) for q, m in oracle_labels(per_mode).items()}


def routing_report(
    per_mode: PerMode, routed: Mapping[str, Mapping[str, Any]], *, reference_mode: str = "mix"
) -> dict[str, Any]:
    """Compare a router's per-question results with the oracle and static pipelines.

    `routed[qid]` = {"mode": str, "correct": bool, "tokens": int, "cost_usd": float|None}.
    """
    labels = oracle_labels(per_mode)
    qids = sorted(set(labels) & set(routed))
    if not qids:
        raise ValueError("no overlapping questions between router results and per-mode results")
    n = len(qids)
    oracle_acc = sum(labels[q] is not None for q in qids) / n
    router_acc = sum(bool(routed[q]["correct"]) for q in qids) / n
    static = {m: sum(bool(per_mode[m][q]["correct"]) for q in qids) / n for m in per_mode}
    best_static = max(static, key=lambda m: (static[m], -MODE_COST_ORDER.get(m, 99)))
    labelled = [q for q in qids if labels[q] is not None]
    route_acc = (
        sum(routed[q]["mode"] == labels[q] for q in labelled) / len(labelled) if labelled else None
    )
    ref_tokens = (
        sum(per_mode[reference_mode][q].get("tokens", 0) for q in qids)
        if reference_mode in per_mode
        else None
    )
    router_tokens = sum(routed[q].get("tokens", 0) for q in qids)
    ref_cost = _sum_cost(per_mode.get(reference_mode, {}), qids)
    router_cost = _sum_cost(routed, qids)
    return {
        "n": n,
        "router_accuracy": router_acc,
        "oracle_accuracy": oracle_acc,
        "regret": oracle_acc - router_acc,
        "best_static_mode": best_static,
        "best_static_accuracy": static[best_static],
        "gain_over_best_static": router_acc - static[best_static],
        "static_accuracy": static,
        "routing_accuracy_vs_oracle": route_acc,
        "token_savings_vs_reference": (1 - router_tokens / ref_tokens) if ref_tokens else None,
        "cost_savings_vs_reference": (1 - router_cost / ref_cost)
        if (ref_cost and router_cost is not None)
        else None,
        "reference_mode": reference_mode,
        "route_distribution": _count(routed[q]["mode"] for q in qids),
    }


def _sum_cost(rows: Mapping[str, Mapping[str, Any]], qids: list[str]) -> float | None:
    vals = [rows[q].get("cost_usd") for q in qids if q in rows]
    return None if not vals or any(v is None for v in vals) else float(sum(vals))  # type: ignore[arg-type]


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
