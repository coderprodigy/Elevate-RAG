"""Markdown tables for the report (each figure has one of these as its table view)."""

from __future__ import annotations

from collections.abc import Sequence

from hgrag.analysis.collect import RunRecord
from hgrag.analysis.compare import Comparison


def _ci(run: RunRecord, group: str = "overall") -> str:
    metrics = run.metrics.overall if group == "overall" else run.metrics.by_query_type.get(group)
    if metrics is None or "correct" not in metrics.metrics:
        return "-"
    m = metrics.metrics["correct"]
    return f"{m.mean:.3f} [{m.low:.3f}, {m.high:.3f}]"


def summary_table(runs: Sequence[RunRecord]) -> str:
    lines = [
        "| Dataset | System | n | Correct [95% CI] | Primary | Tokens/q | $ / 1k q "
        "| p95 latency ms |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda r: (r.dataset, r.pipeline)):
        o = r.metrics.overall
        lines.append(
            f"| {r.dataset} | {r.pipeline} | {o.n} | {_ci(r)} | {r.metrics.primary_metric} | "
            f"{o.tokens_per_query:,.0f} | {o.usd_per_1k_queries:.3f} | {o.latency_p95_ms:,.0f} |"
        )
    return "\n".join(lines) + "\n"


def by_type_table(runs: Sequence[RunRecord]) -> str:
    types = sorted({t for r in runs for t in r.metrics.by_query_type})
    lines = [
        "| Dataset | System | " + " | ".join(types) + " |",
        "|---|---|" + "---|" * len(types),
    ]
    for r in sorted(runs, key=lambda r: (r.dataset, r.pipeline)):
        cells = [
            f"{r.metrics.by_query_type[t].metrics['correct'].mean:.3f}"
            if t in r.metrics.by_query_type
            else "-"
            for t in types
        ]
        lines.append(f"| {r.dataset} | {r.pipeline} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def comparison_table(rows: Sequence[Comparison]) -> str:
    lines = [
        "| Dataset | System | vs | Query type | n | Δ accuracy [95% CI] | p (Holm) | MDE "
        "| Verdict |",
        "|---|---|---|---|---:|---|---:|---:|---|",
    ]
    for c in rows:
        lines.append(
            f"| {c.dataset} | {c.system} | {c.reference} | {c.query_type} | {c.n} | "
            f"{c.diff:+.3f} [{c.diff_low:+.3f}, {c.diff_high:+.3f}] | {c.p_holm:.4f} | "
            f"{c.mde:.3f} | {c.verdict} |"
        )
    return "\n".join(lines) + "\n"
