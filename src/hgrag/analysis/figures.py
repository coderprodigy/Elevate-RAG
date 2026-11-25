"""Report figures (matplotlib, static PNG + SVG). Each has a markdown table as its table view.

1. Accuracy by query type, one bar per system (the central figure, E1/E2).
2. Router accuracy vs cost, one point per router level (E3).
3. Vector store recall-vs-QPS curves (E4).
4. Failure causes per system (error analysis).
5. Factor importance: Δ accuracy vs the reference with 95% CI (E5).
6. Mode x generator interaction (E6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hgrag.analysis.collect import RunRecord
from hgrag.analysis.compare import Comparison
from hgrag.analysis.errors import CAUSES
from hgrag.analysis.style import CATEGORICAL, INK_SECONDARY, MUTED, SURFACE, apply_style, color_for

ROUTER_FAMILY = ["R0", "R1", "R2", "R3", "R4", "Oracle"]


def _save(fig: Any, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    outputs = [path.with_suffix(".png"), path.with_suffix(".svg")]
    for out in outputs:
        fig.savefig(out, dpi=200)
    plt.close(fig)
    return outputs


def accuracy_by_type(
    runs: Sequence[RunRecord], family: list[str], title: str, path: Path
) -> list[Path]:
    apply_style()
    systems = [p for p in family if any(r.pipeline == p for r in runs)]
    types = sorted({t for r in runs for t in r.metrics.by_query_type})
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(types) + 2), 3.6))
    width = 0.8 / max(len(systems), 1)
    for i, system in enumerate(systems):
        pooled: dict[str, list[float]] = {t: [] for t in types}
        for run in runs:
            if run.pipeline == system:
                for s in run.scores:
                    pooled[s.query_type].append(s.correct)
        xs, ys = [], []
        for j, t in enumerate(types):
            if pooled[t]:
                xs.append(j + (i - (len(systems) - 1) / 2) * width)
                ys.append(sum(pooled[t]) / len(pooled[t]))
        ax.bar(
            xs,
            ys,
            width=width,
            color=color_for(system, family),
            edgecolor=SURFACE,
            linewidth=1.0,
            label=system,
        )
    ax.set_xticks(range(len(types)), types)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Correct (primary metric)")
    ax.set_title(title, loc="left")
    ax.legend(ncols=min(len(systems), 6), loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def router_scatter(results: Sequence[Mapping[str, Any]], dataset: str, path: Path) -> list[Path]:
    """One point per router; only anchor points are labelled (full list: the companion table)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    real = [r for r in results if not r["router"].startswith(("R0", "Oracle"))]
    best = max(real, key=lambda r: r["accuracy"])["router"] if real else None
    anchors = {"Oracle", "R0-always-naive", "R0-always-mix", best}
    for r in results:
        family = next((f for f in ROUTER_FAMILY if r["router"].startswith(f)), "R0")
        color = CATEGORICAL[ROUTER_FAMILY.index(family)]
        ax.scatter(
            r["tokens_per_query"],
            r["accuracy"],
            s=40,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        if r["router"] in anchors:
            ax.annotate(
                r["router"].split("(")[0],
                (r["tokens_per_query"], r["accuracy"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color=INK_SECONDARY,
            )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=CATEGORICAL[i], label=f)
        for i, f in enumerate(ROUTER_FAMILY)
    ]
    ax.legend(handles=handles, ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_xlabel("Tokens per question")
    ax.set_ylabel("Correct")
    ax.set_title(f"Router accuracy vs cost: {dataset}", loc="left")
    return _save(fig, path)


def recall_vs_qps(reports: Sequence[Mapping[str, Any]], title: str, path: Path) -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    stores = [r["store"] for r in reports]
    for r in reports:
        curve = r["curve"]
        ax.plot(
            [p["recall"] for p in curve],
            [p["qps"] for p in curve],
            marker="o",
            markersize=4,
            color=color_for(r["store"], stores),
            label=r["store"],
        )
    ax.axvline(0.95, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Recall@10 vs exact search")
    ax.set_ylabel("Queries per second (1 client, log)")
    ax.set_title(title, loc="left")
    ax.legend()
    return _save(fig, path)


def failure_causes(counts: Mapping[str, Mapping[str, int]], path: Path) -> list[Path]:
    apply_style()
    systems = sorted(counts)
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(systems) + 1.6))
    left = [0.0] * len(systems)
    for i, cause in enumerate(CAUSES):
        widths = [counts[s].get(cause, 0) / max(sum(counts[s].values()), 1) for s in systems]
        ax.barh(
            systems,
            widths,
            left=left,
            color=CATEGORICAL[i],
            edgecolor=SURFACE,
            linewidth=1.0,
            label=cause,
        )
        left = [a + b for a, b in zip(left, widths, strict=True)]
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of sampled failures")
    ax.set_title("Failure causes per system", loc="left")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def factor_importance(rows: Sequence[Comparison], title: str, path: Path) -> list[Path]:
    apply_style()
    overall = [r for r in rows if r.query_type == "all"]
    labels = [f"{r.system} ({r.dataset})" for r in overall]
    fig, ax = plt.subplots(figsize=(6.0, 0.35 * len(overall) + 1.4))
    for y, r in enumerate(overall):
        ax.plot([r.diff_low, r.diff_high], [y, y], color=MUTED, linewidth=2)
        ax.scatter([r.diff], [y], s=30, color=CATEGORICAL[0], zorder=3)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_yticks(range(len(overall)), labels)
    ax.set_xlabel("Δ correct vs reference (95% CI)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def interaction(
    points: Mapping[str, Mapping[str, float]], generators: Sequence[str], title: str, path: Path
) -> list[Path]:
    """``points``: retrieval mode → generator → accuracy (one line per mode)."""
    apply_style()
    modes = list(points)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for mode in modes:
        ys = [points[mode].get(g, float("nan")) for g in generators]
        ax.plot(range(len(generators)), ys, marker="o", color=color_for(mode, modes), label=mode)
    ax.set_xticks(range(len(generators)), generators)
    ax.set_ylabel("Correct")
    ax.set_title(title, loc="left")
    ax.legend()
    return _save(fig, path)
