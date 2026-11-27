"""Result tables, paired comparisons and figures (FR-O3, FR-E10).

Everything here reads finished run directories (`runs/<experiment>/<run_id>/`) or benchmark
JSON — no number is ever produced without a measurement behind it. Figures built from demo
runs carry a "DEMO — not research results" watermark.

Colour: categorical slots of the validated reference palette in fixed order; light surface,
single y-axis per chart, recessive grid, legend whenever there are ≥ 2 series.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elevate_rag.stats import holm_bonferroni, mcnemar

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"


# ------------------------------------------------------------------------------- loading


def load_runs(runs_root: Path, experiment: str | None = None) -> list[dict[str, Any]]:
    out = []
    root = runs_root / experiment if experiment else runs_root
    for mf in sorted(root.rglob("manifest.json")):
        d = mf.parent
        man = json.loads(mf.read_text())
        if man.get("status") != "complete" or not (d / "summary.json").exists():
            continue
        out.append(
            {"dir": d, "manifest": man, "summary": json.loads((d / "summary.json").read_text())}
        )
    return out


def _metric(summary: dict[str, Any], key: str, block: str = "overall") -> float | None:
    v = summary.get(block, {}).get(key)
    return v["mean"] if isinstance(v, dict) else None


# ------------------------------------------------------------------------------- tables

TABLE_FIELDS = [
    "experiment",
    "variant",
    "dataset",
    "demo",
    "n",
    "accuracy",
    "accuracy_ci",
    "f1",
    "em",
    "recall@10",
    "ndcg@10",
    "refusal_correct",
    "error_rate",
    "latency_p50_s",
    "latency_p95_s",
    "prompt_tokens_mean",
    "context_tokens_mean",
    "cost_per_1k_queries_usd",
]


def results_table(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in runs:
        s, d = r["summary"], r["manifest"]["definition"]
        acc = s["overall"].get("correct")
        eff = s.get("efficiency", {})
        rows.append(
            {
                "experiment": d["experiment"],
                "variant": d["variant"]["name"],
                "dataset": d["dataset"],
                "demo": r["manifest"].get("demo", False),
                "n": s["overall"]["n"],
                "accuracy": acc["mean"] if acc else None,
                "accuracy_ci": f"[{acc['ci_low']:.3f}, {acc['ci_high']:.3f}]" if acc else "",
                "f1": _metric(s, "f1"),
                "em": _metric(s, "em"),
                "recall@10": _metric(s, "recall@10"),
                "ndcg@10": _metric(s, "ndcg@10"),
                "refusal_correct": _metric(s, "refusal_correct"),
                "error_rate": s.get("error_rate"),
                **{
                    k: eff.get(k)
                    for k in (
                        "latency_p50_s",
                        "latency_p95_s",
                        "prompt_tokens_mean",
                        "context_tokens_mean",
                        "cost_per_1k_queries_usd",
                    )
                },
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _correct_by_question(run_dir: Path) -> dict[str, bool]:
    out = {}
    for line in (run_dir / "scored.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["scores"].get("correct") is not None:
            out[r["question_id"]] = bool(r["scores"]["correct"])
    return out


def paired_comparisons(
    runs: Sequence[dict[str, Any]], reference_variant: str
) -> list[dict[str, Any]]:
    """McNemar of every variant vs the reference variant, per dataset, Holm-corrected within
    each (experiment, dataset) family."""
    groups: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for r in runs:
        d = r["manifest"]["definition"]
        groups[(d["experiment"], d["dataset"])][d["variant"]["name"]] = r["dir"]
    out = []
    for (exp, ds), variants in sorted(groups.items()):
        if reference_variant not in variants:
            continue
        ref = _correct_by_question(variants[reference_variant])
        fam: list[dict[str, Any]] = []
        for name, path in sorted(variants.items()):
            if name == reference_variant:
                continue
            other = _correct_by_question(path)
            qids = sorted(set(ref) & set(other))  # alignment by question ID
            if not qids:
                continue
            res = mcnemar([other[q] for q in qids], [ref[q] for q in qids])
            fam.append(
                {
                    "experiment": exp,
                    "dataset": ds,
                    "variant": name,
                    "reference": reference_variant,
                    "n": res.n,
                    "diff": res.diff,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "p": res.p_value,
                    "test": res.test,
                    "cohens_h": res.effect_size,
                }
            )
        for row, (p_adj, rej) in zip(fam, holm_bonferroni([f["p"] for f in fam]), strict=True):
            row["p_holm"], row["significant"] = p_adj, rej
        out += fam
    return out


# ------------------------------------------------------------------------------- figures


def _style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2)


def _finish(fig: Any, path: Path, demo: bool) -> Path:
    if demo:
        fig.text(
            0.99,
            0.995,
            "DEMO — offline stand-in models, not research results",
            ha="right",
            va="top",
            color="#ec4647",
            fontsize=8,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if demo else None)
    fig.savefig(path, dpi=150, facecolor="#ffffff")
    plt.close(fig)
    return path


def fig_accuracy_by_type(runs: Sequence[dict[str, Any]], dataset: str, path: Path) -> Path | None:
    """Central figure (RQ2): accuracy by query type for every mode/variant.

    Small multiples — one panel per query type, one labelled row per variant — so any number
    of variants is shown (no colour-slot cap) and identity never depends on colour."""
    sel = [r for r in runs if r["manifest"]["definition"]["dataset"] == dataset]
    if not sel:
        return None
    sel = sorted(sel, key=lambda r: r["manifest"]["definition"]["variant"]["name"])
    names = [r["manifest"]["definition"]["variant"]["name"] for r in sel]
    types = sorted({t for r in sel for t in r["summary"].get("by_query_type", {})})
    ncols = min(3, len(types))
    nrows = -(-len(types) // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 0.28 * len(names) * nrows + 1.6),
        sharex=True,
        squeeze=False,
    )
    for ax, t in zip(axes.flat, types, strict=False):
        ys, lo, hi, n = [], [], [], 0
        for r in sel:
            c = r["summary"]["by_query_type"].get(t, {}).get("correct")
            n = max(n, r["summary"]["by_query_type"].get(t, {}).get("n", 0))
            ys.append(c["mean"] if c else float("nan"))
            lo.append(c["mean"] - c["ci_low"] if c else 0.0)
            hi.append(c["ci_high"] - c["mean"] if c else 0.0)
        pos = list(range(len(names)))[::-1]
        ax.errorbar(ys, pos, xerr=[lo, hi], fmt="none", ecolor=GRID, elinewidth=3, zorder=1)
        ax.scatter(ys, pos, s=36, color=SERIES[0], edgecolor="#ffffff", linewidth=1.5, zorder=2)
        ax.set_yticks(pos, names, fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f"{t}  (n={n})", loc="left", color=TEXT, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=TEXT2)
    for ax in list(axes.flat)[len(types) :]:
        ax.axis("off")
    fig.suptitle(
        f"Accuracy by query type — {dataset} (dot = accuracy, bar = 95% CI)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
    )
    return _finish(fig, path, any(r["manifest"].get("demo") for r in sel))


def fig_router_scatter(runs: Sequence[dict[str, Any]], dataset: str, path: Path) -> Path | None:
    """RQ3: accuracy vs mean prompt tokens per query, one point per router level / mode."""
    sel = [r for r in runs if r["manifest"]["definition"]["dataset"] == dataset]
    pts: list[tuple[str, float, float]] = []
    for r in sel:
        acc = (r["summary"]["overall"].get("correct") or {}).get("mean")
        if acc is not None:
            pts.append(
                (
                    r["manifest"]["definition"]["variant"]["name"],
                    float(r["summary"]["efficiency"]["prompt_tokens_mean"]),
                    float(acc),
                )
            )
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, x, y) in enumerate(pts):
        ax.scatter(
            [x],
            [y],
            s=64,
            color=SERIES[i % len(SERIES)],
            edgecolor="#ffffff",
            linewidth=2,
            label=name,
            zorder=3,
        )
    _style(ax, f"Router cost vs accuracy — {dataset}", "accuracy")
    ax.set_xlabel("mean prompt tokens per query", color=TEXT2)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, any(r["manifest"].get("demo") for r in sel))


def fig_recall_qps(bench: Sequence[dict[str, Any]], path: Path) -> Path | None:
    """RQ4: one recall@10-vs-QPS curve per vector engine."""
    if not bench:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, b in enumerate(bench[: len(SERIES)]):
        c = b["curve"]
        ax.plot(
            [p["recall@10"] for p in c],
            [p["qps_1_client"] for p in c],
            color=SERIES[i],
            linewidth=2,
            marker="o",
            markersize=5,
            label=b["backend"],
        )
    ax.axvline(0.95, color=TEXT2, linewidth=1, linestyle="--")
    _style(ax, "Recall@10 vs QPS (1 client)", "queries per second")
    ax.set_xlabel("recall@10 vs exact search", color=TEXT2)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, False)


def fig_factor_importance(
    rows: Sequence[dict[str, Any]], path: Path, demo: bool = False
) -> Path | None:
    """RQ5: accuracy change vs the reference per factor level, with 95 % CI (from
    `paired_comparisons`)."""
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["diff"])
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1.5))
    ys = range(len(rows))
    ax.barh(
        list(ys),
        [r["diff"] for r in rows],
        color=SERIES[0],
        height=0.6,
        xerr=[[r["diff"] - r["ci_low"] for r in rows], [r["ci_high"] - r["diff"] for r in rows]],
        error_kw={"elinewidth": 1, "ecolor": TEXT2},
    )
    ax.set_yticks(list(ys), [f"{r['variant']} ({r['dataset']})" for r in rows], fontsize=8)
    ax.axvline(0, color=TEXT2, linewidth=1)
    _style(ax, "Accuracy change vs reference configuration", "")
    ax.set_xlabel("Δ accuracy (paired, 95% CI)", color=TEXT2)
    return _finish(fig, path, demo)


def fig_interaction(
    runs: Sequence[dict[str, Any]],
    path: Path,
    *,
    mode_key: str = "mode",
    generator_key: str = "generator_model",
) -> Path | None:
    """E6: retrieval mode × generator size — one line per mode across generators."""
    pts: dict[str, dict[str, float]] = defaultdict(dict)
    for r in runs:
        d = r["manifest"]["definition"]
        acc = (r["summary"]["overall"].get("correct") or {}).get("mean")
        gen = d["config"]["models"]["generator"]["model"]
        if acc is not None and d["variant"].get(mode_key):
            pts[d["variant"][mode_key]][gen] = acc
    if not pts:
        return None
    gens = sorted({g for v in pts.values() for g in v})
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (mode, by_gen) in enumerate(sorted(pts.items())):
        ax.plot(
            range(len(gens)),
            [by_gen.get(g, float("nan")) for g in gens],
            color=SERIES[i % len(SERIES)],
            linewidth=2,
            marker="o",
            markersize=6,
            label=mode,
        )
    ax.set_xticks(range(len(gens)), gens, fontsize=8)
    _style(ax, "Retrieval mode × generator", "accuracy")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, any(r["manifest"].get("demo") for r in runs))


def fig_failure_causes(counts: dict[str, dict[str, int]], path: Path) -> Path | None:
    """Failure-cause chart: one horizontal bar group per system (human-labelled counts)."""
    if not counts:
        return None
    causes = sorted({c for v in counts.values() for c in v})
    systems = sorted(counts)[: len(SERIES)]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(causes) * max(1, len(systems)) / 2 + 2))
    h = 0.8 / len(systems)
    for i, sysname in enumerate(systems):
        ax.barh(
            [j + i * h for j in range(len(causes))],
            [counts[sysname].get(c, 0) for c in causes],
            height=h * 0.9,
            color=SERIES[i],
            label=sysname,
        )
    ax.set_yticks([j + 0.4 - h / 2 for j in range(len(causes))], causes)
    _style(ax, "Failure causes (human-labelled samples)", "")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, False)


def write_markdown_report(
    path: Path,
    table: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    figures: Sequence[Path],
) -> None:
    demo = any(r["demo"] for r in table)
    lines = ["# Elevate-RAG results", ""]
    if demo:
        lines += [
            "> **DEMO OUTPUT** — produced with offline stand-in providers on synthetic "
            "data. These numbers are *not* research results.",
            "",
        ]
    lines += [
        "## Runs",
        "",
        "| " + " | ".join(TABLE_FIELDS) + " |",
        "|" + "---|" * len(TABLE_FIELDS),
    ]
    for r in table:
        lines.append("| " + " | ".join(_fmt(r.get(f)) for f in TABLE_FIELDS) + " |")
    if comparisons:
        cols = [
            "experiment",
            "dataset",
            "variant",
            "reference",
            "n",
            "diff",
            "ci_low",
            "ci_high",
            "p",
            "p_holm",
            "significant",
        ]
        lines += [
            "",
            "## Paired comparisons (McNemar, Holm-corrected)",
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "---|" * len(cols),
        ]
        lines += ["| " + " | ".join(_fmt(c.get(k)) for k in cols) + " |" for c in comparisons]
        lines += [
            "",
            "Differences whose CI includes 0 are *no clear difference*, not "
            "evidence of equivalence.",
        ]
    if figures:
        lines += ["", "## Figures", ""] + [f"![{f.stem}]({f.name})" for f in figures]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
