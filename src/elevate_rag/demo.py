"""Offline end-to-end demonstration (FR-U2).

Runs the real experiment machinery — data prep, shared index (chunks, BM25, dense, LightRAG
graph), every retrieval mode, routers R0/R1/R3/R4, scoring, statistics and report — on the
SYNTHETIC corpus with deterministic offline providers. No network, no credentials, no cost.
Everything is written under `out/` and labelled DEMO; nothing here is a research result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from elevate_rag.config import load_yaml_with_extends
from elevate_rag.data.synthetic import write_synthetic
from elevate_rag.experiments.runner import ExperimentSpec, Variant, execute_run, expand, validate

DEMO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pipelines" / "demo_offline.yaml"

DEMO_VARIANTS = [
    Variant(name="V0", mode="v0"),
    Variant(name="V1", mode="v1"),
    Variant(name="V2", mode="v2"),
    Variant(name="V3", mode="v3"),
    Variant(name="V4", mode="v4"),
    Variant(name="naive", mode="naive"),
    Variant(name="local", mode="local"),
    Variant(name="global", mode="global"),
    Variant(name="hybrid", mode="hybrid"),
    Variant(name="mix", mode="mix"),
    Variant(name="R1-rules", router={"level": "rules"}),
    Variant(name="R3-llm", router={"level": "llm"}),
    Variant(name="R4-fallback", router={"level": "fallback"}, tune_on_dev=True),
]


async def run_demo(out: Path, config: Path = DEMO_CONFIG) -> dict[str, Any]:
    out = out.resolve()
    base = load_yaml_with_extends(config)
    base.update(
        {
            "data_dir": str(out / "data"),
            "cache_dir": str(out / "cache"),
            "index_dir": str(out / "indexes"),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    write_synthetic(out / "data")
    spec = ExperimentSpec(
        id="DEMO",
        question="demo (all RQs, synthetic)",
        description="Offline demo — not research results",
        pipeline="pipeline.yaml",
        datasets=["synthetic"],
        split="test",
        variants=DEMO_VARIANTS,
        judge=True,
        concurrency=4,
    )
    runs = expand(spec, out)
    problems = validate(spec, runs)
    if problems:
        raise RuntimeError(f"demo spec invalid: {problems}")
    results = {}
    for r in runs:
        summary = await execute_run(r, spec, out / "runs")
        acc = (summary["overall"].get("correct") or {}).get("mean")
        results[r.variant.name] = {
            "accuracy": acc,
            "n": summary["overall"]["n"],
            "recall@10": (summary["overall"].get("recall@10") or {}).get("mean"),
        }
    from elevate_rag.reporting import report as R

    loaded = R.load_runs(out / "runs")
    table = R.results_table(loaded)
    R.write_csv(out / "report" / "results.csv", table, R.TABLE_FIELDS)
    comps = R.paired_comparisons(loaded, "V4")
    figs = [
        p
        for p in (
            R.fig_accuracy_by_type(loaded, "synthetic", out / "report" / "accuracy_by_type.png"),
            R.fig_router_scatter(loaded, "synthetic", out / "report" / "cost_vs_accuracy.png"),
        )
        if p
    ]
    R.write_markdown_report(out / "report" / "REPORT.md", table, comps, figs)
    return {
        "label": "DEMO — synthetic data, offline stand-in models; not research results",
        "runs": results,
        "report": str(out / "report" / "REPORT.md"),
    }
