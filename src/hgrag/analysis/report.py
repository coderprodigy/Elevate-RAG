"""Regenerate every report table and figure from ``runs/`` and ``reports/`` (no API calls)."""

from __future__ import annotations

import json
from pathlib import Path

from hgrag.analysis import figures, tables
from hgrag.analysis.collect import RunRecord, load_experiment_runs
from hgrag.analysis.compare import compare_to_reference
from hgrag.analysis.errors import FailureTag, cause_counts
from hgrag.analysis.style import PIPELINE_ORDER
from hgrag.config import AppConfig
from hgrag.eval.batch import read_models

REFERENCE = "v4"


def _family(runs: list[RunRecord]) -> list[str]:
    names = {r.pipeline for r in runs}
    return next(
        (order for order in PIPELINE_ORDER.values() if names <= set(order)),
        sorted(names),
    )


def build_report(config: AppConfig, reports_dir: Path) -> list[Path]:
    written: list[Path] = []
    tables_dir, figures_dir = reports_dir / "tables", reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    experiments = sorted(
        p.name for p in config.paths.runs_dir.glob("*") if p.is_dir() and p.name != "cache"
    )
    for experiment in experiments:
        for split in ("dev", "test"):
            runs = load_experiment_runs(config, experiment, split)
            if not runs:
                continue
            stem = f"{experiment}_{split}"
            written.append(_write(tables_dir / f"{stem}_summary.md", tables.summary_table(runs)))
            written.append(_write(tables_dir / f"{stem}_by_type.md", tables.by_type_table(runs)))
            reference = (
                REFERENCE if any(r.pipeline == REFERENCE for r in runs) else runs[0].pipeline
            )
            comparisons = compare_to_reference(runs, reference)
            if comparisons:
                written.append(
                    _write(
                        tables_dir / f"{stem}_vs_{reference}.md",
                        tables.comparison_table(comparisons),
                    )
                )
                if experiment.startswith("e5"):
                    written += figures.factor_importance(
                        comparisons,
                        f"What matters most ({experiment}, {split})",
                        figures_dir / f"{stem}_factors",
                    )
            if any(r.scores for r in runs):
                written += figures.accuracy_by_type(
                    runs,
                    _family(runs),
                    f"Correct by query type ({experiment}, {split})",
                    figures_dir / f"{stem}_by_type",
                )
        tags = [
            tag
            for f in (config.paths.runs_dir / experiment / "errors").glob("*.jsonl")
            if f.stem != "sample"
            for tag in read_models(f, FailureTag)
        ]
        if tags:
            written += figures.failure_causes(
                cause_counts(tags), figures_dir / f"{experiment}_failures"
            )

    for router_file in sorted(reports_dir.glob("router_*.json")):
        for dataset_report in json.loads(router_file.read_text()):
            written += figures.router_scatter(
                dataset_report["results"],
                dataset_report["dataset"],
                figures_dir / f"{router_file.stem}_{dataset_report['dataset']}",
            )
    for bench_file in sorted((reports_dir / "bench").glob("vector_*.json")):
        written += figures.recall_vs_qps(
            json.loads(bench_file.read_text()),
            f"Recall vs QPS ({bench_file.stem})",
            figures_dir / bench_file.stem,
        )
    return written


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
