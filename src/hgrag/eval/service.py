"""Orchestration used by ``hgrag eval``: prepare (and price) runs, execute them, score them."""

from __future__ import annotations

import asyncio
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hgrag.config import AppConfig
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, load_queries
from hgrag.data.schema import Query, Split
from hgrag.eval.batch import BatchOutcome, read_models
from hgrag.eval.experiments import load_experiment
from hgrag.eval.faithfulness import ContextMetric, ContextScore, context_metric_for_run
from hgrag.eval.human_labels import LabelItem, build_label_sample
from hgrag.eval.judge import AnswerJudge, Judgment, judge_predictions, pending_judgments
from hgrag.eval.runner import (
    EvalRunner,
    RunOutcome,
    RunPlan,
    RunSpec,
    check_split_allowed,
    load_predictions,
    select_queries,
    write_run_config,
)
from hgrag.eval.scoring import RunMetrics, score_run
from hgrag.index.embedders import Embedder
from hgrag.index.paths import IndexPaths
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.ledger import GroupBy, UsageLedger
from hgrag.pipelines.base import Pipeline, Prediction
from hgrag.pipelines.registry import PipelineContext, build_pipeline
from hgrag.retrieval.rerank import Reranker

GatewayFactory = Callable[[AppConfig, RunContext], LLMGateway]
JUDGMENTS_FILE = "judgments.jsonl"
CONTEXT_METRIC_FILES: dict[str, str] = {
    "faithfulness": "faithfulness.jsonl",
    "evidence_recall": "evidence_recall.jsonl",
}


@dataclass(frozen=True)
class PipelineOptions:
    """Retrieval overrides for one experiment arm (E5 / ablations). None = the reference."""

    chunker: str | None = None
    embedder: str | None = None
    reranker: str | None = None
    repeat: int | None = None  # variance check: uncached repeat number (1, 2, 3...)
    extractor: str | None = None  # graph pipelines (E5 extractor comparison)
    graph_backend: str | None = None  # graph pipelines (E4 Level B)
    context_token_budget: int | None = None  # ablation A3
    embedder_model: Embedder | None = None  # injected models (tests)
    reranker_model: Reranker | None = None

    @property
    def tag(self) -> str:
        """Distinguishes run directories of non-reference arms, e.g. ``chunker=sent256``."""
        parts = [
            f"{key}={value}"
            for key, value in (
                ("chunker", self.chunker),
                ("embedder", self.embedder),
                ("reranker", self.reranker),
                ("extractor", self.extractor),
                ("backend", self.graph_backend),
                ("budget", str(self.context_token_budget) if self.context_token_budget else None),
                ("repeat", str(self.repeat) if self.repeat else None),
            )
            if value
        ]
        return ",".join(parts)


@dataclass(frozen=True)
class PreparedRun:
    spec: RunSpec
    out_dir: Path
    queries: list[Query]
    pipeline: Pipeline
    runner: EvalRunner
    plan: RunPlan


def run_dir(config: AppConfig, spec: RunSpec) -> Path:
    return config.paths.runs_dir / spec.experiment / spec.run_name


def prepare_runs(
    config: AppConfig,
    *,
    experiment: str,
    datasets: Sequence[str],
    split: Split,
    pipeline_name: str,
    limit: int | None = None,
    model_key: str | None = None,
    concurrency: int = 8,
    gateway_factory: GatewayFactory = build_gateway,
    options: PipelineOptions | None = None,
) -> list[PreparedRun]:
    """Build every run and its cost plan. Raises before anything is spent if a rule is broken."""
    options = options or PipelineOptions()
    frozen = load_experiment(experiment).frozen
    prepared: list[PreparedRun] = []
    for dataset in datasets:
        spec = RunSpec(experiment, dataset, split, pipeline_name, limit, frozen, options.tag)
        check_split_allowed(spec)
        queries = select_queries(load_queries(dataset_files(config, dataset), split), limit)
        context = RunContext(run_id=spec.run_id, experiment=experiment, phase="eval")
        pipeline = build_pipeline(
            pipeline_name,
            PipelineContext(
                config=config,
                gateway=gateway_factory(config, context),
                dataset=dataset,
                model_key=model_key,
                chunker=options.chunker,
                embedder=options.embedder,
                reranker=options.reranker,
                embedder_model=options.embedder_model,
                reranker_model=options.reranker_model,
                use_cache=options.repeat is None,
                extractor=options.extractor,
                graph_backend=options.graph_backend,
                context_token_budget=options.context_token_budget,
            ),
        )
        out_dir = run_dir(config, spec)
        runner = EvalRunner(pipeline, out_dir, concurrency=concurrency)
        prepared.append(PreparedRun(spec, out_dir, queries, pipeline, runner, runner.plan(queries)))
    return prepared


def execute(config: AppConfig, run: PreparedRun) -> tuple[RunOutcome, RunMetrics]:
    """Answer the pending questions, record usage, then score the run."""
    write_run_config(run.out_dir, run.spec, run.pipeline.describe())
    outcome = asyncio.run(run.runner.run(run.queries))
    usage = UsageLedger(config.paths.ledger_db).summarize(GroupBy.ROLE, run_id=run.spec.run_id)
    (run.out_dir / "usage.json").write_text(
        json.dumps([asdict(row) for row in usage], indent=2), encoding="utf-8"
    )
    return outcome, score_run_dir(config, run.out_dir)


@dataclass(frozen=True)
class RunInputs:
    """Everything stored for one run directory, reloaded for scoring and judging."""

    spec: RunSpec
    pipeline: dict[str, Any]
    queries: dict[str, Query]
    predictions: list[Prediction]


def load_run(config: AppConfig, out_dir: Path) -> RunInputs:
    stored = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    spec = RunSpec(**stored["spec"])
    files = dataset_files(config, spec.dataset)
    queries = select_queries(load_queries(files, spec.split), spec.limit)
    return RunInputs(
        spec=spec,
        pipeline=stored["pipeline"],
        queries={q.qid: q for q in queries},
        predictions=load_predictions(out_dir / "predictions.jsonl"),
    )


def score_run_dir(config: AppConfig, out_dir: Path) -> RunMetrics:
    """(Re)compute ``scores.jsonl`` and ``metrics.json`` for one run directory."""
    run = load_run(config, out_dir)
    scores, metrics = score_run(
        run.queries,
        run.predictions,
        {j.qid: j.score for j in read_models(out_dir / JUDGMENTS_FILE, Judgment)},
        gold_chunks=load_gold_chunks(config, run.pipeline),
        faithfulness=_context_scores(out_dir, "faithfulness"),
        evidence_recall=_context_scores(out_dir, "evidence_recall"),
    )
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(s.model_dump_json() + "\n" for s in scores)
    (out_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics


def _context_scores(out_dir: Path, metric: str) -> dict[str, float | None]:
    path = out_dir / CONTEXT_METRIC_FILES[metric]
    return {row.qid: row.score for row in read_models(path, ContextScore)}


def load_gold_chunks(config: AppConfig, pipeline: dict[str, Any]) -> dict[str, list[str]] | None:
    """Gold chunk IDs from the index a retrieval pipeline used (None for V0)."""
    index = pipeline.get("index")
    if not index:
        return None
    path = IndexPaths.of(config, index["dataset"], index["chunker"]).gold_chunks
    if not path.is_file():
        return None
    gold: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    return gold


# -- judge and faithfulness ----------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeJob:
    out_dir: Path
    judge: AnswerJudge
    pending: list[tuple[Query, Prediction]]
    max_cost_usd: float


def prepare_judging(
    config: AppConfig,
    out_dirs: Sequence[Path],
    *,
    model_key: str | None = None,
    gateway_factory: GatewayFactory = build_gateway,
) -> list[JudgeJob]:
    jobs = []
    for out_dir in out_dirs:
        run = load_run(config, out_dir)
        context = RunContext(
            run_id=f"{run.spec.run_id}.judge", experiment=run.spec.experiment, phase="judge"
        )
        judge = AnswerJudge(gateway_factory(config, context), model_key=model_key)
        pending = pending_judgments(run.queries, run.predictions, out_dir / JUDGMENTS_FILE)
        cost = sum(judge.estimate_cost(q, p) for q, p in pending)
        jobs.append(JudgeJob(out_dir, judge, pending, cost))
    return jobs


def execute_judging(config: AppConfig, job: JudgeJob, *, concurrency: int = 8) -> BatchOutcome:
    outcome = asyncio.run(
        judge_predictions(
            job.judge, job.pending, job.out_dir / JUDGMENTS_FILE, concurrency=concurrency
        )
    )
    score_run_dir(config, job.out_dir)
    return outcome


def execute_context_metric(
    config: AppConfig,
    out_dir: Path,
    metric: ContextMetric,
    *,
    limit: int | None,
    model_key: str | None = None,
    concurrency: int = 8,
    gateway_factory: GatewayFactory = build_gateway,
) -> BatchOutcome:
    """Faithfulness or evidence recall for the first ``limit`` eligible answers (fixed order)."""
    run = load_run(config, out_dir)
    context = RunContext(
        run_id=f"{run.spec.run_id}.{metric}", experiment=run.spec.experiment, phase="judge"
    )
    order = {q.qid: i for i, q in enumerate(select_queries(list(run.queries.values()), None))}
    eligible = sorted(
        (
            p
            for p in run.predictions
            if p.contexts
            and (metric == "evidence_recall" or not p.refused)
            and (metric == "faithfulness" or run.queries[p.qid].gold_evidence)
        ),
        key=lambda p: order[p.qid],
    )
    outcome = asyncio.run(
        context_metric_for_run(
            gateway_factory(config, context),
            metric,
            run.queries,
            eligible[:limit],
            out_dir / CONTEXT_METRIC_FILES[metric],
            model_key=model_key,
            concurrency=concurrency,
        )
    )
    score_run_dir(config, out_dir)
    return outcome


# -- human labels --------------------------------------------------------------------------------


def label_sample(config: AppConfig, experiment: str, n: int) -> list[LabelItem]:
    runs = []
    for out_dir in experiment_run_dirs(config, experiment):
        run = load_run(config, out_dir)
        runs.append((out_dir.name, run.queries, run.predictions))
    return build_label_sample(runs, n, config.seed)


def judge_verdicts(config: AppConfig, experiment: str) -> dict[tuple[str, str], str]:
    """(run name, qid) → judge verdict, across all runs of an experiment."""
    return {
        (out_dir.name, j.qid): j.verdict
        for out_dir in experiment_run_dirs(config, experiment)
        for j in read_models(out_dir / JUDGMENTS_FILE, Judgment)
        if j.judge_model != "rule"
    }


def experiment_run_dirs(config: AppConfig, experiment: str) -> list[Path]:
    root = config.paths.runs_dir / experiment
    return (
        sorted(p for p in root.glob("*") if (p / "config.json").is_file()) if root.is_dir() else []
    )


# -- variance -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceReport:
    n_repeats: int
    n_questions: int  # answered in every repeat
    accuracies: list[float]
    accuracy_sd: float
    identical_answer_rate: float


def repeat_groups(config: AppConfig, experiment: str) -> dict[str, VarianceReport]:
    """Group ``--repeat`` runs by their base run and compare them question by question."""
    grouped: dict[str, list[Path]] = {}
    for out_dir in experiment_run_dirs(config, experiment):
        spec = load_run(config, out_dir).spec
        if "repeat=" in spec.tag:
            base_tag = ",".join(p for p in spec.tag.split(",") if not p.startswith("repeat="))
            base = RunSpec(spec.experiment, spec.dataset, spec.split, spec.pipeline, tag=base_tag)
            grouped.setdefault(base.run_name, []).append(out_dir)

    reports = {}
    for name, dirs in sorted(grouped.items()):
        runs = [load_run(config, d) for d in dirs]
        answers = [{p.qid: p.answer for p in run.predictions} for run in runs]
        shared = sorted(set.intersection(*(set(a) for a in answers)))
        correct = [
            {s.qid: s.correct for s in score_run(run.queries, run.predictions)[0]} for run in runs
        ]
        accuracies = [sum(c[q] for q in shared) / max(len(shared), 1) for c in correct]
        identical = sum(len({a[q] for a in answers}) == 1 for q in shared)
        reports[name] = VarianceReport(
            n_repeats=len(runs),
            n_questions=len(shared),
            accuracies=accuracies,
            accuracy_sd=statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
            identical_answer_rate=identical / max(len(shared), 1),
        )
    return reports
