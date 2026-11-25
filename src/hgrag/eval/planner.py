"""Plan and run whole experiments: expand arms x datasets, price every run, report blockers.

A run whose prerequisites are missing (index, graph, model key) is reported as *blocked* with
the reason, instead of failing the whole plan, so the plan doubles as a to-do list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hgrag.config import AppConfig, ConfigError
from hgrag.eval.experiments import ArmSpec, ExperimentConfig
from hgrag.eval.runner import RunOutcome, SplitLockedError
from hgrag.eval.scoring import RunMetrics
from hgrag.eval.service import GatewayFactory, PipelineOptions, PreparedRun, execute, prepare_runs
from hgrag.llm.gateway import build_gateway


@dataclass(frozen=True)
class PlannedRun:
    arm: ArmSpec
    label: str
    dataset: str
    prepared: PreparedRun | None
    blocked: str | None  # why it cannot run yet

    @property
    def cost(self) -> float:
        return self.prepared.plan.max_cost_usd if self.prepared else 0.0


def options_for(arm: ArmSpec) -> PipelineOptions:
    return PipelineOptions(
        chunker=arm.chunker,
        embedder=arm.embedder,
        reranker=arm.reranker,
        extractor=arm.extractor,
        graph_backend=arm.graph_backend,
        context_token_budget=arm.context_token_budget,
    )


def plan_experiment(
    config: AppConfig,
    experiment: ExperimentConfig,
    *,
    datasets: Sequence[str],
    split: str | None = None,
    gateway_factory: GatewayFactory = build_gateway,
) -> list[PlannedRun]:
    planned = []
    for arm in experiment.expand():
        label = arm.label(experiment.base)
        for dataset in datasets:
            try:
                (prepared,) = prepare_runs(
                    config,
                    experiment=experiment.name,
                    datasets=[dataset],
                    split=(split or experiment.split),  # type: ignore[arg-type]
                    pipeline_name=arm.pipeline,
                    limit=experiment.limit,
                    model_key=arm.model,
                    options=options_for(arm),
                    gateway_factory=gateway_factory,
                )
                planned.append(PlannedRun(arm, label, dataset, prepared, None))
            except (FileNotFoundError, KeyError, ConfigError, SplitLockedError) as error:
                reason = f"{type(error).__name__}: {str(error).splitlines()[0][:120]}"
                planned.append(PlannedRun(arm, label, dataset, None, reason))
    return planned


def run_planned(
    config: AppConfig, planned: Sequence[PlannedRun]
) -> list[tuple[PlannedRun, RunOutcome, RunMetrics]]:
    results = []
    for run in planned:
        if run.prepared is None:
            continue
        outcome, metrics = execute(config, run.prepared)
        results.append((run, outcome, metrics))
        if outcome.stopped_reason:
            break  # budget cap reached: stop the whole experiment (resumable)
    return results
