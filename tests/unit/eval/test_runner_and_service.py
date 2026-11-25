import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from hgrag.config import AppConfig, BudgetConfig, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, write_jsonl
from hgrag.data.schema import Query
from hgrag.eval.runner import (
    EvalRunner,
    RunSpec,
    SplitLockedError,
    check_split_allowed,
    load_predictions,
    select_queries,
)
from hgrag.eval.service import execute, prepare_runs, score_run_dir
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.providers.fake import FakeProvider
from hgrag.pipelines.closed_book import ClosedBookPipeline

FIXTURE = Path(__file__).parents[2] / "fixtures" / "tiny_corpus" / "queries.jsonl"


def _fixture_queries() -> list[Query]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [Query(**json.loads(line), split="dev") for line in handle if line.strip()]


def _fake_gateway(config: AppConfig, context: RunContext | None = None) -> LLMGateway:
    provider = FakeProvider()

    def factory(_: ProviderSpec) -> FakeProvider:
        return provider

    return build_gateway(config, context or RunContext(run_id="t"), provider_factory=factory)


def _pipeline(config: AppConfig) -> ClosedBookPipeline:
    return ClosedBookPipeline(_fake_gateway(config), model_key="fake-small")


def test_runner_writes_predictions_and_resumes(config: AppConfig, tmp_path: Path) -> None:
    queries = _fixture_queries()
    runner = EvalRunner(_pipeline(config), tmp_path / "run", concurrency=4)

    first = asyncio.run(runner.run(queries[:5]))
    assert first.n_new == 5 and first.stopped_reason is None
    second = asyncio.run(runner.run(queries))  # resume: only the 10 new ones
    assert second.n_new == 10
    predictions = load_predictions(runner.predictions_path)
    assert sorted(p.qid for p in predictions) == sorted(q.qid for q in queries)
    assert runner.plan(queries).n_pending == 0


def test_plan_prices_pending_and_counts_cache_as_free(config: AppConfig, tmp_path: Path) -> None:
    queries = _fixture_queries()[:3]
    pipeline = _pipeline(config)
    runner = EvalRunner(pipeline, tmp_path / "run")
    assert runner.plan(queries).max_cost_usd > 0
    asyncio.run(runner.run(queries[:1]))
    other = EvalRunner(pipeline, tmp_path / "other")  # same answers now cached
    plan = other.plan(queries)
    assert plan.n_pending == 3
    assert plan.max_cost_usd == pytest.approx(runner.plan(queries[1:]).max_cost_usd)


def test_budget_stop_is_graceful_and_resumable(config: AppConfig, tmp_path: Path) -> None:
    broke = config.model_copy(
        update={
            "budget": BudgetConfig(global_cap_usd=1e-9, per_run_cap_usd=1e-9, confirm_above_usd=0)
        }
    )
    runner = EvalRunner(_pipeline(broke), tmp_path / "run")
    outcome = asyncio.run(runner.run(_fixture_queries()))
    assert outcome.n_new == 0 and outcome.stopped_reason and "cap" in outcome.stopped_reason


class _FlakyPipeline(ClosedBookPipeline):
    async def answer(self, query: Query) -> Any:
        if query.qid == "q01":
            raise ConnectionError("boom")
        return await super().answer(query)


def test_one_failing_question_does_not_stop_the_run(config: AppConfig, tmp_path: Path) -> None:
    pipeline = _FlakyPipeline(_fake_gateway(config), model_key="fake-small")
    outcome = asyncio.run(EvalRunner(pipeline, tmp_path / "run").run(_fixture_queries()))
    assert outcome.n_failed == 1 and outcome.n_new == 14


def test_test_split_is_locked_until_frozen() -> None:
    with pytest.raises(SplitLockedError):
        check_split_allowed(RunSpec("e1", "nq", "test", "v0"))
    check_split_allowed(RunSpec("e1", "nq", "test", "v0", frozen=True))
    check_split_allowed(RunSpec("e1", "nq", "dev", "v0"))


def test_select_queries_is_deterministic_and_mixes_types() -> None:
    queries = _fixture_queries()
    picked = select_queries(queries, 6)
    assert picked == select_queries(list(reversed(queries)), 6)
    assert len({q.query_type for q in picked}) > 1


def test_prepare_execute_and_rescore_end_to_end(config: AppConfig) -> None:
    write_jsonl(dataset_files(config, "tiny").queries, _fixture_queries())

    runs = prepare_runs(
        config,
        experiment="unit",
        datasets=["tiny"],
        split="dev",
        pipeline_name="v0",
        limit=10,
        model_key="fake-small",
        gateway_factory=_fake_gateway,
    )
    (run,) = runs
    assert run.plan.n_pending == 10 and run.plan.max_cost_usd > 0

    outcome, metrics = execute(config, run)
    assert outcome.n_new == 10 and metrics.overall.n == 10
    assert metrics.n_missing_predictions == 0
    for name in ("config.json", "predictions.jsonl", "usage.json", "scores.jsonl", "metrics.json"):
        assert (run.out_dir / name).is_file(), name
    config_json = json.loads((run.out_dir / "config.json").read_text())
    assert config_json["pipeline"]["generator_model"] == "fake-small"
    assert config_json["run_id"] == "unit.tiny__v0__dev"

    usage = json.loads((run.out_dir / "usage.json").read_text())
    assert usage[0]["group"] == "generator" and usage[0]["calls"] == 10

    assert score_run_dir(config, run.out_dir) == metrics  # re-scoring is reproducible


def test_repeat_runs_bypass_cache_and_report_variance(config: AppConfig) -> None:
    from hgrag.eval.service import PipelineOptions, repeat_groups

    write_jsonl(dataset_files(config, "tiny").queries, _fixture_queries())
    for repeat in (1, 2):
        (run,) = prepare_runs(
            config,
            experiment="var",
            datasets=["tiny"],
            split="dev",
            pipeline_name="v0",
            limit=5,
            model_key="fake-small",
            gateway_factory=_fake_gateway,
            options=PipelineOptions(repeat=repeat),
        )
        assert run.out_dir.name == f"tiny__v0[repeat={repeat}]__dev"
        assert run.plan.max_cost_usd > 0  # repeats are priced even when answers are cached
        outcome, _ = execute(config, run)
        assert outcome.n_new == 5
        predictions = load_predictions(run.out_dir / "predictions.jsonl")
        assert not any(p.cache_hit for p in predictions)  # uncached even on the 2nd repeat

    report = repeat_groups(config, "var")["tiny__v0__dev"]
    assert report.n_repeats == 2 and report.n_questions == 5
    assert report.identical_answer_rate == 1.0  # the fake model is deterministic
