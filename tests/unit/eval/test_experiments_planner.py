import json
from pathlib import Path

from hgrag.config import AppConfig, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, write_jsonl
from hgrag.data.schema import Query
from hgrag.eval.experiments import ArmSpec, ExperimentConfig, list_experiments, load_experiment
from hgrag.eval.planner import plan_experiment, run_planned
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.providers.fake import FakeProvider

FIXTURE = Path(__file__).parents[2] / "fixtures" / "tiny_corpus" / "queries.jsonl"


def _fake_gateway(config: AppConfig, context: RunContext) -> LLMGateway:
    provider = FakeProvider()

    def factory(_: ProviderSpec) -> FakeProvider:
        return provider

    return build_gateway(config, context, provider_factory=factory)


def test_vary_gives_one_factor_at_a_time() -> None:
    exp = ExperimentConfig(
        name="x", base=ArmSpec(pipeline="v4"), vary={"embedder": ["a", "b"], "model": ["m"]}
    )
    arms = exp.expand()
    assert len(arms) == 4 and arms[0] == exp.base
    assert all(len(arm.differs_from(exp.base)) <= 1 for arm in arms)
    assert [a.label(exp.base) for a in arms] == ["reference", "embedder=a", "embedder=b", "model=m"]
    assert exp.violations() == []


def test_grid_is_full_factorial_and_unknown_factors_are_flagged() -> None:
    exp = ExperimentConfig(
        name="g",
        base=ArmSpec(pipeline="v4"),
        grid={"pipeline": ["v4", "lr_mix"], "model": ["s", "l"]},
    )
    assert len(exp.expand()) == 5  # base + 4 combinations
    bad = ExperimentConfig(name="b", vary={"temperature": [0.5]})
    assert any("unknown factor" in v for v in bad.violations())


def test_every_shipped_experiment_is_valid() -> None:
    names = list_experiments()
    assert {"g1_smoke", "e1_baselines", "e2_modes", "e6_interaction"} <= set(names)
    for name in names:
        experiment = load_experiment(name)
        assert experiment.violations() == [], name
        assert experiment.expand(), name


def test_plan_reports_ready_and_blocked_runs_then_runs_the_ready_ones(config: AppConfig) -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    write_jsonl(dataset_files(config, "tiny").queries, [Query(**row, split="dev") for row in rows])
    exp = ExperimentConfig(
        name="unit-plan",
        limit=4,
        base=ArmSpec(pipeline="v0", model="fake-small"),
        vary={"pipeline": ["lr_local"], "model": ["no-such-model@waterloo"]},
    )
    planned = plan_experiment(config, exp, datasets=["tiny"], gateway_factory=_fake_gateway)
    status = {p.label: p.blocked for p in planned}
    assert status["reference"] is None
    assert status["pipeline=lr_local"] and "graph" in status["pipeline=lr_local"].lower()
    assert status["model=no-such-model@waterloo"] is not None
    assert sum(p.cost for p in planned) > 0

    results = run_planned(config, planned)
    assert len(results) == 1 and results[0][1].n_new == 4
