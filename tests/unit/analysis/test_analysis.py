import json
from pathlib import Path

import pytest

from hgrag.analysis.collect import load_experiment_runs
from hgrag.analysis.compare import compare_to_reference
from hgrag.analysis.errors import FailureTag, cause_counts, sample_failures, suggest_cause
from hgrag.analysis.report import build_report
from hgrag.analysis.tables import comparison_table, summary_table
from hgrag.config import AppConfig, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, write_jsonl
from hgrag.data.schema import Document, Query
from hgrag.eval.scoring import QuestionScore, RetrievalScore
from hgrag.eval.service import execute, prepare_runs
from hgrag.index.build import build_index
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.providers.fake import FakeProvider

FIXTURE = Path(__file__).parents[2] / "fixtures" / "tiny_corpus"


def _fake(config: AppConfig, context: RunContext) -> LLMGateway:
    provider = FakeProvider()

    def factory(_: ProviderSpec) -> FakeProvider:
        return provider

    return build_gateway(config, context, provider_factory=factory)


def _read(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines() if line.strip()]


@pytest.fixture
def experiment(config: AppConfig) -> AppConfig:
    """Two real systems (v0 closed-book, v1 BM25) run on the fixture corpus with the fake LLM."""
    files = dataset_files(config, "tiny")
    write_jsonl(files.corpus, [Document(**r) for r in _read("corpus.jsonl")])  # type: ignore[arg-type]
    write_jsonl(files.queries, [Query(**r, split="dev") for r in _read("queries.jsonl")])  # type: ignore[arg-type]
    build_index(config, "tiny", embedders=[])
    for pipeline in ("v0", "v1"):
        (run,) = prepare_runs(
            config,
            experiment="ana",
            datasets=["tiny"],
            split="dev",
            pipeline_name=pipeline,
            model_key="fake-small",
            gateway_factory=_fake,
        )
        execute(config, run)
    return config


def test_collect_compare_and_tables(experiment: AppConfig) -> None:
    runs = load_experiment_runs(experiment, "ana", "dev")
    assert {r.pipeline for r in runs} == {"v0", "v1"} and all(len(r.scores) == 15 for r in runs)
    rows = compare_to_reference(runs, reference="v0")
    overall = next(r for r in rows if r.query_type == "all")
    assert overall.system == "v1" and overall.n == 15
    assert overall.verdict in ("better", "worse", "no clear difference")
    assert "| tiny | v1 |" in summary_table(runs) and "Δ accuracy" in comparison_table(rows)


def test_build_report_writes_tables_and_figures(experiment: AppConfig, tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    (reports / "bench").mkdir(parents=True)
    (reports / "bench" / "vector_toy.json").write_text(
        json.dumps(
            [
                {
                    "store": "faiss-hnsw",
                    "curve": [
                        {"ef": 8, "recall": 0.8, "qps": 900.0},
                        {"ef": 64, "recall": 0.97, "qps": 400.0},
                    ],
                }
            ]
        )
    )
    (reports / "router_e2.json").write_text(
        json.dumps(
            [
                {
                    "dataset": "toy",
                    "results": [
                        {"router": "R0-always-naive", "tokens_per_query": 900, "accuracy": 0.5},
                        {"router": "R2-tfidf-type", "tokens_per_query": 1200, "accuracy": 0.6},
                        {"router": "Oracle", "tokens_per_query": 1000, "accuracy": 0.7},
                    ],
                }
            ]
        )
    )
    errors = experiment.paths.runs_dir / "ana" / "errors"
    errors.mkdir(parents=True)
    tags = [
        FailureTag(run="tiny__v1__dev", qid=f"q{i}", labeller="me", cause=c).model_dump_json()
        for i, c in enumerate(["retrieval_miss", "generation_error", "retrieval_miss"])
    ]
    (errors / "me.jsonl").write_text("\n".join(tags) + "\n")

    written = {p.name for p in build_report(experiment, reports)}
    for name in (
        "ana_dev_summary.md",
        "ana_dev_by_type.md",
        "ana_dev_by_type.png",
        "ana_failures.png",
        "router_e2_toy.png",
        "vector_toy.png",
        "vector_toy.svg",
    ):
        assert name in written, name


def test_failure_sampling_and_cause_heuristics(experiment: AppConfig) -> None:
    (v1,) = [r for r in load_experiment_runs(experiment, "ana", "dev") if r.pipeline == "v1"]
    items = sample_failures(v1, n=5, seed=1, budget_tokens=8000)
    assert len(items) == 5 and all(
        i.suggested in ("retrieval_miss", "generation_error", "context_overload") for i in items
    )

    def score(recall: float, tokens: int = 100) -> QuestionScore:
        r = RetrievalScore(
            recall={5: recall, 10: recall}, hit={5: recall, 10: recall}, ndcg10=recall
        )
        return QuestionScore(
            qid="q",
            query_type="multi_hop",
            answerable=True,
            correct=0.0,
            exact_match=0.0,
            f1=0.0,
            judge=None,
            faithfulness=None,
            evidence_recall=None,
            refused=False,
            doc_retrieval=r,
            chunk_retrieval=r,
            tokens=tokens,
            cost_usd=0.0,
            latency_ms=1.0,
        )

    assert (
        suggest_cause(score(0.0), is_graph=False, routed_away=False, budget_tokens=8000)
        == "retrieval_miss"
    )
    assert (
        suggest_cause(score(0.0), is_graph=True, routed_away=False, budget_tokens=8000)
        == "extraction_miss"
    )
    assert (
        suggest_cause(score(1.0, 7000), is_graph=False, routed_away=False, budget_tokens=8000)
        == "context_overload"
    )
    assert (
        suggest_cause(score(1.0), is_graph=False, routed_away=False, budget_tokens=8000)
        == "generation_error"
    )
    assert (
        suggest_cause(score(1.0), is_graph=False, routed_away=True, budget_tokens=8000)
        == "wrong_route"
    )
    counts = cause_counts([FailureTag(run="a", qid="1", labeller="x", cause="retrieval_miss")])
    assert counts["a"]["retrieval_miss"] == 1 and counts["a"]["judge_error"] == 0
