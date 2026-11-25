"""Index build, search and V1-V4 end to end on the fictional fixture corpus (offline)."""

import json
from pathlib import Path

import pytest

from hgrag.config import AppConfig, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, write_jsonl
from hgrag.data.schema import Document, Query
from hgrag.eval.service import PipelineOptions, execute, prepare_runs
from hgrag.index.build import build_index
from hgrag.index.embedders import HashingEmbedder
from hgrag.index.paths import IndexPaths
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.providers.fake import FakeProvider
from hgrag.retrieval.evaluate import evaluate_retrieval
from hgrag.retrieval.rerank import OverlapReranker
from hgrag.retrieval.searcher import RetrievalSpec, Searcher

FIXTURE = Path(__file__).parents[2] / "fixtures" / "tiny_corpus"


def _read(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines() if line.strip()]


@pytest.fixture
def tiny(config: AppConfig) -> AppConfig:
    """Fixture corpus written as a processed dataset named 'tiny', with an index built."""
    files = dataset_files(config, "tiny")
    write_jsonl(files.corpus, [Document(**row) for row in _read("corpus.jsonl")])  # type: ignore[arg-type]
    write_jsonl(files.queries, [Query(**row, split="dev") for row in _read("queries.jsonl")])  # type: ignore[arg-type]
    build_index(config, "tiny", embedders=["hashing"], embedder_loader=lambda _: HashingEmbedder())
    return config


def _searcher(config: AppConfig, retriever: str, rerank: bool = False) -> Searcher:
    return Searcher(
        config,
        "tiny",
        "sent512",
        RetrievalSpec(retriever, rerank),  # type: ignore[arg-type]
        embedder=HashingEmbedder(),
        reranker=OverlapReranker() if rerank else None,
    )


def test_index_layout_and_gold_map(tiny: AppConfig) -> None:
    paths = IndexPaths.of(tiny, "tiny", "sent512")
    for path in (paths.chunks, paths.gold_chunks, paths.manifest, paths.hnsw("hashing")):
        assert path.exists(), path
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["n_chunks"] == 15  # every fixture doc is short → one chunk each
    gold = json.loads(paths.gold_chunks.read_text())
    assert "q01" in gold and "q14" not in gold  # unanswerable questions have no gold


def test_rebuild_is_idempotent(tiny: AppConfig) -> None:
    paths = IndexPaths.of(tiny, "tiny", "sent512")
    before = paths.chunks.stat().st_mtime_ns
    build_index(tiny, "tiny", embedders=["hashing"], embedder_loader=lambda _: HashingEmbedder())
    assert paths.chunks.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    ("retriever", "rerank"),
    [("bm25", False), ("dense", False), ("hybrid", False), ("hybrid", True)],
)
def test_every_variant_finds_single_hop_gold(tiny: AppConfig, retriever: str, rerank: bool) -> None:
    gold = json.loads(IndexPaths.of(tiny, "tiny", "sent512").gold_chunks.read_text())
    queries = [Query(**row) for row in _read("queries.jsonl") if row["query_type"] == "single_hop"]  # type: ignore[arg-type]
    report = evaluate_retrieval(_searcher(tiny, retriever, rerank), queries, gold, retriever)
    assert report.n_queries == 4
    assert report.chunk_recall[5] >= 0.75


def test_search_results_are_cached(tiny: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = _searcher(tiny, "bm25")
    first = searcher.search("Who built the Halden Refractor?")
    monkeypatch.setattr(searcher, "_rank", lambda _: pytest.fail("should hit the cache"))
    assert searcher.search("Who built the Halden Refractor?") == first


def test_dense_retrieval_without_embedder_is_rejected(tiny: AppConfig) -> None:
    with pytest.raises(ValueError, match="needs an embedder"):
        Searcher(tiny, "tiny", "sent512", RetrievalSpec("dense", False))


def _fake_gateway(config: AppConfig, context: RunContext) -> LLMGateway:
    provider = FakeProvider()

    def factory(_: ProviderSpec) -> FakeProvider:
        return provider

    return build_gateway(config, context, provider_factory=factory)


def test_v4_pipeline_end_to_end_with_chunk_level_metrics(tiny: AppConfig) -> None:
    (run,) = prepare_runs(
        tiny,
        experiment="unit",
        datasets=["tiny"],
        split="dev",
        pipeline_name="v4",
        model_key="fake-small",
        gateway_factory=_fake_gateway,
        options=PipelineOptions(embedder_model=HashingEmbedder(), reranker_model=OverlapReranker()),
    )
    assert run.plan.n_pending == 15 and run.plan.max_cost_usd > 0
    outcome, metrics = execute(tiny, run)
    assert outcome.n_new == 15

    described = json.loads((run.out_dir / "config.json").read_text())["pipeline"]
    assert described["retriever"] == "hybrid" and described["reranker"] == "overlap"
    assert described["index"] == {"dataset": "tiny", "chunker": "sent512"}
    assert "chunk_recall@5" in metrics.overall.metrics and "doc_recall@5" in metrics.overall.metrics

    prediction = json.loads((run.out_dir / "predictions.jsonl").read_text().splitlines()[0])
    assert prediction["contexts"] and prediction["contexts"][0]["kind"] == "chunk"
    assert "Passages:" in prediction["answer"] or prediction["answer"].startswith("FAKE")


def test_rerank_candidates_limits_what_the_reranker_scores(tiny: AppConfig) -> None:
    class CountingReranker(OverlapReranker):
        seen = 0

        def score(self, query: str, passages: list[str]) -> list[float]:  # type: ignore[override]
            CountingReranker.seen = len(passages)
            return super().score(query, passages)

    settings = tiny.retrieval.search.model_copy(update={"rerank_candidates": 3})
    searcher = Searcher(
        tiny,
        "tiny",
        "sent512",
        RetrievalSpec("hybrid", True),
        embedder=HashingEmbedder(),
        reranker=CountingReranker(),
        settings=settings,
    )
    hits = searcher.search("Who built the Halden Refractor?")
    assert CountingReranker.seen == 3
    assert len(hits) <= tiny.retrieval.search.final_k
    assert searcher.describe()["search"]["rerank_candidates"] == 3  # recorded in run configs
