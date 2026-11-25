"""Phase 4 end to end with the real LightRAG and a scripted LLM (offline, no API).

The scripted LLM answers LightRAG's real prompts: entity/relation extraction (capitalised names in
the ``---Input Text---`` section), gleaning (nothing more), and keyword extraction (JSON).
"""

import itertools
import json
import re
from pathlib import Path

import pytest

from hgrag.config import AppConfig, ProviderSpec
from hgrag.core.runs import RunContext
from hgrag.data.io import dataset_files, write_jsonl
from hgrag.data.schema import Document, Query
from hgrag.eval.service import PipelineOptions, execute, prepare_runs
from hgrag.graph.audit import sample_triples
from hgrag.graph.build import build_graph, estimate_build
from hgrag.graph.lightrag_adapter import GraphLocation
from hgrag.graph.quality import answer_entity_coverage, graph_shape, load_graph
from hgrag.index.build import build_index, load_chunks
from hgrag.index.embedders import HashingEmbedder
from hgrag.index.paths import IndexPaths
from hgrag.llm.gateway import LLMGateway, build_gateway
from hgrag.llm.types import LLMRequest, ProviderResult
from hgrag.retrieval.rerank import OverlapReranker

FIXTURE = Path(__file__).parents[2] / "fixtures" / "tiny_corpus"
D = "<|#|>"
NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")


class ScriptedLightRAGLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMRequest) -> ProviderResult:
        self.calls += 1
        prompt = request.messages[-1].content
        system = request.system or ""
        if "single JSON object" in system:  # keyword extraction
            names = NAME.findall(prompt.split("Query:")[-1]) or ["Varenne"]
            text = json.dumps({"high_level_keywords": ["related"], "low_level_keywords": names})
        elif "---Input Text---" in prompt:  # entity / relation extraction for one chunk
            body = prompt.split("---Input Text---")[-1]
            names = list(dict.fromkeys(NAME.findall(body)))[:4]
            rows = [f"entity{D}{n}{D}concept{D}{n} appears in the corpus." for n in names]
            rows += [
                f"relation{D}{a}{D}{b}{D}related{D}{a} is related to {b}."
                for a, b in itertools.pairwise(names)
            ]
            text = "\n".join([*rows, "<|COMPLETE|>"])
        else:  # gleaning or description summary
            text = "<|COMPLETE|>" if "missed" in prompt.lower() else "A short summary."
        return ProviderResult(
            text=text, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4
        )


def _gateway_factory(provider: ScriptedLightRAGLLM):  # type: ignore[no-untyped-def]
    def make(config: AppConfig, context: RunContext) -> LLMGateway:
        def factory(_: ProviderSpec) -> ScriptedLightRAGLLM:
            return provider

        return build_gateway(config, context, provider_factory=factory)  # type: ignore[arg-type]

    return make


def _read(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines() if line.strip()]


@pytest.fixture
def built(config: AppConfig) -> tuple[AppConfig, GraphLocation, ScriptedLightRAGLLM]:
    files = dataset_files(config, "tiny")
    write_jsonl(files.corpus, [Document(**row) for row in _read("corpus.jsonl")])  # type: ignore[arg-type]
    write_jsonl(files.queries, [Query(**row, split="dev") for row in _read("queries.jsonl")])  # type: ignore[arg-type]
    build_index(config, "tiny", embedders=["hashing"], embedder_loader=lambda _: HashingEmbedder())
    location = GraphLocation("tiny", "sent512", "claude-haiku-5-5", "hashing", "local")
    provider = ScriptedLightRAGLLM()
    manifest = build_graph(
        config, location, embedder=HashingEmbedder(), gateway_factory=_gateway_factory(provider)
    )
    assert manifest.status_counts.get("processed") == 15, manifest.status_counts
    return config, location, provider


def test_graph_uses_exactly_our_chunks(built: tuple[AppConfig, GraphLocation, object]) -> None:
    config, location, _ = built
    stored = json.loads((location.working_dir(config) / "kv_store_text_chunks.json").read_text())
    ours = {(c.doc_id, c.text) for c in load_chunks(IndexPaths.of(config, "tiny", "sent512"))}
    theirs = {(v["file_path"], v["content"]) for v in stored.values()}
    assert theirs == ours  # decision D2: identical chunks in the vector and graph indexes


def test_graph_quality_and_audit(built: tuple[AppConfig, GraphLocation, object]) -> None:
    config, location, _ = built
    graph = load_graph(location.working_dir(config))
    shape = graph_shape(graph)
    assert shape.nodes > 10 and shape.edges > 5
    queries = [Query(**row) for row in _read("queries.jsonl")]  # type: ignore[arg-type]
    coverage = answer_entity_coverage(graph, queries)
    assert coverage.eligible > 0 and 0 < coverage.rate <= 1  # e.g. "Mira Halden" is a node
    items = sample_triples(graph, location.working_dir(config), n=5, seed=1)
    assert len(items) == 5 and all(item.evidence for item in items)  # source text attached


def test_estimate_and_sample_build(
    config: AppConfig, built: tuple[AppConfig, GraphLocation, object]
) -> None:
    config, _location, provider = built
    estimate = estimate_build(config, "tiny", "sent512", "claude-haiku-5-5")
    assert (
        estimate.n_chunks == 15 and estimate.upper_bound_usd > estimate.sample_upper_bound_usd > 0
    )
    sample = GraphLocation("tiny", "sent512", "claude-haiku-5-5", "hashing", "local", sample=True)
    manifest = build_graph(
        config, sample, embedder=HashingEmbedder(), gateway_factory=_gateway_factory(provider)
    )  # type: ignore[arg-type]
    assert manifest.n_chunks == 1 and manifest.extrapolated_full_usd is not None


@pytest.mark.parametrize("mode", ["naive", "local", "global", "hybrid", "mix"])
def test_every_graph_mode_answers_end_to_end(
    built: tuple[AppConfig, GraphLocation, ScriptedLightRAGLLM], mode: str
) -> None:
    config, _, provider = built
    (run,) = prepare_runs(
        config,
        experiment="unit",
        datasets=["tiny"],
        split="dev",
        pipeline_name=f"lr_{mode}",
        limit=3,
        model_key="fake-small",
        gateway_factory=_gateway_factory(provider),
        options=PipelineOptions(
            embedder_model=HashingEmbedder(),
            reranker_model=OverlapReranker(),
            extractor="claude-haiku-5-5",
        ),
    )
    assert run.plan.max_cost_usd > 0
    outcome, _metrics = execute(config, run)
    assert outcome.n_new == 3 and outcome.n_failed == 0
    described = json.loads((run.out_dir / "config.json").read_text())["pipeline"]
    assert described["mode"] == mode and described["graph"]["backend"] == "local"
    prediction = json.loads((run.out_dir / "predictions.jsonl").read_text().splitlines()[0])
    assert prediction["contexts"], "graph retrieval returned context"
    kinds = {c["kind"] for c in prediction["contexts"]}
    if mode == "naive":
        assert kinds == {"chunk"}
    else:
        assert kinds & {"entity", "relation"}
        assert prediction["meta"]["retrieval_cost_usd"] > 0  # keyword call attributed to query
    chunk_ids = {c["unit_id"] for c in prediction["contexts"] if c["kind"] == "chunk"}
    assert all(cid.startswith("c_") for cid in chunk_ids)  # mapped back to *our* chunk IDs
