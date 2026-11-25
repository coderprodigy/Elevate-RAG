"""Pipeline names usable from the CLI and experiment configs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hgrag.config import AppConfig, SearchSettings
from hgrag.graph.build import default_extractor, load_graph_manifest
from hgrag.graph.index import GraphIndex
from hgrag.graph.lightrag_adapter import GRAPH_MODES, GraphLocation
from hgrag.index.build import default_chunker
from hgrag.index.embedders import Embedder
from hgrag.index.models import load_embedder, load_reranker
from hgrag.llm.gateway import LLMGateway
from hgrag.pipelines.base import Pipeline
from hgrag.pipelines.closed_book import ClosedBookPipeline
from hgrag.pipelines.graph import GraphPipeline
from hgrag.pipelines.vector import VARIANTS, VectorPipeline
from hgrag.retrieval.rerank import Reranker
from hgrag.retrieval.searcher import Searcher


@dataclass(frozen=True)
class PipelineContext:
    config: AppConfig
    gateway: LLMGateway
    dataset: str
    model_key: str | None = None  # generator override
    chunker: str | None = None  # default: configs/retrieval.yaml dataset_chunker
    embedder: str | None = None  # default: reference embedder
    reranker: str | None = None  # default: reference reranker
    use_cache: bool = True  # False for variance repeats
    extractor: str | None = None  # graph pipelines: extraction model key (default: role)
    graph_backend: str | None = None  # graph pipelines: configs/graph.yaml backend
    context_token_budget: int | None = None  # ablation A3 (default: configs/retrieval.yaml)
    # Injected models (tests); real runs load them from configs/retrieval.yaml.
    embedder_model: Embedder | None = None
    reranker_model: Reranker | None = None


def _search_settings(ctx: PipelineContext) -> SearchSettings:
    settings = ctx.config.retrieval.search
    if ctx.context_token_budget is None:
        return settings
    return settings.model_copy(update={"context_token_budget": ctx.context_token_budget})


def _closed_book(ctx: PipelineContext) -> Pipeline:
    return ClosedBookPipeline(ctx.gateway, model_key=ctx.model_key, use_cache=ctx.use_cache)


def _vector(variant: str) -> Callable[[PipelineContext], Pipeline]:
    spec = VARIANTS[variant]

    def factory(ctx: PipelineContext) -> Pipeline:
        retrieval = ctx.config.retrieval
        embedder = ctx.embedder_model
        if embedder is None and spec.retriever != "bm25":
            name = ctx.embedder or retrieval.reference_embedder
            embedder = load_embedder(name, retrieval.embedders[name])
        reranker = ctx.reranker_model
        if reranker is None and spec.rerank:
            name = ctx.reranker or retrieval.reference_reranker
            reranker = load_reranker(name, retrieval.rerankers[name])
        searcher = Searcher(
            ctx.config,
            ctx.dataset,
            ctx.chunker or default_chunker(ctx.config, ctx.dataset),
            spec,
            embedder=embedder,
            reranker=reranker,
            settings=_search_settings(ctx),
        )
        return VectorPipeline(
            variant, ctx.gateway, searcher, model_key=ctx.model_key, use_cache=ctx.use_cache
        )

    return factory


def _graph(mode: str) -> Callable[[PipelineContext], Pipeline]:
    def factory(ctx: PipelineContext) -> Pipeline:
        config = ctx.config
        retrieval = config.retrieval
        embedder_name = (
            ctx.embedder_model.name
            if ctx.embedder_model is not None
            else ctx.embedder or retrieval.reference_embedder
        )
        location = GraphLocation(
            dataset=ctx.dataset,
            chunker=ctx.chunker or default_chunker(config, ctx.dataset),
            extractor=ctx.extractor or default_extractor(config),
            embedder=embedder_name,
            backend=ctx.graph_backend or config.graph.reference_backend,
        )
        load_graph_manifest(config, location)  # fail fast (before loading models) if not built
        embedder = ctx.embedder_model or load_embedder(
            embedder_name, retrieval.embedders[embedder_name]
        )
        reranker = ctx.reranker_model
        if reranker is None and config.graph.query.rerank_chunks:
            name = ctx.reranker or retrieval.reference_reranker
            reranker = load_reranker(name, retrieval.rerankers[name])
        graph = GraphIndex(config, location, gateway=ctx.gateway, embedder=embedder)
        return GraphPipeline(
            mode,
            config,
            ctx.gateway,
            graph,
            reranker=reranker,
            model_key=ctx.model_key,
            use_cache=ctx.use_cache,
            settings=_search_settings(ctx),
        )

    return factory


PIPELINES: dict[str, Callable[[PipelineContext], Pipeline]] = {
    "v0": _closed_book,
    **{variant: _vector(variant) for variant in VARIANTS},
    **{f"lr_{mode}": _graph(mode) for mode in GRAPH_MODES},
}


def build_pipeline(name: str, ctx: PipelineContext) -> Pipeline:
    if name not in PIPELINES:
        raise KeyError(f"unknown pipeline '{name}'; choose from {sorted(PIPELINES)}")
    return PIPELINES[name](ctx)
